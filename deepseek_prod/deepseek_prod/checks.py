"""
checks.py — Chứng minh từng tính năng của Cấp 4 bằng số, không bằng lời.

Nguyên tắc: mọi kỹ thuật ở đây đều phải cho ra KẾT QUẢ Y HỆT cách làm
thường. Nếu cắt model ra 2 máy mà logits khác đi thì cắt làm gì.

    python -m deepseek_prod check          chạy hết
    python -m deepseek_prod check tp       chỉ chạy phần tensor parallel
"""

from __future__ import annotations

import torch

from .base_model import DeepSeekLite, Tokenizer
from .config import ParallelConfig
from .expert_parallel import expert_parallelize
from .launcher import run_distributed
from .parallel import Parallel
from .paths import LEVEL3_CHECKPOINT
from .paged_cache import PagedKVCache
from .pipeline import PipelineStage, run_pipeline_batch
from .tensor_parallel import tensor_parallelize
from .zero import ShardedOptimizer


def load_level3():
    """Đọc model đã học của Cấp 3. Cấp 4 không huấn luyện lại từ đầu."""
    if not LEVEL3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Chưa có model Cấp 3 ở {LEVEL3_CHECKPOINT}.\n"
            "Vào deepseek_lite/ chạy:  python -m deepseek_lite train"
        )

    checkpoint = torch.load(LEVEL3_CHECKPOINT, map_location="cpu")
    from deepseek_lite import Config

    config = Config.from_dict(checkpoint["config"])
    tokenizer = Tokenizer.load(checkpoint["chars"])

    model = DeepSeekLite(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return config, tokenizer, model


def sample_tokens(tokenizer, text="Chuyên gia giỏi toán trả lời câu hỏi.") -> list[int]:
    ids = tokenizer.encode(text)[:16]
    return ids or [0] * 4


# ======================================================================
# 1. Tensor parallel
# ======================================================================


def _tp_worker(rank, world, queue, tokens):
    parallel = Parallel(ParallelConfig(tp=world))
    _, _, model = load_level3()
    model = tensor_parallelize(model, parallel)
    model.eval()

    inputs = torch.tensor([tokens])

    with torch.no_grad():
        logits, loss = model(inputs, inputs)

    queue.put(
        {
            "rank": rank,
            "logits": logits.tolist(),
            "loss": loss.item(),
            "params": sum(p.numel() for p in model.parameters()),
            "heads": model.blocks[0].attention.n_heads,
            "kv_heads": model.blocks[0].attention.n_kv_heads,
        }
    )


def check_tensor_parallel(world: int = 2) -> dict:
    """Cắt ngang ra `world` máy — logits phải giống hệt chạy một máy."""
    _, tokenizer, baseline = load_level3()
    tokens = sample_tokens(tokenizer)

    with torch.no_grad():
        want, _ = baseline(torch.tensor([tokens]))

    results = run_distributed(_tp_worker, world, tokens, timeout=300)
    got = torch.tensor(results[0]["logits"])

    diff = (want - got).abs().max().item()

    return {
        "name": f"Tensor parallel ({world} máy)",
        "ok": diff < 1e-4,
        "details": [
            f"lệch logits lớn nhất: {diff:.2e}  (cần < 1e-4)",
            f"thông số mỗi máy   : {results[0]['params']:,}",
            f"đầu Q / K-V mỗi máy: {results[0]['heads']} / {results[0]['kv_heads']}"
            f"  (một máy: {baseline.config.n_heads} / {baseline.config.n_kv_heads})",
        ],
    }


# ======================================================================
# 2. Pipeline parallel
# ======================================================================

BATCH = 4  # pipeline cần chia lô thành nhiều vi lô, nên lô phải > 1


def _gradient_summary(module, owned=None) -> dict:
    """Một con số cho mỗi tham số, để so gradient giữa các cách chạy.

    `owned(name)` cho biết máy này có thật sự sở hữu tham số đó không.

    Phải lọc chứ không thể lấy cả model: bảng tra dùng chung giữa embedding
    và LM Head, nên nếu máy nào cũng báo cáo cả model thì hai máy báo cùng
    một tên `embedding.weight` và bản ghi sau đè lên bản ghi trước — phép so
    sẽ sai một cách âm thầm.
    """
    return {
        name: round(param.grad.abs().sum().item(), 6)
        for name, param in module.named_parameters()
        if (owned is None or owned(name))
        and param.grad is not None
        and param.grad.abs().sum() > 0
    }


def _pp_worker(rank, world, queue, tokens):
    parallel = Parallel(ParallelConfig(pp=world))

    _, _, model = load_level3()
    model.train()
    model.zero_grad()

    stage = PipelineStage(model, parallel)
    inputs = torch.tensor([tokens] * BATCH)

    loss = run_pipeline_batch(stage, inputs, inputs, micro_batches=2)

    # Chỉ báo cáo tham số mà máy NÀY giữ. Tên phải theo chỉ số tầng gốc,
    # vì PipelineStage dựng ModuleList mới nên tầng 2 sẽ thành "blocks.0".
    lo, hi = stage.layer_range

    def owned(name: str) -> bool:
        if name.startswith("blocks."):
            return lo <= int(name.split(".")[1]) < hi
        if name.startswith("embedding."):
            return stage.first
        if name.startswith(("norm.", "lm_head.")):
            return stage.last
        return False

    queue.put(
        {
            "rank": rank,
            "loss": loss,
            "layers": list(stage.layer_range),
            "first": stage.first,
            "last": stage.last,
            "grads": _gradient_summary(model, owned),
            "params": sum(p.numel() for p in model.parameters()),
        }
    )


def check_pipeline(world: int = 2) -> dict:
    """Cắt dọc ra `world` máy — loss và gradient phải giống hệt chạy một máy."""
    _, tokenizer, baseline = load_level3()
    tokens = sample_tokens(tokenizer)

    baseline.train()
    baseline.zero_grad()
    inputs = torch.tensor([tokens] * BATCH)
    _, want_loss = baseline(inputs, inputs)
    want_loss.backward()
    want_grads = _gradient_summary(baseline)

    results = run_distributed(_pp_worker, world, tokens, timeout=300)

    # Mỗi máy giữ một đoạn nên chỉ có gradient của đoạn mình; gộp lại thành đủ bộ.
    got_grads: dict[str, float] = {}
    for result in results:
        got_grads.update(result["grads"])

    loss_diff = abs(want_loss.item() - results[-1]["loss"])

    worst_name, worst = "", 0.0
    for name, want in want_grads.items():
        if name not in got_grads:
            worst_name, worst = name, float("inf")
            break
        diff = abs(want - got_grads[name]) / max(abs(want), 1e-6)
        if diff > worst:
            worst_name, worst = name, diff

    layout = ", ".join(f"máy {r['rank']}: tầng {r['layers'][0]}-{r['layers'][1] - 1}" for r in results)

    return {
        "name": f"Pipeline parallel ({world} máy)",
        "ok": loss_diff < 1e-4 and worst < 1e-3,
        "details": [
            f"chia tầng          : {layout}",
            f"lệch loss          : {loss_diff:.2e}  (cần < 1e-4)",
            f"lệch gradient nhất : {worst:.2e} ở {worst_name or '(không có)'}  (cần < 1e-3)",
        ],
    }


# ======================================================================
# 3. Expert parallel
# ======================================================================


def _ep_worker(rank, world, queue, tokens):
    parallel = Parallel(ParallelConfig(tp=world, ep=world))

    _, _, model = load_level3()
    model = expert_parallelize(model, parallel)
    model.eval()

    inputs = torch.tensor([tokens])

    with torch.no_grad():
        logits, loss = model(inputs, inputs)

    held = sum(p.numel() for p in model.parameters())
    experts = [expert for block in model.blocks for expert in block.moe.experts]

    queue.put(
        {
            "rank": rank,
            "logits": logits.tolist(),
            "loss": loss.item(),
            "params": held,
            "experts_held": len(experts),
            "expert_span": (model.blocks[0].moe.start, model.blocks[0].moe.end),
        }
    )


def check_expert_parallel(world: int = 2) -> dict:
    """Chia chuyên gia ra `world` máy — logits phải giống hệt chạy một máy."""
    _, tokenizer, baseline = load_level3()
    tokens = sample_tokens(tokenizer)

    with torch.no_grad():
        want, _ = baseline(torch.tensor([tokens]))

    results = run_distributed(_ep_worker, world, tokens, timeout=300)
    got = torch.tensor(results[0]["logits"])
    diff = (want - got).abs().max().item()

    total_experts = baseline.config.n_experts
    spans = ", ".join(
        f"máy {r['rank']}: chuyên gia {r['expert_span'][0]}-{r['expert_span'][1] - 1}" for r in results
    )

    return {
        "name": f"Expert parallel ({world} máy)",
        "ok": diff < 1e-4,
        "details": [
            f"chia chuyên gia    : {spans}  (tổng {total_experts})",
            f"mỗi máy giữ        : {results[0]['experts_held'] // baseline.config.n_layers}"
            f" trong {total_experts} chuyên gia mỗi tầng",
            f"lệch logits lớn nhất: {diff:.2e}  (cần < 1e-4)",
        ],
    }


# ======================================================================
# 4. ZeRO-1 — chia trạng thái optimizer
# =====================================================================


def _zero_worker(rank, world, queue, tokens):
    parallel = Parallel(ParallelConfig(dp=world))

    _, _, model = load_level3()
    model.train()

    optimizer = ShardedOptimizer(model.parameters(), parallel, lr=1e-2, weight_decay=0.0)

    # Mọi máy dùng CÙNG một lô, nên sau khi lấy trung bình gradient thì
    # kết quả phải y hệt chạy một máy với đúng lô đó.
    inputs = torch.tensor([tokens])

    for _ in range(3):
        optimizer.zero_grad()
        _, loss = model(inputs, inputs)
        loss.backward()
        optimizer.step()

    queue.put(
        {
            "rank": rank,
            "state_bytes": optimizer.state_bytes,
            "shard": [optimizer.start, optimizer.count],
            "total_params": len(optimizer.params),
            # Vài tham số đại diện để so với cách chạy thường.
            "probe": {
                "embed": model.embedding.weight[0, :4].tolist(),
                "attn": model.blocks[0].attention.q_proj.weight[0, :4].tolist(),
                "moe": model.blocks[0].moe.experts[0].gate.weight[0, :4].tolist(),
            },
            "loss": loss.item(),
        }
    )


def check_zero(world: int = 2) -> dict:
    """ZeRO-1 phải cho ra model Y HỆT cách chạy thường, nhưng tốn ít bộ nhớ hơn."""
    _, tokenizer, baseline = load_level3()
    tokens = sample_tokens(tokenizer)

    baseline.train()
    plain = torch.optim.AdamW(baseline.parameters(), lr=1e-2, weight_decay=0.0)
    inputs = torch.tensor([tokens])

    for _ in range(3):
        plain.zero_grad()
        _, loss = baseline(inputs, inputs)
        loss.backward()
        plain.step()

    plain_state = sum(
        value.numel() * value.element_size()
        for state in plain.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )

    results = run_distributed(_zero_worker, world, tokens, timeout=300)

    want = {
        "embed": baseline.embedding.weight[0, :4].tolist(),
        "attn": baseline.blocks[0].attention.q_proj.weight[0, :4].tolist(),
        "moe": baseline.blocks[0].moe.experts[0].gate.weight[0, :4].tolist(),
    }

    worst = 0.0
    for key, values in want.items():
        for a, b in zip(values, results[0]["probe"][key]):
            worst = max(worst, abs(a - b))

    shards = ", ".join(f"máy {r['rank']}: {r['shard'][1]} tham số" for r in results)

    return {
        "name": f"ZeRO-1 ({world} máy)",
        "ok": worst < 1e-5,
        "details": [
            f"chia tham số       : {shards}  (tổng {results[0]['total_params']})",
            f"trạng thái mỗi máy : {results[0]['state_bytes'] / 1024:.0f} KB"
            f"   (chạy thường: {plain_state / 1024:.0f} KB)",
            f"tiết kiệm          : {plain_state / max(results[0]['state_bytes'], 1):.1f} lần",
            f"model sau 3 bước giống hệt: lệch {worst:.2e}  (cần < 1e-5)",
        ],
    }


# ======================================================================
# 5. Paged KV cache
# ======================================================================


def check_paged_cache() -> dict:
    """Đệm chia khối phải cho ra logits y hệt đệm liền mạch của Cấp 3."""
    from deepseek_lite import KVCache

    _, tokenizer, model = load_level3()
    tokens = sample_tokens(tokenizer, "Máy tính không hiểu như con người.")
    inputs = torch.tensor([tokens])

    plain_cache = KVCache(model.config, batch_size=1)
    with torch.no_grad():
        want, _ = model(inputs, cache=plain_cache, start_pos=0)

    paged = PagedKVCache(model.config, num_blocks=16, block_size=8)
    paged.add_sequence(0)
    paged.begin_batch([0])
    with torch.no_grad():
        got, _ = model(inputs, cache=paged, start_pos=0)

    diff = (want - got).abs().max().item()

    # So sánh công bằng: CÙNG một lượng bộ nhớ, cách nào phục vụ được nhiều
    # câu hơn? Lấy mốc là 4 câu theo cách thường.
    turns = 40  # một câu chuyện dài trung bình 40 token
    mono = 4  # 4 câu theo cách thường

    slots_plain = mono * model.config.max_seq_len
    blocks_needed = -(-turns // paged.block_size)  # làm tròn lên
    slots_paged = blocks_needed * paged.block_size
    how_many_paged = slots_plain // slots_paged

    return {
        "name": "Paged KV cache",
        "ok": diff < 1e-4,
        "details": [
            f"lệch logits so với đệm liền mạch: {diff:.2e}  (cần < 1e-4)",
            f"khối đang dùng     : {paged.blocks_in_use}/{paged.num_blocks}"
            f"  (hao hụt trong khối: {paged.waste_bytes / 1024:.1f} KB)",
            f"cùng {slots_plain} chỗ chứa token:",
            f"  cách thường      : {mono} câu"
            f"  (mỗi câu xin sẵn {model.config.max_seq_len} token, dùng 40)",
            f"  chia khối {paged.block_size:>2} token: {how_many_paged} câu"
            f"  (mỗi câu tốn {blocks_needed} khối = {slots_paged} token)",
            f"  -> phục vụ được gấp {how_many_paged / mono:.1f} lần số câu cùng lúc",
        ],
    }


# ======================================================================
# 6. Continuous batching
# ======================================================================

PROMPTS = [
    "Học mãi thì",
    "Chuyên gia giỏi toán",
    "Máy tính không hiểu",
    "Attention giúp tôi",
    "Bé cũng làm được",
    "Đoán sai thì tôi",
]


def _run_engine(engine_class, model, tokenizer, max_batch):
    from .engine import Request

    cache = PagedKVCache(model.config, num_blocks=32, block_size=16)
    engine = engine_class(model, tokenizer, cache, max_batch=max_batch)

    for index, prompt in enumerate(PROMPTS):
        engine.submit(
            Request(
                request_id=index,
                prompt_ids=tokenizer.encode(prompt),
                max_new_tokens=12,
                temperature=0.0,  # chọn chữ chắc chắn nhất -> chạy lại phải ra y hệt
            )
        )

    finished = engine.run()
    return engine.elapsed, {r.request_id: tokenizer.decode(r.output) for r in finished}


def _first_decode_logits(model, prompts: list[list[int]], batched: bool) -> dict:
    """Đọc câu mở đầu cho từng câu, rồi so điểm ở bước sinh chữ ĐẦU TIÊN.

    So logits chứ không so ký tự: gom lô làm đệm phải lót cho bằng nhau,
    nên thứ tự cộng số thực có đổi chút. Ở chỗ hai chữ gần bằng điểm nhau
    thì chỉ cần lệch 1e-6 là kết quả đã lật — đó là chuyện bình thường của
    suy luận theo lô, không phải lỗi. So logits mới thấy đúng bản chất.
    """
    cache = PagedKVCache(model.config, num_blocks=32, block_size=16)
    for index in range(len(prompts)):
        cache.add_sequence(index)

    first_tokens = []
    for index, ids in enumerate(prompts):
        cache.begin_batch([index])
        with torch.no_grad():
            logits, _ = model(torch.tensor([ids]), cache=cache, start_pos=0)
        first_tokens.append(int(logits[:, -1, :].argmax(-1).item()))

    out: dict[int, torch.Tensor] = {}

    if batched:
        cache.begin_batch(list(range(len(prompts))))
        positions = torch.tensor([len(ids) for ids in prompts])
        tokens = torch.tensor([[token] for token in first_tokens])

        with torch.no_grad():
            logits, _ = model(tokens, cache=cache, start_pos=positions)

        for index in range(len(prompts)):
            out[index] = logits[index, -1, :]
    else:
        for index, ids in enumerate(prompts):
            cache.begin_batch([index])
            positions = torch.tensor([len(ids)])
            tokens = torch.tensor([[first_tokens[index]]])

            with torch.no_grad():
                logits, _ = model(tokens, cache=cache, start_pos=positions)

            out[index] = logits[0, -1, :]

    return out


def check_continuous_batching() -> dict:
    """Gom lô phải cho ra kết quả y hệt chạy lần lượt, nhưng nhanh hơn."""
    from .engine import ContinuousEngine, SequentialEngine

    _, tokenizer, model = load_level3()

    # 1. Đúng đắn: so logits giữa chạy chung lô và chạy riêng từng câu.
    prompts = [tokenizer.encode(p) for p in PROMPTS]
    together = _first_decode_logits(model, prompts, batched=True)
    alone = _first_decode_logits(model, prompts, batched=False)

    worst = max((together[i] - alone[i]).abs().max().item() for i in range(len(prompts)))

    # 2. Tốc độ.
    sequential_time, sequential_text = _run_engine(SequentialEngine, model, tokenizer, max_batch=1)
    batched_time, batched_text = _run_engine(ContinuousEngine, model, tokenizer, max_batch=4)

    same_text = sequential_text == batched_text
    speedup = sequential_time / max(batched_time, 1e-9)
    total_tokens = sum(len(tokenizer.encode(p)) + 12 for p in PROMPTS)

    return {
        "name": "Continuous batching",
        "ok": worst < 1e-4 and speedup > 1.2,
        "details": [
            f"{len(PROMPTS)} câu hỏi, mỗi câu sinh 12 chữ, lô tối đa 4",
            f"lệch logits khi chạy chung lô: {worst:.2e}  (cần < 1e-4)",
            f"chữ sinh ra giống hệt cách lần lượt: {'CÓ' if same_text else 'KHÁC (do lệch số thực, xem trên)'}",
            f"lần lượt từng câu : {sequential_time:.2f} giây"
            f"  ({total_tokens / sequential_time:.0f} token/giây)",
            f"gom lô            : {batched_time:.2f} giây"
            f"  ({total_tokens / batched_time:.0f} token/giây)",
            f"-> nhanh hơn {speedup:.1f} lần",
        ],
    }


# ======================================================================
# 7. Quantization — nén 8 bit
# ======================================================================


def check_quantization() -> dict:
    """Nén 8 bit: nhỏ hơn bao nhiêu, nhanh hơn bao nhiêu, sai đi bao nhiêu."""
    import time

    from .quantize import model_bytes, quantize_int8

    _, tokenizer, model = load_level3()
    model.eval()

    tokens = tokenizer.encode("Chuyên gia giỏi toán trả lời câu hỏi.")
    inputs = torch.tensor([tokens])

    with torch.no_grad():
        want, _ = model(inputs)

    full_bytes = model_bytes(model)

    # Đo tốc độ bản gốc trước khi nén (nén tại chỗ nên phải đo trước).
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
        started = time.perf_counter()
        for _ in range(10):
            model(inputs)
        full_time = (time.perf_counter() - started) / 10

    quantized = quantize_int8(model)
    quantized.eval()

    with torch.no_grad():
        got, _ = quantized(inputs)

    small_bytes = model_bytes(quantized)
    diff = (want - got).abs().max().item()

    # Chữ mà model chọn có đổi không?
    keep = (want.argmax(-1) == got.argmax(-1)).float().mean().item()

    with torch.no_grad():
        for _ in range(3):
            quantized(inputs)
        started = time.perf_counter()
        for _ in range(10):
            quantized(inputs)
        small_time = (time.perf_counter() - started) / 10

    return {
        "name": "Quantization (int8)",
        "ok": small_bytes < full_bytes * 0.5 and keep > 0.9,
        "details": [
            f"kích thước         : {full_bytes / 1024:.0f} KB -> {small_bytes / 1024:.0f} KB"
            f"  (nhỏ đi {full_bytes / small_bytes:.1f} lần)",
            f"lệch logits lớn nhất: {diff:.3f}  (nén thì phải lệch, không sao)",
            f"chữ được chọn giống : {keep:.1%}  (cần > 90%)",
            f"tốc độ             : {full_time * 1000:.0f} ms -> {small_time * 1000:.0f} ms"
            f"  ({full_time / small_time:.2f} lần)",
            "trên CPU, int8 chủ yếu giúp BỘ NHỚ, tốc độ có khi còn chậm hơn chút",
        ],
    }
