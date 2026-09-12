"""
analyze.py — Người phiên dịch giữa trang web (TypeScript) và ba model (PyTorch).

Studio xem được cả ba cấp:

    tiny_gpt/         Cấp 1 — Attention + MLP
    mini_deepseek/    Cấp 2 — thêm RoPE, SwiGLU, MoE
    deepseek_lite/    Cấp 3 — thêm GQA, cửa sổ trượt, KV cache, hyper-connections

Ba model được viết theo ba kiểu khác nhau (một file / nhiều file rời / package),
nên file này có một "bộ chuyển" cho mỗi model, đưa hết về cùng một hình dạng
dữ liệu để trang web chỉ phải hiểu một kiểu.

Chạy:
    python analyze.py --serve           # thường trực, đọc JSON từng dòng
    python analyze.py --models          # xem có những model nào
"""

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TINY_DIR = ROOT / "tiny_gpt"
MINI_DIR = ROOT / "mini_deepseek"
LITE_DIR = ROOT / "deepseek_lite"

# Cắt câu cho ngắn, để mỗi ô trong bản đồ attention còn đủ to mà nhìn.
MAX_TOKENS = 64

# Chạy bao nhiêu token khi gom "ký tự này hay hỏi chuyên gia nào".
# Càng nhiều thì càng nhiều ký tự được tô màu, nhưng chạy lâu hơn.
ROUTING_TOKENS = 2048


# =====================================================================
# Bộ chuyển cho từng model
# =====================================================================


def _build_tiny_gpt():
    """tiny_gpt cố tình để mọi con số ở cấp module (cho dễ đọc), nên phải
    gán lại chúng trước khi dựng model. Đó là cái giá của "một file duy nhất".
    """
    sys.path.insert(0, str(TINY_DIR))
    import tiny_gpt as tg  # noqa: PLC0415

    checkpoint = torch.load(TINY_DIR / "tiny_gpt.pt", map_location="cpu")

    tg.vocab_size = checkpoint["vocab_size"]
    tg.block_size = checkpoint["block_size"]
    tg.n_embd = checkpoint["n_embd"]
    tg.n_head = checkpoint["n_head"]
    tg.n_layer = checkpoint["n_layer"]

    model = tg.TinyGPT()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Bật ghi lại bản đồ attention.
    for block in model.blocks:
        block.attn.capture = True

    return {
        "model": model,
        "chars": checkpoint["chars"],
        "embedding": model.token_embedding.weight,
        "blocks": list(model.blocks),
        "attn_attr": "attn",
        "moe_attr": None,
        "ffn_attr": "mlp",
        "history": checkpoint.get("history") or [],
        "history_kind": "plain",
        "n_params": sum(p.numel() for p in model.parameters()),
        "about": {
            "nLayers": checkpoint["n_layer"],
            "nHeads": checkpoint["n_head"],
            "nKvHeads": checkpoint["n_head"],  # Cấp 1 chưa có GQA
            "nExperts": 0,
            "topK": 0,
            "nShared": 0,
            "nStreams": 1,
            "window": 0,
            "maxSeqLen": checkpoint["block_size"],
            "cacheBytes": 0,
        },
    }


def _build_mini_deepseek():
    sys.path.insert(0, str(MINI_DIR))
    from config import Config  # noqa: PLC0415
    from model import DeepSeekMini  # noqa: PLC0415

    checkpoint = torch.load(MINI_DIR / "mini_deepseek.pt", map_location="cpu")

    config = Config(**checkpoint["config"])
    config.capture = True

    model = DeepSeekMini(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "model": model,
        "chars": checkpoint["chars"],
        "embedding": model.embedding.weight,
        "blocks": list(model.blocks),
        "attn_attr": "attention",
        "moe_attr": "moe",
        "ffn_attr": "moe",
        "history": checkpoint.get("history") or [],
        "history_kind": "aux",
        "n_params": n_params,
        "about": {
            "nLayers": config.n_layers,
            "nHeads": config.n_heads,
            "nKvHeads": config.n_heads,  # Cấp 2 chưa có GQA
            "nExperts": config.n_experts,
            "topK": config.top_k,
            "nShared": 0,
            "nStreams": config.n_streams,
            "window": 0,
            "maxSeqLen": config.max_seq_len,
            "cacheBytes": 0,
        },
    }


def _build_deepseek_lite():
    sys.path.insert(0, str(LITE_DIR))
    from deepseek_lite import Config, DeepSeekLite  # noqa: PLC0415

    checkpoint = torch.load(LITE_DIR / "runs" / "model.pt", map_location="cpu")

    config = Config.from_dict(checkpoint["config"])
    config.capture = True

    model = DeepSeekLite(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    total, _ = model.parameter_counts()

    return {
        "model": model,
        "chars": checkpoint["chars"],
        "embedding": model.embedding.weight,
        "blocks": list(model.blocks),
        "attn_attr": "attention",
        "moe_attr": "moe",
        "ffn_attr": "moe",
        "history": checkpoint.get("history") or [],
        "history_kind": "balance",
        "n_params": total,
        "about": {
            "nLayers": config.n_layers,
            "nHeads": config.n_heads,
            "nKvHeads": config.n_kv_heads,
            "nExperts": config.n_experts,
            "topK": config.top_k,
            "nShared": config.n_shared,
            "nStreams": config.n_streams,
            "window": config.window,
            "maxSeqLen": config.max_seq_len,
            "cacheBytes": model.cache_memory_bytes(),
        },
    }


MODELS = {
    "tiny_gpt": {
        "label": "Cấp 1 — TinyGPT",
        "short": "Cấp 1",
        "checkpoint": TINY_DIR / "tiny_gpt.pt",
        "data": None,  # Cấp 1 không có MoE nên không cần gom routing
        "build": _build_tiny_gpt,
        "what": "Attention + MLP. Chưa có chuyên gia, chưa có RoPE.",
    },
    "mini_deepseek": {
        "label": "Cấp 2 — Mini DeepSeek",
        "short": "Cấp 2",
        "checkpoint": MINI_DIR / "mini_deepseek.pt",
        "data": MINI_DIR / "data.txt",
        "build": _build_mini_deepseek,
        "what": "Thêm RoPE, SwiGLU và 4 chuyên gia to.",
    },
    "deepseek_lite": {
        "label": "Cấp 3 — deepseek_lite",
        "short": "Cấp 3",
        "checkpoint": LITE_DIR / "runs" / "model.pt",
        "data": LITE_DIR / "data" / "data.txt",
        "build": _build_deepseek_lite,
        "what": "Thêm GQA, cửa sổ trượt, KV cache, 8 chuyên gia nhỏ + 1 dùng chung.",
    },
}


def available_models():
    """Danh sách model, kèm cho biết model nào đã học xong rồi."""
    return [
        {
            "id": name,
            "label": spec["label"],
            "short": spec["short"],
            "what": spec["what"],
            "available": spec["checkpoint"].exists(),
        }
        for name, spec in MODELS.items()
    ]


# =====================================================================
# Nạp model (giữ trong bộ nhớ, tự nạp lại khi file model thay đổi)
# =====================================================================

_loaded: dict[str, dict] = {}
_space_cache: dict[str, tuple[float, dict]] = {}


def load(name: str) -> dict:
    if name not in MODELS:
        raise ValueError(f"Không biết model {name!r}. Chọn một trong: {', '.join(MODELS)}")

    spec = MODELS[name]
    checkpoint = spec["checkpoint"]

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Chưa có model {name}: {checkpoint}\n"
            "Hãy học nó trước (xem README ở thư mục gốc)."
        )

    # Học lại xong thì chỉ cần tải lại trang, không phải khởi động lại máy chủ.
    mtime = checkpoint.stat().st_mtime
    cached = _loaded.get(name)
    if cached and cached["mtime"] == mtime:
        return cached

    built = spec["build"]()
    built["mtime"] = mtime
    built["name"] = name
    _loaded[name] = built
    return built


# =====================================================================
# Dữ liệu cho trang web
# =====================================================================


def _loss_data(loaded: dict) -> dict:
    """Đưa lịch sử học của ba model về cùng một hình dạng.

    Mỗi cấp lưu lịch sử hơi khác nhau:
        Cấp 1:  [bước, loss]
        Cấp 2:  [bước, loss, aux loss]
        Cấp 3:  [bước, loss, lr, độ lệch chia việc]
    """
    history = loaded["history"]
    kind = loaded["history_kind"]

    if not history:
        return {"step": [], "loss": [], "extra": [], "extraName": ""}

    data = {
        "step": [row[0] for row in history],
        "loss": [row[1] for row in history],
        "extra": [],
        "extraName": "",
    }

    if kind == "aux":
        data["extra"] = [row[2] for row in history]
        data["extraName"] = "aux loss — trung bình mỗi tầng (2,0 = chia việc đều)"
    elif kind == "balance":
        data["extra"] = [row[3] for row in history]
        data["extraName"] = "độ lệch chia việc (1,00 = chia đều hoàn hảo)"

    return data


def meta(name: str) -> dict:
    loaded = load(name)
    spec = MODELS[name]

    return {
        "id": name,
        "label": spec["label"],
        "what": spec["what"],
        "models": available_models(),
        "vocabSize": len(loaded["chars"]),
        "nParams": loaded["n_params"],
        "loss": _loss_data(loaded),
        **loaded["about"],
    }


def space(name: str) -> dict:
    """Không gian embedding: mỗi ký tự là một điểm trong không gian 3 chiều.

    Bảng embedding của model rộng 96–192 chiều. Con người không nhìn được
    192 chiều, nên ta chiếu xuống 3 chiều bằng PCA — giữ lại 3 hướng mà
    bảng embedding "trải ra" nhiều nhất.

    Kết quả được nhớ lại, vì gom routing cho cả bài học hơi lâu.
    """
    loaded = load(name)

    cached = _space_cache.get(name)
    if cached and cached[0] == loaded["mtime"]:
        return cached[1]

    weight = loaded["embedding"].detach().float()  # [vocab, d_model]

    centered = weight - weight.mean(dim=0, keepdim=True)
    _, singular, directions = torch.linalg.svd(centered, full_matrices=False)

    coords = centered @ directions[:3].T  # [vocab, 3]

    # Thu nhỏ về khoảng [-1, 1] cho dễ nhìn. Dùng phân vị 98 để một điểm
    # quá xa không làm cả đám còn lại bẹp dí.
    scale = torch.quantile(coords.abs().flatten(), 0.98).item() or 1.0
    coords = (coords / scale).clamp(-1.5, 1.5)

    total_variance = (singular**2).sum()
    explained = ((singular[:3] ** 2) / total_variance).tolist()

    dominant, share, count = _expert_per_char(loaded)

    points = [
        {
            "ch": ch,
            "x": round(coords[i, 0].item(), 4),
            "y": round(coords[i, 1].item(), 4),
            "z": round(coords[i, 2].item(), 4),
            "expert": int(dominant[i]),
            "share": round(float(share[i]), 3),
            "count": int(count[i]),
        }
        for i, ch in enumerate(loaded["chars"])
    ]

    result = {
        "model": name,
        "nExperts": loaded["about"]["nExperts"],
        "dModel": weight.shape[1],
        "explained": [round(v, 4) for v in explained],
        "points": points,
    }

    _space_cache[name] = (loaded["mtime"], result)
    return result


@torch.no_grad()
def _expert_per_char(loaded: dict):
    """Ký tự này hay hỏi chuyên gia nào?

    Chạy model trên bài học của chính nó, gom lại xem mỗi ký tự được router
    gửi tới chuyên gia nào nhiều nhất. Nhờ vậy mới tô màu được không gian
    embedding và thấy chuyên gia có chuyên môn hoá theo vùng nghĩa không.
    """
    vocab_size = len(loaded["chars"])
    n_experts = loaded["about"]["nExperts"]

    counts = torch.zeros(vocab_size, max(n_experts, 1))

    if loaded["moe_attr"] and n_experts:
        data_path = MODELS[loaded["name"]]["data"]
        if data_path and data_path.exists():
            text = data_path.read_text(encoding="utf-8")
            stoi = {ch: i for i, ch in enumerate(loaded["chars"])}
            ids = [stoi[ch] for ch in text if ch in stoi][:ROUTING_TOKENS]

            top_k = loaded["about"]["topK"]
            window = loaded["about"]["maxSeqLen"]

            # Model chỉ nhận được `max_seq_len` token một lần, nên cắt bài
            # thành nhiều đoạn rồi chạy lần lượt. Nhờ vậy phủ được nhiều
            # ký tự hơn là chỉ chạy đoạn đầu.
            for start in range(0, len(ids), window):
                chunk = ids[start : start + window]
                if len(chunk) < 2:
                    continue

                tokens = torch.tensor([chunk])
                loaded["model"](tokens)

                for block in loaded["blocks"]:
                    moe = getattr(block, loaded["moe_attr"])
                    if moe.routing is None:
                        continue
                    indices, weights = moe.routing  # [1, T, top_k]

                    flat_expert = indices[0].reshape(-1)  # [T * top_k]
                    flat_weight = weights[0].reshape(-1)
                    flat_token = tokens[0].unsqueeze(-1).expand(-1, top_k).reshape(-1)

                    contribution = torch.zeros(len(flat_expert), n_experts)
                    contribution.scatter_(1, flat_expert.unsqueeze(1), flat_weight.unsqueeze(1))
                    counts.index_add_(0, flat_token, contribution)

    total = counts.sum(dim=1)
    seen = total > 0

    dominant = torch.full((vocab_size,), -1, dtype=torch.long)
    share = torch.zeros(vocab_size)

    if seen.any():
        dominant[seen] = counts[seen].argmax(dim=1)
        share[seen] = counts[seen].max(dim=1).values / total[seen]

    return dominant, share, total


# =====================================================================
# Kiến trúc model — để vẽ sơ đồ 3D
# =====================================================================
#
# Cách bày biện:
#     trục X  = dòng chảy, đọc từ trái sang phải
#     trục Z  = tầng thứ mấy (mỗi tầng lùi ra sau một quãng)
#     trục Y  = nhánh: đường tắt (residual) vòng LÊN TRÊN,
#               chuyên gia toả XUỐNG DƯỚI
#
# Nhờ trục Z mà nhìn nghiêng là thấy ngay model "sâu" bao nhiêu tầng —
# thứ mà sơ đồ 2D không thể hiện được.

RESIDUAL_Y = 1.05  # đường tắt vòng lên trên
BRANCH_Y = -1.25  # chuyên gia toả xuống dưới
LAYER_STEP_Z = 2.7  # mỗi tầng lùi ra sau bao nhiêu


def architecture(name: str) -> dict:
    """Sơ đồ kiến trúc kèm toạ độ 3D.

    Python tính toạ độ vì Python là bên biết model có những bộ phận gì;
    trang web chỉ việc vẽ theo.
    """
    loaded = load(name)
    about = loaded["about"]

    n_layers = about["nLayers"]
    n_experts = about["nExperts"]
    n_shared = about["nShared"]
    top_k = about["topK"]
    n_heads = about["nHeads"]
    n_kv = about["nKvHeads"]
    window = about["window"]
    has_moe = n_experts > 0

    nodes: list[dict] = []
    edges: list[dict] = []

    # Đường trục: dãy nút mà dòng chảy chính đi qua, theo đúng thứ tự.
    # Trang web dùng nó để thả mấy hạt sáng chạy dọc model, cho thấy
    # thông tin đi theo chiều nào.
    spine: list[str] = ["embed"]

    def node(node_id, label, kind, x, y, z, detail, params=0):
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "kind": kind,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "detail": detail,
                "params": int(params),
            }
        )

    def link(source, target, kind="data"):
        edges.append({"from": source, "to": target, "kind": kind})

    def count(module):
        return sum(p.numel() for p in module.parameters())

    # --- Vị trí các bước trong một tầng --------------------------------
    if has_moe:
        step = {"norm1": 0.0, "attn": 1.05, "add1": 2.0, "norm2": 2.9, "add2": 6.25}
        router_x, expert_x, combine_x = 3.65, 4.7, 5.6
    else:
        step = {"norm1": 0.0, "attn": 1.05, "add1": 2.0, "norm2": 2.9, "add2": 5.0}
        router_x = expert_x = combine_x = 0.0

    span_x = step["add2"] + 1.4

    # --- Đầu vào -------------------------------------------------------
    embedding = loaded["embedding"]
    d_model = embedding.shape[1]

    node(
        "embed",
        "Token Embedding",
        "embed",
        -2.4,
        0,
        0,
        f"Đổi mỗi ký tự thành một vector {d_model} chiều. Đây là bảng tra duy nhất "
        "của cả model — LM Head ở cuối dùng lại chính bảng này (weight tying), "
        "nên không tốn thêm thông số.",
        embedding.numel(),
    )

    inputs = ["embed"]

    position = getattr(loaded["model"], "position_embedding", None)
    if position is not None:
        node(
            "pos",
            "Position Embedding",
            "embed",
            -2.4,
            -1.2,
            0,
            f"Cho model biết token đang ở vị trí thứ mấy (tối đa {position.weight.shape[0]} vị trí). "
            "Cộng vào Token Embedding trước khi vào tầng đầu. "
            "Cấp 2 và Cấp 3 không cần bảng này vì dùng RoPE — xoay vector theo vị trí.",
            position.weight.numel(),
        )
        inputs.append("pos")

    # --- Các tầng ------------------------------------------------------
    norm_name = type(loaded["blocks"][0].norm1).__name__

    attn_extra = []
    if n_kv != n_heads:
        attn_extra.append(f"{n_heads} đầu Q dùng chung {n_kv} đầu K/V (GQA)")
    if window:
        attn_extra.append(f"cửa sổ trượt {window} token")

    for i in range(n_layers):
        base = i * span_x
        z = i * LAYER_STEP_Z
        block = loaded["blocks"][i]

        previous = f"L{i - 1}.add2" if i > 0 else None
        sources = [previous] if previous else inputs

        n1, at, a1, n2, a2 = (f"L{i}.{k}" for k in ("norm1", "attn", "add1", "norm2", "add2"))
        spine += [n1, at, a1, n2]

        node(
            n1,
            norm_name,
            "norm",
            base + step["norm1"],
            0,
            z,
            "Kéo các con số về cùng một thang trước khi đưa vào Attention. "
            "Không có nó thì tầng càng sâu, giá trị càng phình ra.",
            count(block.norm1),
        )

        node(
            at,
            "Attention",
            "attention",
            base + step["attn"],
            0,
            z,
            f"{n_heads} đầu cùng nhìn. Mỗi token tự hỏi: trong các token đứng trước, "
            f"token nào giống mình nhất? Token KHÔNG được nhìn về tương lai. "
            + (" ".join(attn_extra) + "." if attn_extra else ""),
            count(getattr(block, loaded["attn_attr"])),
        )

        node(
            a1,
            "Cộng",
            "add",
            base + step["add1"],
            RESIDUAL_Y,
            z,
            "Lấy bài cũ CỘNG THÊM kết quả Attention. Đây là đường tắt (residual): "
            "mỗi tầng chỉ học 'ghi thêm một chút' chứ không phải học lại từ đầu."
            + (
                f" Ở model này có {about['nStreams']} dòng suy nghĩ song song, "
                "mỗi dòng nhận kết quả trả về với một trọng số riêng."
                if about["nStreams"] > 1
                else ""
            ),
            0,
        )

        node(
            n2,
            norm_name,
            "norm",
            base + step["norm2"],
            0,
            z,
            "Lại kéo về cùng một thang, lần này là trước khi đưa vào phần suy nghĩ.",
            count(block.norm2),
        )

        if has_moe:
            moe = getattr(block, loaded["ffn_attr"])

            node(
                f"L{i}.router",
                "Router",
                "router",
                base + router_x,
                -0.35,
                z,
                f"Xem từng token rồi cho điểm {n_experts} chuyên gia. Chỉ {top_k} chuyên gia "
                "điểm cao nhất được gọi. Điểm thiên vị của router được chỉnh sau mỗi "
                "bước học để việc chia đều quay lại — không cần thêm hàm phạt vào loss.",
                count(moe.router),
            )

            for e in range(n_experts):
                spread = (e - (n_experts - 1) / 2) / max(n_experts - 1, 1)
                node(
                    f"L{i}.expert{e}",
                    f"CG{e}",
                    "expert",
                    base + expert_x,
                    BRANCH_Y,
                    z + spread * 2.4,
                    f"Chuyên gia {e} — một SwiGLU nhỏ. Chỉ nhận những token mà router "
                    f"chọn nó. Mỗi token chỉ hỏi {top_k} trong {n_experts} chuyên gia, "
                    "nên phần lớn thông số không được dùng cùng lúc.",
                    count(moe.experts[e]),
                )
                link(f"L{i}.router", f"L{i}.expert{e}", "route")
                link(f"L{i}.expert{e}", f"L{i}.combine", "combine")

            if n_shared:
                node(
                    f"L{i}.shared",
                    "Dùng chung",
                    "shared",
                    base + router_x - 0.1,
                    BRANCH_Y,
                    z - 1.9,
                    "Chuyên gia dùng chung — token nào cũng hỏi, không qua router. "
                    "Nó lo những việc cơ bản như giữ cho câu văn trôi chảy, "
                    "để các chuyên gia riêng rảnh tay lo việc chuyên môn.",
                    count(moe.shared_experts[0]),
                )
                link(n2, f"L{i}.shared", "route")
                link(f"L{i}.shared", f"L{i}.combine", "combine")

            node(
                f"L{i}.combine",
                "Cộng lại",
                "add",
                base + combine_x,
                -0.35,
                z,
                f"Cộng kết quả của {top_k} chuyên gia được chọn"
                + (" và chuyên gia dùng chung" if n_shared else "")
                + ", theo đúng trọng số mà router đã cho.",
                0,
            )

            for source in sources:
                link(source, n1)
                link(source, a1, "residual")
            link(n1, at)
            link(at, a1)
            link(a1, n2)
            link(n2, f"L{i}.router")
            link(f"L{i}.combine", a2)
            link(a1, a2, "residual")

            spine += [f"L{i}.router", f"L{i}.combine", a2]

        else:
            node(
                f"L{i}.mlp",
                "MLP",
                "ffn",
                base + (step["norm2"] + step["add2"]) / 2,
                0,
                z,
                "Suy nghĩ riêng: mở rộng ra 4 lần rồi thu lại. "
                "Attention lo việc các token trao đổi với nhau, còn MLP là chỗ "
                "model tự tính toán một mình. Cấp 1 chỉ có một MLP chung cho mọi token.",
                count(getattr(block, loaded["ffn_attr"])),
            )

            for source in sources:
                link(source, n1)
                link(source, a1, "residual")
            link(n1, at)
            link(at, a1)
            link(a1, n2)
            link(n2, f"L{i}.mlp")
            link(f"L{i}.mlp", a2)
            link(a1, a2, "residual")

            spine += [f"L{i}.mlp", a2]

    # --- Đầu ra --------------------------------------------------------
    last = f"L{n_layers - 1}.add2"
    out_x = n_layers * span_x + 0.4

    node(
        "final_norm",
        norm_name,
        "norm",
        out_x,
        0,
        0,
        "Chuẩn hoá lần cuối trước khi phát biểu."
        + (f" Cũng là chỗ gộp {about['nStreams']} dòng suy nghĩ lại thành một." if about["nStreams"] > 1 else ""),
        count(loaded["model"].norm),
    )

    node(
        "lm_head",
        "LM Head",
        "head",
        out_x + 1.5,
        0,
        0,
        f"Biến vector {d_model} chiều thành điểm cho cả {len(loaded['chars'])} ký tự. "
        "Ký tự nào điểm cao nhất thì đó là chữ tiếp theo. "
        "Trọng số của lớp này CHÍNH LÀ bảng Token Embedding ở đầu — dùng chung, "
        "nên đường đi của model khép thành một vòng.",
        loaded["model"].lm_head.weight.numel(),
    )

    link(last, "final_norm")
    link("final_norm", "lm_head")

    spine += ["final_norm", "lm_head"]

    return {
        "model": name,
        "nodes": nodes,
        "edges": edges,
        "spine": spine,
        "kinds": [
            {"kind": "embed", "what": "Đổi chữ thành vector, và vị trí"},
            {"kind": "norm", "what": "Kéo các con số về cùng một thang"},
            {"kind": "attention", "what": "Các token trao đổi thông tin với nhau"},
            {"kind": "add", "what": "Chỗ cộng — nơi đường tắt nhập lại"},
            {"kind": "router", "what": "Chọn chuyên gia cho từng token"},
            {"kind": "expert", "what": "Chuyên gia riêng, chỉ nhận token được chọn"},
            {"kind": "shared", "what": "Chuyên gia dùng chung, token nào cũng hỏi"},
            {"kind": "ffn", "what": "Suy nghĩ riêng (MLP)"},
            {"kind": "head", "what": "Đổi vector thành điểm cho từng chữ"},
        ],
        "edgeKinds": [
            {"kind": "data", "what": "Dòng chảy chính"},
            {"kind": "residual", "what": "Đường tắt: bài cũ đi vòng qua, không bị viết lại"},
            {"kind": "route", "what": "Router gửi token tới chuyên gia"},
            {"kind": "combine", "what": "Chuyên gia trả kết quả về"},
        ],
        "summary": _architecture_summary(loaded),
    }


def _architecture_summary(loaded: dict) -> str:
    """Một câu tóm tắt đường đi, cho người mới nhìn."""
    about = loaded["about"]
    parts = [
        f"{len(loaded['chars'])} ký tự → vector {loaded['embedding'].shape[1]} chiều",
        f"→ {about['nLayers']} tầng",
        f"→ vector → {len(loaded['chars'])} điểm",
    ]
    if about["nExperts"]:
        parts.insert(2, f"(mỗi tầng: Attention rồi MoE {about['topK']}/{about['nExperts']} chuyên gia)")
    else:
        parts.insert(2, "(mỗi tầng: Attention rồi MLP)")
    return " ".join(parts)


def analyze(name: str, text: str) -> dict:
    """Chạy model trên một câu, trả về attention và đường đi của chuyên gia."""
    loaded = load(name)

    chars = [ch for ch in text if ch in set(loaded["chars"])][:MAX_TOKENS]

    if not chars:
        raise ValueError("Câu này không có ký tự nào mà model biết.")

    stoi = {ch: i for i, ch in enumerate(loaded["chars"])}
    ids = [stoi[ch] for ch in chars]

    with torch.no_grad():
        loaded["model"](torch.tensor([ids]))

    layers = []
    for block in loaded["blocks"]:
        attention = getattr(block, loaded["attn_attr"])

        layer = {
            "attention": [
                [[round(v, 3) for v in row] for row in head]
                for head in attention.attention_map[0].tolist()
            ],
            "experts": [],
            "weights": [],
        }

        if loaded["moe_attr"]:
            moe = getattr(block, loaded["moe_attr"])
            if moe.routing is not None:
                indices, weights = moe.routing
                layer["experts"] = indices[0].tolist()
                layer["weights"] = [[round(v, 3) for v in row] for row in weights[0].tolist()]

        layers.append(layer)

    return {"text": "".join(chars), "chars": chars, "ids": ids, "layers": layers}


# =====================================================================
# Chế độ thường trực
# =====================================================================


def _handle(request: dict) -> dict:
    action = request.get("action", "analyze")
    name = request.get("model", "mini_deepseek")

    if action == "meta":
        return meta(name)
    if action == "space":
        return space(name)
    if action == "architecture":
        return architecture(name)
    if action == "analyze":
        return analyze(name, request.get("text", ""))

    raise ValueError(f"Không biết lệnh {action!r}.")


def _serve():
    """Đọc một câu JSON mỗi dòng, trả về một dòng JSON.

    KHÔNG nạp sẵn model nào — model chỉ được nạp khi trang web hỏi tới.
    Nhờ vậy trang mở lên rất nhanh, và không tốn RAM cho model không xem.
    """
    print(json.dumps({"ready": True, "models": available_models()}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            result = _handle(json.loads(line))
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    # Trên Windows, mặc định không phải UTF-8 -> phải chỉnh, nếu không
    # tiếng Việt sẽ lỗi khi đọc/ghi qua đường ống.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    if "--serve" in sys.argv:
        _serve()
    elif "--models" in sys.argv:
        print(json.dumps(available_models(), ensure_ascii=False, indent=2))
    else:
        target = sys.argv[1] if len(sys.argv) > 1 else "mini_deepseek"
        question = " ".join(a for a in sys.argv[2:])
        print(json.dumps(analyze(target, question), ensure_ascii=False))
