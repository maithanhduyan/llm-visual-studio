"""
train.py — Dạy model. Một vòng lặp, không có gì che giấu.

Bốn quyết định trong file này, và lý do của từng cái:

1. **BCEWithLogitsLoss, không phải CrossEntropy.** Model có một đầu ra là điểm
   "mức độ lỗi". Xem `model.py` để biết vì sao.

2. **pos_weight = số ảnh OK / số ảnh NG.** Trong xưởng, sản phẩm lỗi luôn ít
   hơn nhiều. Không cân bằng thì model học được rằng đoán "OK" suốt là đạt
   95%, và nó sẽ làm đúng như vậy — đó là kiểu sai tốn tiền nhất.

3. **Dừng sớm theo loss của tập THI, không theo tập học.** Loss tập học luôn
   giảm. Nó giảm kể cả khi model đang học vẹt. Chỉ loss tập thi mới nói được
   lúc nào nên dừng.

4. **Giữ lại trọng số của epoch TỐT NHẤT, không phải epoch cuối.** Epoch cuối
   là epoch đã học vẹt nhiều nhất.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .augment import AugmentConfig
from .config import CameraConfig, ModelConfig
from .dataset import ImageDataset, Sample, split_by_group, statistics
from .model import DefectNet, save_model
from .paths import MODELS_DIR, RUNS_DIR, ensure_dirs


@dataclass
class History:
    """Nhật ký huấn luyện — để vẽ lại và để đọc lại sau này."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)

    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False

    train_size: int = 0
    val_size: int = 0
    train_groups: int = 0
    val_groups: int = 0
    pos_weight: float = 1.0
    num_params: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _evaluate_loss(
    model: DefectNet,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Trả về (loss trung bình, độ chính xác) trên một tập."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.shape[0]
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels).sum())
            count += labels.shape[0]

    if count == 0:
        return 0.0, 0.0
    return total_loss / count, correct / count


def train_model(
    train_samples: list[Sample],
    val_samples: list[Sample],
    model_cfg: ModelConfig | None = None,
    camera_cfg: CameraConfig | None = None,
    augment_cfg: AugmentConfig | None = None,
    epochs: int | None = None,
    quiet: bool = False,
) -> tuple[DefectNet, History]:
    """Huấn luyện và trả về (model tốt nhất, nhật ký)."""
    model_cfg = model_cfg or ModelConfig()
    camera_cfg = camera_cfg or CameraConfig()
    augment_cfg = augment_cfg or AugmentConfig()
    epochs = epochs or model_cfg.epochs

    torch.manual_seed(model_cfg.seed)
    np.random.seed(model_cfg.seed)

    train_set = ImageDataset(train_samples, model_cfg, camera_cfg, train=True,
                             augment_cfg=augment_cfg, seed=model_cfg.seed)
    val_set = ImageDataset(val_samples, model_cfg, camera_cfg, train=False)

    train_loader = DataLoader(train_set, batch_size=model_cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=len(train_set) > model_cfg.batch_size)
    val_loader = DataLoader(val_set, batch_size=model_cfg.batch_size, shuffle=False,
                            num_workers=0)

    model = DefectNet(model_cfg)

    # Chỉ tính pos_weight khi thật sự mất cân bằng. Lệch nhẹ mà đặt trọng số
    # vẫn được, nhưng đẩy mạnh thì model thiên hẳn về phía báo lỗi.
    pos_weight = torch.tensor([train_set.pos_weight], dtype=torch.float32) \
        if train_set.num_ng > 0 else torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr,
                                  weight_decay=model_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = History(
        train_size=len(train_set),
        val_size=len(val_set),
        train_groups=len({s.group for s in train_samples}),
        val_groups=len({s.group for s in val_samples}),
        pos_weight=float(pos_weight.item()),
        num_params=model.num_params,
    )

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()

        # -- học --
        model.train()
        running = 0.0
        seen = 0

        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Kẹp gradient. Đo được khi không kẹp: loss của tập thi nhảy từ
            # 0,60 lên 3,07 rồi về 0,60 trong hai epoch liền — một lô ảnh
            # khó đẩy gradient vọt lên và model bị bắn ra khỏi vùng tốt.
            # Kẹp ở 1,0 là chuẩn mực và rẻ.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]

        scheduler.step()
        train_loss = running / max(seen, 1)
        val_loss, val_accuracy = _evaluate_loss(model, val_loader, criterion)

        elapsed = time.perf_counter() - started
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_accuracy)
        history.seconds.append(elapsed)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- tốt nhất"
        else:
            stale += 1
            marker = ""

        if not quiet:
            print(f"  epoch {epoch:3d}/{epochs}  học {train_loss:.4f}  "
                  f"thi {val_loss:.4f}  đúng {val_accuracy:6.1%}  "
                  f"{elapsed:5.1f}s{marker}")

        if stale >= model_cfg.patience:
            history.stopped_early = True
            if not quiet:
                print(f"  dừng sớm: {model_cfg.patience} epoch liền không tốt hơn "
                      f"(tốt nhất là epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    model.eval()
    history.best_epoch = best_epoch
    history.best_val_loss = best_val

    return model, history


# ---------------------------------------------------------------------
# Lệnh đầy đủ
# ---------------------------------------------------------------------


def run_training(
    data_dir: Path,
    name: str = "defect",
    model_cfg: ModelConfig | None = None,
    camera_cfg: CameraConfig | None = None,
    augment_cfg: AugmentConfig | None = None,
    val_fraction: float = 0.2,
    epochs: int | None = None,
    quiet: bool = False,
) -> tuple[DefectNet, History]:
    """Quét dữ liệu -> chia tập -> huấn luyện -> lưu model + nhật ký."""
    ensure_dirs()

    from .dataset import scan

    model_cfg = model_cfg or ModelConfig()
    camera_cfg = camera_cfg or CameraConfig()

    samples = scan(data_dir)
    if not samples:
        raise SystemExit(
            f"Không có ảnh nào trong {data_dir}.\n"
            f"  Chụp ảnh thật:   python -m vision_qc capture\n"
            f"  Hoặc sinh ảnh giả để thử: python -m vision_qc synth --ok 200 --ng 200"
        )

    stats = statistics(samples, duplicates=False)
    if stats.counts.get(0, 0) == 0 or stats.counts.get(1, 0) == 0:
        raise SystemExit(
            f"Cần ảnh của CẢ HAI lớp. Đang có: OK={stats.counts.get(0, 0)}, "
            f"NG={stats.counts.get(1, 0)}."
        )

    train_samples, val_samples = split_by_group(samples, val_fraction, model_cfg.seed)

    if not quiet:
        print(f"Dữ liệu: {stats.total} ảnh "
              f"(OK {stats.counts.get(0, 0)} · NG {stats.counts.get(1, 0)})")
        print("Chia theo sản phẩm, không theo ảnh:")
        print(f"  học: {len(train_samples):4d} ảnh / "
              f"{len({s.group for s in train_samples})} sản phẩm")
        print(f"  thi: {len(val_samples):4d} ảnh / "
              f"{len({s.group for s in val_samples})} sản phẩm")

        gap = stats.brightness_gap()
        if gap > 5.0:
            print(f"  CẢNH BÁO: độ sáng hai lớp lệch {gap:.1f}%. Model có thể đang "
                  f"đoán theo ánh sáng chứ không nhìn sản phẩm.")
        print()

    model, history = train_model(
        train_samples, val_samples, model_cfg, camera_cfg, augment_cfg,
        epochs=epochs, quiet=quiet,
    )

    # -- lưu --
    model_path = MODELS_DIR / f"{name}.pt"
    save_model(model, model_path, extra={
        "history": history.to_dict(),
        # ROI phải đi cùng model — xem `model.restore_camera`.
        "camera_roi": list(camera_cfg.roi) if camera_cfg.roi else None,
        "threshold": model_cfg.threshold,
        "train_groups": sorted({s.group for s in train_samples})[:500],
        "val_groups": sorted({s.group for s in val_samples})[:500],
    })

    history_path = RUNS_DIR / f"{name}_history.json"
    history_path.write_text(json.dumps(history.to_dict(), indent=2), encoding="utf-8")

    if not quiet:
        print(f"\n  model  -> {model_path}")
        print(f"  nhật ký -> {history_path}")
        print(f"  {history.num_params:,} thông số · dừng ở epoch {history.best_epoch} "
              f"· loss thi tốt nhất {history.best_val_loss:.4f}")

    return model, history


def overfit_warning(history: History) -> str | None:
    """Cảnh báo học vẹt — nói thẳng khi model chỉ đang học thuộc lòng."""
    if not history.train_loss or not history.val_loss:
        return None

    train_loss = history.train_loss[-1]
    val_loss = history.val_loss[-1]

    if val_loss > train_loss * 2.5 and val_loss > 0.3:
        return (
            f"Model học vẹt: loss học {train_loss:.3f} nhưng loss thi {val_loss:.3f}. "
            f"Nó thuộc lòng tập học chứ chưa học được lỗi. Cách chữa, theo thứ tự "
            f"hiệu quả: chụp thêm ảnh (nhiều sản phẩm khác nhau, không phải nhiều "
            f"khung hình của cùng một sản phẩm), rồi mới tới tăng cường mạnh hơn."
        )
    return None
