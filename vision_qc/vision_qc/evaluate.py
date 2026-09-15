"""
evaluate.py — Đo model, và chọn ngưỡng để ghi xuống PLC.

Độ chính xác không phải con số quyết định. Ba con số mới là:

    **Bỏ sót** (FN) — sản phẩm lỗi đi ra thị trường. Tốn tiền bảo hành, tốn
    uy tín, có khi tốn cả một đợt triệu hồi.

    **Loại oan** (FP) — sản phẩm tốt bị thổi ra thùng phế. Tốn đúng bằng giá
    thành sản phẩm, cộng thêm thời gian dây chuyền phải dừng nếu thổi nhầm
    quá nhiều.

    **Ngưỡng** — ranh giới quyết định. Model cho ra một con số 0..1; ngưỡng
    biến con số đó thành "thổi" hoặc "không thổi".

Hai loại sai này không đối xứng, thường lệch nhau 50-1000 lần. Vì vậy ngưỡng
0,5 là một lựa chọn tuỳ tiện, và gần như luôn là lựa chọn sai. `sweep()` quét
toàn bộ dải ngưỡng để tìm chỗ tốn ít tiền nhất.

Một điều đáng nhớ: **ngưỡng tìm trên tập thi, không phải tập học.** Tìm trên
tập học thì con số đẹp mà vô nghĩa.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CameraConfig, ModelConfig
from .dataset import ImageDataset, Sample

# ---------------------------------------------------------------------
# Số đo
# ---------------------------------------------------------------------


@dataclass
class Metrics:
    """Kết quả ở MỘT ngưỡng."""

    threshold: float = 0.5

    true_positive: int = 0    # lỗi, bị thổi  (đúng)
    false_positive: int = 0   # tốt, bị thổi  (loại oan)
    true_negative: int = 0    # tốt, không thổi (đúng)
    false_negative: int = 0   # lỗi, không thổi (bỏ sót)

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """Trong số bị thổi, bao nhiêu phần thật sự là lỗi."""
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Trong số lỗi thật, bắt được bao nhiêu. Đây là con số quan trọng nhất."""
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def false_reject_rate(self) -> float:
        """Tỉ lệ sản phẩm TỐT bị thổi oan."""
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else 0.0

    @property
    def escape_rate(self) -> float:
        """Tỉ lệ sản phẩm LỖI đi lọt. 1 - recall."""
        return 1.0 - self.recall if (self.true_positive + self.false_negative) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def cost(self, cost_false_reject: float, cost_escape: float) -> float:
        """Tiền mất cho cả lô, tính theo đơn vị của `cost_false_reject`."""
        return self.false_positive * cost_false_reject + self.false_negative * cost_escape

    def cost_per_part(self, cost_false_reject: float, cost_escape: float) -> float:
        return self.cost(cost_false_reject, cost_escape) / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"ngưỡng {self.threshold:.3f} · đúng {self.accuracy:6.1%} · "
            f"bắt được {self.recall:6.1%} lỗi · thổi oan {self.false_reject_rate:6.1%}"
        )


@dataclass
class Report:
    """Toàn bộ kết quả đánh giá."""

    chosen: Metrics = field(default_factory=Metrics)
    at_half: Metrics = field(default_factory=Metrics)
    sweep: list[dict] = field(default_factory=list)

    auc: float = 0.0
    num_ok: int = 0
    num_ng: int = 0
    num_params: int = 0

    latency_ms: float = 0.0
    latency_p95_ms: float = 0.0
    budget_ms: float = 0.0

    cost_false_reject: float = 1.0
    cost_escape: float = 50.0
    max_false_reject_rate: float = 1.0
    constraint_blocked: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["chosen"] = asdict(self.chosen)
        data["at_half"] = asdict(self.at_half)
        return data


# ---------------------------------------------------------------------
# Tính toán
# ---------------------------------------------------------------------


def metrics_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Metrics:
    """Dựng ma trận nhầm lẫn ở một ngưỡng. `scores` là P(NG), `labels` 0/1."""
    predicted_ng = scores >= threshold
    actually_ng = labels >= 0.5

    return Metrics(
        threshold=float(threshold),
        true_positive=int((predicted_ng & actually_ng).sum()),
        false_positive=int((predicted_ng & ~actually_ng).sum()),
        true_negative=int((~predicted_ng & ~actually_ng).sum()),
        false_negative=int((~predicted_ng & actually_ng).sum()),
    )


def sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    cost_false_reject: float = 1.0,
    cost_escape: float = 50.0,
    points: int = 201,
) -> list[dict]:
    """Quét ngưỡng từ 0 tới 1, trả về số đo ở từng mức."""
    rows: list[dict] = []

    for threshold in np.linspace(0.0, 1.0, points):
        m = metrics_at(scores, labels, float(threshold))
        rows.append({
            "threshold": round(float(threshold), 4),
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "false_reject_rate": m.false_reject_rate,
            "false_positive": m.false_positive,
            "false_negative": m.false_negative,
            "cost_per_part": m.cost_per_part(cost_false_reject, cost_escape),
        })

    return rows


def best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    cost_false_reject: float = 1.0,
    cost_escape: float = 50.0,
    max_false_reject_rate: float = 1.0,
) -> tuple[float, Metrics, bool]:
    """Ngưỡng tốn ít tiền nhất TRONG SỐ những ngưỡng vận hành được.

    Trả về (ngưỡng, số đo, có bị ràng buộc chặn không).

    Quét trên các giá trị điểm có thật trong dữ liệu chứ không phải lưới 0,01:
    ngưỡng đúng thường nằm lệch giữa lưới, và ở bài toán lệch 50 lần thì lệch
    một nấc cũng đổi hàng chục lần số tiền.

    VÌ SAO CÓ RÀNG BUỘC
    -------------------
    Toán học thuần cho ra câu trả lời không dùng được. Đo trên tập thi của dự
    án, tỉ lệ chi phí 50:1: ngưỡng rẻ nhất là 0,044 — bắt 100% lỗi và thổi oan
    98,1% sản phẩm tốt. Toán đúng (bỏ sót 5 lỗi đắt bằng thổi oan 250 sản phẩm
    tốt), nhưng thổi 98% sản phẩm ra thùng phế thì dây chuyền dừng, và khoản lỗ
    đó không nằm trong mô hình hai số hạng.

    Nên chỉ xét những ngưỡng có tỉ lệ thổi oan dưới `max_false_reject_rate`.
    """
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))

    best_cost = float("inf")
    best: tuple[float, Metrics] | None = None
    unconstrained: tuple[float, Metrics] | None = None
    unconstrained_cost = float("inf")

    for threshold in candidates:
        m = metrics_at(scores, labels, float(threshold))
        cost = m.cost(cost_false_reject, cost_escape)

        if cost < unconstrained_cost:
            unconstrained_cost = cost
            unconstrained = (float(threshold), m)

        if m.false_reject_rate > max_false_reject_rate:
            continue

        if cost < best_cost:
            best_cost = cost
            best = (float(threshold), m)

    if best is None:
        # Không ngưỡng nào thoả ràng buộc. Nói thẳng ra thay vì lặng lẽ bỏ qua
        # ràng buộc — bỏ qua thì lại quay về con số 98% ở trên.
        if unconstrained is None:
            unconstrained = (0.5, metrics_at(scores, labels, 0.5))
        return unconstrained[0], unconstrained[1], True

    blocked = unconstrained is not None and unconstrained[0] != best[0]
    return best[0], best[1], blocked


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Diện tích dưới đường ROC — đo mức TÁCH BIỆT, không phụ thuộc ngưỡng.

    Đây là con số đáng tin nhất về chất lượng model: nó không đổi khi ta đổi
    ngưỡng. 0,5 = đoán bừa. 1,0 = tách hoàn hảo.

    Tính bằng thống kê hạng (công thức Mann-Whitney U), không cần sklearn.
    """
    positive = scores[labels >= 0.5]
    negative = scores[labels < 0.5]

    if len(positive) == 0 or len(negative) == 0:
        return 0.0

    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)

    # Xử lý điểm trùng nhau: gán hạng trung bình
    sorted_values = combined[order]
    index = 0
    while index < len(sorted_values):
        end = index
        while end + 1 < len(sorted_values) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        if end > index:
            average = (index + end + 2) / 2.0  # hạng bắt đầu từ 1
            ranks[order[index:end + 1]] = average
        index = end + 1

    rank_sum = ranks[:len(positive)].sum()
    n_pos, n_neg = len(positive), len(negative)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------
# Chạy model trên tập dữ liệu
# ---------------------------------------------------------------------


def score_samples(
    model,
    samples: list[Sample],
    model_cfg: ModelConfig | None = None,
    camera_cfg: CameraConfig | None = None,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Chạy model trên từng ảnh. Trả về (điểm P(NG), nhãn thật)."""
    model_cfg = model_cfg or model.cfg
    camera_cfg = camera_cfg or CameraConfig()

    dataset = ImageDataset(samples, model_cfg, camera_cfg, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            probabilities = torch.sigmoid(model(images))
            scores.append(probabilities.numpy())
            labels.append(batch_labels.numpy())

    if not scores:
        return np.array([]), np.array([])

    return np.concatenate(scores), np.concatenate(labels)


def measure_latency(
    model,
    model_cfg: ModelConfig | None = None,
    runs: int = 30,
) -> tuple[float, float]:
    """Đo thời gian suy luận một ảnh, trên CPU. Trả về (trung bình, p95) ms.

    Đây là con số phải so với `LineConfig.time_budget_ms`. Model chính xác
    99,9% mà chạy 200 ms trong khi ngân sách là 40 ms thì vô dụng.
    """
    import time

    model_cfg = model_cfg or model.cfg
    # Lô 1 vì thực tế mỗi sản phẩm tới một lúc, không đi theo lô.
    dummy = torch.randn(1, 1 if model_cfg.grayscale else 3,
                        model_cfg.image_size, model_cfg.image_size)

    model.eval()
    timings: list[float] = []

    with torch.no_grad():
        for _ in range(max(runs, 5)):
            start = time.perf_counter()
            model(dummy)
            timings.append((time.perf_counter() - start) * 1000.0)

    array = np.array(timings)
    return float(array.mean()), float(np.percentile(array, 95))


# ---------------------------------------------------------------------
# Lệnh đầy đủ
# ---------------------------------------------------------------------


def run_evaluation(
    model,
    val_samples: list[Sample],
    model_cfg: ModelConfig | None = None,
    camera_cfg: CameraConfig | None = None,
    cost_false_reject: float = 1.0,
    cost_escape: float = 50.0,
    max_false_reject_rate: float = 0.05,
    budget_ms: float = 0.0,
    name: str = "defect",
    quiet: bool = False,
    save: bool = True,
) -> Report:
    """Đánh giá đầy đủ và lưu báo cáo."""
    from .paths import RUNS_DIR

    scores, labels = score_samples(model, val_samples, model_cfg, camera_cfg)

    report = Report(
        auc=roc_auc(scores, labels),
        num_ok=int((labels < 0.5).sum()),
        num_ng=int((labels >= 0.5).sum()),
        num_params=getattr(model, "num_params", 0),
        cost_false_reject=cost_false_reject,
        cost_escape=cost_escape,
        max_false_reject_rate=max_false_reject_rate,
        budget_ms=budget_ms,
    )

    report.at_half = metrics_at(scores, labels, 0.5)
    threshold, chosen, blocked = best_threshold(
        scores, labels, cost_false_reject, cost_escape, max_false_reject_rate
    )
    report.chosen = chosen
    report.constraint_blocked = blocked
    report.sweep = sweep(scores, labels, cost_false_reject, cost_escape)

    mean_ms, p95_ms = measure_latency(model, model_cfg)
    report.latency_ms = mean_ms
    report.latency_p95_ms = p95_ms

    if not quiet:
        print(f"Tập thi: {report.num_ok} ảnh OK · {report.num_ng} ảnh NG "
              f"· {report.num_params:,} thông số\n")

        print(f"  Mức tách biệt (AUC): {report.auc:.4f}   "
              f"(0,5 = đoán bừa · 1,0 = hoàn hảo)\n")

        print("  Nếu để ngưỡng 0,5 (mặc định ngây thơ):")
        print(f"    {report.at_half.summary()}")
        print(f"    bỏ sót {report.at_half.false_negative} lỗi · "
              f"loại oan {report.at_half.false_positive} sản phẩm tốt\n")

        print(f"  Ngưỡng rẻ nhất (loại oan {cost_false_reject:g} · "
              f"bỏ sót {cost_escape:g}), thổi oan tối đa {max_false_reject_rate:.0%}:")
        print(f"    {report.chosen.summary()}")
        print(f"    -> nên đặt ngưỡng = {threshold:.4f}")
        print(f"    bỏ sót {report.chosen.false_negative} lỗi · "
              f"loại oan {report.chosen.false_positive} sản phẩm tốt")
        print(f"    tiền mất mỗi sản phẩm: "
              f"{report.chosen.cost_per_part(cost_false_reject, cost_escape):.3f} "
              f"(so với {report.at_half.cost_per_part(cost_false_reject, cost_escape):.3f} "
              f"ở ngưỡng 0,5)")

        if blocked:
            print("\n    RÀNG BUỘC ĐANG CHẶN. Bỏ ràng buộc thì ngưỡng rẻ nhất")
            print(f"    theo toán học sẽ thổi oan hơn {max_false_reject_rate:.0%} "
                  f"sản phẩm tốt —")
            print("    đúng về số học nhưng dây chuyền không chạy được như vậy.")
            print("    Muốn bắt nhiều lỗi hơn thì phải làm model TỐT HƠN")
            print("    (AUC cao hơn), không phải hạ ngưỡng.")

        print(f"\n  Thời gian suy luận: {mean_ms:.1f} ms trung bình · {p95_ms:.1f} ms p95")
        if budget_ms > 0:
            if p95_ms < budget_ms:
                print(f"    ngân sách {budget_ms:.0f} ms -> ĐẠT "
                      f"(còn thừa {budget_ms - p95_ms:.0f} ms)")
            else:
                print(f"    ngân sách {budget_ms:.0f} ms -> KHÔNG ĐẠT "
                      f"(vượt {p95_ms - budget_ms:.0f} ms). "
                      f"Giảm `image_size` hoặc tăng tốc băng tải sẽ tệ hơn.")

    if save:
        path = RUNS_DIR / f"{name}_eval.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        if not quiet:
            print(f"\n  báo cáo -> {path}")

    return report
