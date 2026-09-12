/** format.ts — Mấy hàm nhỏ dùng chung khi hiển thị. */

/**
 * Màu của từng chuyên gia. Để ở đây vì cả panel 3D lẫn panel chuyên gia
 * đều dùng — hai chỗ phải cùng màu thì mới đối chiếu được với nhau.
 */
const EXPERT_COLORS = [
  "#4a9eff",
  "#ff9f45",
  "#56d364",
  "#bc8cff",
  "#f778ba",
  "#7ee787",
  "#ff7b72",
  "#79c0ff",
];

/** Model chưa có chuyên gia (Cấp 1) thì tô màu trung tính. */
const NO_EXPERT = "#8b98a8";

export const expertColor = (id: number) =>
  id < 0 ? NO_EXPERT : EXPERT_COLORS[id % EXPERT_COLORS.length];

/** Ký tự trắng thì hiện thành ký hiệu, nếu không sẽ nhìn không thấy gì. */
export function tokenLabel(ch: string): string {
  if (ch === " ") return "␣";
  if (ch === "\n") return "⏎";
  if (ch === "\t") return "⇥";
  return ch;
}

/** 1234567 -> "1.234.567" */
export function fmt(value: number): string {
  return value.toLocaleString("vi-VN");
}

/** Số nhỏ thì hiện 3 chữ số, số lớn thì 2. */
export function fmtValue(value: number): string {
  if (value === 0) return "0";
  if (Math.abs(value) >= 1) return value.toFixed(2);
  if (Math.abs(value) < 0.001) return value.toExponential(1);
  return value.toFixed(3);
}

/** Chữ do người dùng gõ phải được thoát, nếu không sẽ phá vỡ HTML. */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
