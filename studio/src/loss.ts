/**
 * loss.ts — Vẽ biểu đồ loss theo từng bước huấn luyện.
 *
 * Trục ngang: số bước đã học. Trục dọc: loss (cross-entropy) — model đoán sai bao nhiêu.
 */

import type { LossData } from "../types";
import { fmtValue } from "./format";

export interface LossOptions {
  /** Thang log: nhìn rõ được cả lúc loss đã nhỏ. */
  log: boolean;
  /** Vẽ thêm đường thứ hai (mỗi cấp một nghĩa — xem loss.extraName). */
  showExtra: boolean;
}

const W = 780;
const H = 290;
const PAD = { left: 66, right: 22, top: 26, bottom: 50 };

export function renderLoss(root: HTMLElement, loss: LossData, opts: LossOptions) {
  const steps = loss.step;

  if (!steps.length) {
    root.innerHTML = `<p class="empty">Chưa có lịch sử học. Chạy <code>python train.py</code> trong thư mục <code>mini_deepseek/</code> rồi tải lại trang.</p>`;
    return;
  }

  const series = [{ name: "loss (cross-entropy)", color: "#4a9eff", values: loss.loss }];
  if (opts.showExtra && loss.extra.length) {
    series.push({
      name: loss.extraName || "đường phụ",
      color: "#9aa7b8",
      values: loss.extra,
    });
  }

  // ---- Miền giá trị của trục dọc -------------------------------------
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  const rawMax = Math.max(...all);
  const positive = all.filter((v) => v > 0);
  const rawMin = positive.length ? Math.min(...positive) : 0.001;

  const yMax = opts.log ? rawMax * 1.35 : rawMax * 1.06;
  const yMin = opts.log ? Math.max(rawMin * 0.6, 1e-4) : 0;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const xAt = (i: number) => PAD.left + (steps.length === 1 ? plotW / 2 : (i / (steps.length - 1)) * plotW);
  const yAt = (value: number) => {
    const t = opts.log
      ? (Math.log10(Math.max(value, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))
      : (value - yMin) / (yMax - yMin);
    return PAD.top + (1 - Math.max(0, Math.min(1, t))) * plotH;
  };

  const path = (values: number[]) =>
    values.map((v, i) => `${i ? "L" : "M"}${xAt(i).toFixed(1)} ${yAt(v).toFixed(1)}`).join(" ");

  // ---- Lưới và vạch chia ---------------------------------------------
  const yTicks = Array.from({ length: 6 }, (_, k) => {
    const t = k / 5;
    return opts.log
      ? Math.pow(10, Math.log10(yMin) + t * (Math.log10(yMax) - Math.log10(yMin)))
      : yMin + t * (yMax - yMin);
  });

  const xTicks = Array.from({ length: 6 }, (_, k) => Math.round(((steps.length - 1) * k) / 5));

  const baseline = (PAD.top + plotH).toFixed(1);
  const last = steps.length - 1;

  const svg = `
<svg viewBox="0 0 ${W} ${H}" width="100%" role="img"
     aria-label="Biểu đồ loss theo bước huấn luyện">
  ${yTicks
    .map(
      (value) =>
        `<line x1="${PAD.left}" x2="${W - PAD.right}" y1="${yAt(value).toFixed(1)}" y2="${yAt(value).toFixed(1)}"
               stroke="#212a35" stroke-width="1" />`,
    )
    .join("")}

  <line x1="${PAD.left}" x2="${PAD.left}" y1="${PAD.top}" y2="${baseline}" stroke="#33404f" stroke-width="1" />
  <line x1="${PAD.left}" x2="${W - PAD.right}" y1="${baseline}" y2="${baseline}" stroke="#33404f" stroke-width="1" />

  ${
    series.length === 1
      ? `<path d="${path(loss.loss)} L${xAt(last).toFixed(1)} ${baseline} L${xAt(0).toFixed(1)} ${baseline} Z"
              fill="#4a9eff" fill-opacity="0.07" stroke="none" />`
      : ""
  }

  ${series
    .map(
      (s, index) =>
        `<path d="${path(s.values)}" fill="none" stroke="${s.color}"
               stroke-width="${index === 0 ? 2 : 1.5}" stroke-linejoin="round"
               ${index === 0 ? "" : 'stroke-dasharray="4 3"'} />`,
    )
    .join("")}

  <circle cx="${xAt(last).toFixed(1)}" cy="${yAt(loss.loss[last]).toFixed(1)}" r="3.5" fill="#4a9eff" />
  <text x="${(xAt(last) - 7).toFixed(1)}" y="${(yAt(loss.loss[last]) - 10).toFixed(1)}"
        text-anchor="end" class="anno">${fmtValue(loss.loss[last])}</text>
  <text x="${(xAt(0) + 7).toFixed(1)}" y="${(yAt(loss.loss[0]) + 17).toFixed(1)}"
        text-anchor="start" class="anno dim">${fmtValue(loss.loss[0])}</text>

  ${yTicks
    .map(
      (value) =>
        `<text x="${PAD.left - 10}" y="${(yAt(value) + 3.5).toFixed(1)}" text-anchor="end" class="tick">${fmtValue(value)}</text>`,
    )
    .join("")}

  ${xTicks
    .map(
      (i) =>
        `<text x="${xAt(i).toFixed(1)}" y="${PAD.top + plotH + 18}" text-anchor="middle" class="tick">${steps[i]}</text>`,
    )
    .join("")}

  <text x="${PAD.left + plotW / 2}" y="${H - 8}" text-anchor="middle" class="axis">bước huấn luyện</text>
  <text x="18" y="${PAD.top + plotH / 2}" text-anchor="middle" class="axis"
        transform="rotate(-90 18 ${PAD.top + plotH / 2})">loss (thang ${opts.log ? "log" : "thường"})</text>
</svg>`;

  root.innerHTML =
    `<div class="chart-legend">${series
      .map((s) => `<span class="lg"><i style="background:${s.color}"></i>${s.name}</span>`)
      .join("")}</div>` +
    svg +
    `<p class="caption">Nguồn: <code>mini_deepseek/mini_deepseek.pt</code> · ${steps.length} bước ·
      loss cuối = ${fmtValue(loss.loss[last])} · đo trên ký tự tiếp theo của <code>data.txt</code></p>`;
}
