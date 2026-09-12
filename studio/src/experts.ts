/**
 * experts.ts — Vẽ "token nào hỏi chuyên gia nào".
 *
 * Mỗi cột là một token, mỗi hàng là một tầng. Trong một ô, các mảnh màu
 * xếp chồng lên nhau: mảnh càng cao nghĩa là chuyên gia đó được tin càng nhiều.
 */

import type { AnalyzeResult, StudioMeta } from "../types";
import { escapeHtml, expertColor, tokenLabel } from "./format";

export function renderExperts(
  grid: HTMLElement,
  legend: HTMLElement,
  result: AnalyzeResult,
  meta: StudioMeta,
  tooltip: HTMLElement,
) {
  // Đếm xem mỗi chuyên gia được chọn bao nhiêu lần trong cả câu.
  const counts = new Array<number>(meta.nExperts).fill(0);
  for (const layer of result.layers) {
    for (const row of layer.experts) for (const id of row) counts[id] += 1;
  }

  legend.innerHTML = counts
    .map(
      (count, id) =>
        `<span class="lg"><i style="background:${expertColor(id)}"></i>chuyên gia ${id}<b>${count} lần</b></span>`,
    )
    .join("");

  const tokenRow = result.chars
    .map((ch) => `<div class="extok">${escapeHtml(tokenLabel(ch))}</div>`)
    .join("");

  const rows = result.layers
    .map((layer, l) => {
      const cells = result.chars
        .map((_, t) => {
          const segments = layer.experts[t]
            .map(
              (id, k) =>
                `<i style="height:${Math.round(layer.weights[t][k] * 100)}%;background:${expertColor(id)}"></i>`,
            )
            .join("");
          return `<div class="ecell" data-l="${l}" data-t="${t}">${segments}</div>`;
        })
        .join("");

      return `<div class="exrow"><div class="exlabel">tầng ${l}</div><div class="excells">${cells}</div></div>`;
    })
    .join("");

  grid.innerHTML =
    `<div class="exrow exhead"><div class="exlabel"></div><div class="excells">${tokenRow}</div></div>` +
    rows;

  grid.onmousemove = (event) => {
    const cell = (event.target as HTMLElement).closest<HTMLElement>(".ecell");
    if (!cell) {
      tooltip.hidden = true;
      return;
    }

    const l = Number(cell.dataset.l);
    const t = Number(cell.dataset.t);
    const picks = result.layers[l].experts[t]
      .map((id, k) => {
        const weight = result.layers[l].weights[t][k];
        return `<span style="color:${expertColor(id)}">■</span> chuyên gia ${id} <span class="val">${weight.toFixed(2)}</span>`;
      })
      .join(" &nbsp;·&nbsp; ");

    tooltip.hidden = false;
    tooltip.innerHTML =
      `tầng ${l} · token #${t} <b>${escapeHtml(tokenLabel(result.chars[t]))}</b><br>${picks}`;
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
  };

  grid.onmouseleave = () => {
    tooltip.hidden = true;
  };
}
