/**
 * attention.ts — Vẽ bản đồ attention.
 *
 * Hàng = token đang hỏi (query). Cột = token được nhìn tới (key).
 * Vì token không được nhìn vào tương lai, chỉ có nửa dưới là có màu.
 */

import type { LayerData } from "../types";
import { escapeHtml, tokenLabel } from "./format";

const PAD = { left: 46, top: 10, right: 10, bottom: 46 };

/** Lấy bản đồ của một tầng. head = -1 nghĩa là trung bình mọi đầu. */
export function matrixFor(layers: LayerData[], layerIndex: number, head: number): number[][] {
  const data = layers[layerIndex];
  if (head >= 0) return data.attention[head];

  // Trung bình cộng tất cả các đầu lại.
  const heads = data.attention;
  const T = heads[0].length;
  const out: number[][] = [];

  for (let i = 0; i < T; i += 1) {
    const row = new Array<number>(T).fill(0);
    for (const h of heads) for (let j = 0; j < T; j += 1) row[j] += h[i][j];
    for (let j = 0; j < T; j += 1) row[j] /= heads.length;
    out.push(row);
  }

  return out;
}

/** Ít chú ý -> gần như nền. Chú ý nhiều -> vàng cam. */
function ramp(t: number): string {
  const g = Math.pow(Math.max(0, Math.min(1, t)), 0.55);
  return `rgb(${Math.round(19 + g * 236)},${Math.round(26 + g * 133)},${Math.round(34 + g * 35)})`;
}

export function createAttentionView(
  canvas: HTMLCanvasElement,
  tooltip: HTMLElement,
  readout: HTMLElement,
) {
  let chars: string[] = [];
  let matrix: number[][] = [];
  let hover: { i: number; j: number } | null = null;
  let cell = 12;
  let originX = PAD.left;
  let originY = PAD.top;

  function draw() {
    const T = matrix.length;
    if (!T) return;

    const dpr = window.devicePixelRatio || 1;
    const available = Math.max(260, Math.min((canvas.parentElement?.clientWidth ?? 720) - 4, 760));
    cell = Math.max(5, Math.min(26, Math.floor((available - PAD.left - PAD.right) / T)));

    const width = T * cell + PAD.left + PAD.right;
    const height = T * cell + PAD.top + PAD.bottom;

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    originX = PAD.left;
    originY = PAD.top;

    let max = 1e-9;
    for (let i = 0; i < T; i += 1) for (let j = 0; j <= i; j += 1) max = Math.max(max, matrix[i][j]);

    // Vùng bị chặn: token không được nhìn về tương lai.
    ctx.fillStyle = "rgba(255,255,255,0.028)";
    for (let i = 0; i < T - 1; i += 1) {
      ctx.fillRect(originX + (i + 1) * cell, originY + i * cell, (T - i - 1) * cell, cell);
    }

    // Các ô màu.
    const grid = cell >= 11;
    ctx.lineWidth = 1;
    for (let i = 0; i < T; i += 1) {
      for (let j = 0; j <= i; j += 1) {
        const x = originX + j * cell;
        const y = originY + i * cell;
        ctx.fillStyle = ramp(matrix[i][j] / max);
        ctx.fillRect(x, y, cell, cell);
        if (grid) {
          ctx.strokeStyle = "rgba(0,0,0,0.28)";
          ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        }
      }
    }

    // Làm nổi hàng và cột đang trỏ tới.
    if (hover) {
      ctx.fillStyle = "rgba(255,255,255,0.07)";
      ctx.fillRect(originX, originY + hover.i * cell, T * cell, cell);
      ctx.fillRect(originX + hover.j * cell, originY, cell, T * cell);

      ctx.strokeStyle = "#e6edf3";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(originX + hover.j * cell + 0.75, originY + hover.i * cell + 0.75, cell - 1.5, cell - 1.5);
    }

    // Nhãn token hai trục.
    const fontSize = Math.max(7, Math.min(11, cell - 1));
    ctx.font = `${fontSize}px ui-monospace, Consolas, monospace`;
    ctx.fillStyle = "#8b98a8";

    if (cell >= 8) {
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i < T; i += 1) {
        ctx.fillText(tokenLabel(chars[i]), originX - 6, originY + i * cell + cell / 2);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let j = 0; j < T; j += 1) {
        ctx.fillText(tokenLabel(chars[j]), originX + j * cell + cell / 2, originY + T * cell + 5);
      }
    }

    // Tên hai trục.
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = "#5d6b7d";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("token được nhìn tới (key)", originX + (T * cell) / 2, originY + T * cell + 34);

    ctx.save();
    ctx.translate(11, originY + (T * cell) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("token đang hỏi (query)", 0, 0);
    ctx.restore();
  }

  canvas.addEventListener("mousemove", (event) => {
    const T = matrix.length;
    if (!T) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left - originX;
    const y = event.clientY - rect.top - originY;
    const j = Math.floor(x / cell);
    const i = Math.floor(y / cell);

    if (x >= 0 && y >= 0 && i >= 0 && i < T && j >= 0 && j <= i) {
      hover = { i, j };

      tooltip.hidden = false;
      tooltip.textContent = `"${tokenLabel(chars[i])}" nhìn "${tokenLabel(chars[j])}" · ${matrix[i][j].toFixed(3)}`;
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;

      // Ba token mà hàng này chú ý nhất.
      const top = matrix[i]
        .map((value, k) => ({ value, k }))
        .slice(0, i + 1)
        .sort((a, b) => b.value - a.value)
        .slice(0, 3)
        .map(
          ({ value, k }) =>
            `<b>${escapeHtml(tokenLabel(chars[k]))}</b> #${k} <span class="val">${value.toFixed(3)}</span>`,
        )
        .join(" &nbsp;·&nbsp; ");

      readout.innerHTML =
        `Token <b>#${i} ${escapeHtml(tokenLabel(chars[i]))}</b> chú ý nhất vào: ${top}`;
    } else {
      hover = null;
      tooltip.hidden = true;
    }

    draw();
  });

  canvas.addEventListener("mouseleave", () => {
    hover = null;
    tooltip.hidden = true;
    draw();
  });

  window.addEventListener("resize", () => draw());

  return {
    set(nextChars: string[], nextMatrix: number[][]) {
      chars = nextChars;
      matrix = nextMatrix;
      hover = null;
      draw();
    },
  };
}
