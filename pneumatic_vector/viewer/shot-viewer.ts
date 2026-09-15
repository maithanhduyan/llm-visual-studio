/**
 * shot-viewer.ts — Chụp ảnh cảnh 3D ở những thời điểm chọn trước.
 *
 * Ảnh chụp tĩnh không điều khiển được thời gian, nên phải qua CDP: tua tới
 * đúng chỗ, đợi một khung hình vẽ xong, rồi mới chụp.
 *
 * Chạy:  bun run shot-viewer.ts <thư-mục-lưu>
 */

import { writeFileSync } from "node:fs";

const PORT = 9333;
const URL = "http://localhost:4180/";
const outDir = process.argv[2] ?? ".";

let id = 0;
const pending = new Map<number, (value: any) => void>();

function send(ws: WebSocket, method: string, params: Record<string, unknown> = {}) {
  const message = ++id;
  return new Promise<any>((resolve) => {
    pending.set(message, resolve);
    ws.send(JSON.stringify({ id: message, method, params }));
  });
}

async function evaluate(ws: WebSocket, expression: string) {
  const reply = await send(ws, "Runtime.evaluate", {
    expression: `(async () => { ${expression} })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (reply.exceptionDetails) throw new Error(reply.exceptionDetails.exception?.description);
  return reply.result?.value;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const targets = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()) as any[];
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve) => (ws.onopen = resolve));

ws.onmessage = (event) => {
  const data = JSON.parse(String(event.data));
  if (data.id && pending.has(data.id)) {
    pending.get(data.id)!(data.result ?? data);
    pending.delete(data.id);
  }
};

/**
 * Chờ trang nạp xong — và phải ĐỨNG YÊN.
 *
 * `server.ts` chạy ở chế độ phát triển, nên khi có file nào trong thư mục
 * `viewer/` thay đổi thì Bun đẩy lệnh nạp lại trang cho mọi tab đang mở.
 * Mà sửa chính file này cũng là thay đổi một file trong `viewer/` — nên
 * vừa sửa xong rồi chạy ngay thì trang sẽ nạp lại giữa chừng, và mọi thứ
 * đang trỏ tới DOM cũ bỗng thành `undefined`.
 *
 * Nên phải đợi điều kiện đúng ở HAI lần cách nhau một giây.
 */
async function waitForReady(): Promise<boolean> {
  const ready = async () => {
    try {
      return (await evaluate(ws, `return document.querySelectorAll("#summary dd").length;`)) >= 6;
    } catch {
      return false; // trang đang được thay
    }
  };

  for (let attempt = 0; attempt < 40; attempt += 1) {
    await sleep(400);
    if ((await ready()) && (await (async () => (await sleep(1000), ready()))())) return true;
  }

  return false;
}

await send(ws, "Runtime.enable");
await send(ws, "Page.enable");
await send(ws, "Emulation.setDeviceMetricsOverride", {
  width: 1000,
  height: 1000,
  deviceScaleFactor: 1,
  mobile: false,
});
await send(ws, "Page.navigate", { url: URL });

// Để yên một nhịp cho cú nạp lại do sửa file kịp tới.
await sleep(1500);

if (!(await waitForReady())) {
  console.error("Trang không nạp xong. Máy chủ đã chạy chưa? (bun run server.ts)");
  process.exit(1);
}

/** Tua tới phần trăm thứ `percent` của chuyến bay rồi chụp lại. */
async function shot(name: string, flightName: string, peak: string, percent: number) {
  const state = await evaluate(
    ws,
    `const p = document.getElementById("flight");
     p.value = ${JSON.stringify(flightName)};
     p.dispatchEvent(new Event("change", { bubbles: true }));

     for (let i = 0; i < 40; i++) {
       await new Promise(r => setTimeout(r, 100));
       const first = document.querySelector("#summary dd");
       if (first && first.textContent.trim().startsWith(${JSON.stringify(peak)})) break;
     }

     const s = document.getElementById("scrub");
     s.value = String(${percent} / 100 * 1000);
     s.dispatchEvent(new Event("input", { bubbles: true }));

     // Đợi vài khung hình cho three.js vẽ xong chỗ mới.
     await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
     await new Promise(r => setTimeout(r, 400));

     return { clock: document.getElementById("clock").textContent,
              altitude: document.querySelector("#hud .hud-row b").textContent,
              throttle: document.querySelectorAll("#hud .hud-row b")[4].textContent,
              phase: document.querySelector("#hud .phase").textContent,
              // Khung vẽ 3D nằm ở đâu trên trang — cần để cắt ảnh ra mà soi
              // đúng phần cảnh, chứ không phải soi cả chữ trên trang.
              rect: (() => { const r = document.getElementById("scene").getBoundingClientRect();
                             return { x: r.x, y: r.y, w: r.width, h: r.height }; })() };`,
  );

  // Giấu bảng số liệu đi trước khi chụp. Nó nằm đè lên góc trên bên trái của
  // khung vẽ, nên nếu để nguyên thì ảnh chụp không còn là cảnh 3D nữa.
  await evaluate(ws, `document.getElementById("hud").style.display = "none"; return 1;`);

  const image = await send(ws, "Page.captureScreenshot", { format: "png" });
  writeFileSync(`${outDir}/${name}.png`, Buffer.from(image.data, "base64"));
  writeFileSync(`${outDir}/${name}.json`, JSON.stringify(state, null, 2));

  await evaluate(ws, `document.getElementById("hud").style.display = ""; return 1;`);
  console.log(
    `  ${name}.png  ·  ${state.clock}  ·  cao ${state.altitude}  ·  ` +
      `van ${state.throttle}  ·  ${state.phase}`,
  );

  return state;
}

console.log("\nChụp cảnh 3D ở từng giai đoạn\n" + "=".repeat(56));

await shot("01-dau", "chuyentay_10m_01", "11.3", 1);
await shot("02-len", "chuyentay_10m_01", "11.3", 25);
await shot("03-dinh", "chuyentay_10m_01", "11.3", 45);
await shot("04-roi", "chuyentay_10m_01", "11.3", 62);
await shot("05-ham", "chuyentay_10m_01", "11.3", 85);
await shot("06-chamdat", "chuyentay_10m_01", "11.3", 99);
await shot("07-hong", "pilot_v0_10m_01", "11.4", 88);

console.log("=".repeat(56));
ws.close();
process.exit(0);
