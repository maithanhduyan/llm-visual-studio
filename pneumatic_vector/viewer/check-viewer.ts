/**
 * check-viewer.ts — Điều khiển Chrome qua CDP để thử trình xem 3D thật.
 *
 * Không phải xem ảnh chụp rồi đoán. Script này bấm nút, kéo thanh trượt,
 * đổi chuyến bay — rồi đọc lại con số trên trang để xem có đúng không.
 *
 * Chạy:  bun run check-viewer.ts
 */

const PORT = 9333;
const URL = "http://localhost:4180/";

let id = 0;
const pending = new Map<number, (value: any) => void>();
const problems: string[] = [];

function send(ws: WebSocket, method: string, params: Record<string, unknown> = {}) {
  const message = ++id;
  return new Promise((resolve) => {
    pending.set(message, resolve);
    ws.send(JSON.stringify({ id: message, method, params }));
  });
}

async function evaluate(ws: WebSocket, expression: string) {
  const reply: any = await send(ws, "Runtime.evaluate", {
    expression: `(async () => { ${expression} })()`,
    awaitPromise: true,
    returnByValue: true,
  });

  if (reply.exceptionDetails) {
    throw new Error(reply.exceptionDetails.exception?.description ?? "lỗi trong trang");
  }
  return reply.result?.value;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------

const targets = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()) as any[];
const page = targets.find((t) => t.type === "page");

if (!page) {
  console.error("Không tìm thấy tab nào của Chrome.");
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve) => (ws.onopen = resolve));

ws.onmessage = (event) => {
  const data = JSON.parse(String(event.data));

  if (data.id && pending.has(data.id)) {
    pending.get(data.id)!(data.result ?? data);
    pending.delete(data.id);
  }

  if (data.method === "Runtime.exceptionThrown") {
    problems.push("lỗi JS: " + data.params.exceptionDetails.text);
  }
  if (data.method === "Runtime.consoleAPICalled" && data.params.type === "error") {
    problems.push("console.error: " + JSON.stringify(data.params.args.map((a: any) => a.value)));
  }
};

await send(ws, "Runtime.enable");
await send(ws, "Page.enable");
await send(ws, "Emulation.setDeviceMetricsOverride", {
  width: 1100,
  height: 1000,
  deviceScaleFactor: 1,
  mobile: false,
});
await send(ws, "Page.navigate", { url: URL });

/**
 * Chờ một điều kiện ĐỨNG YÊN, không phải chỉ đúng một lần.
 *
 * `server.ts` chạy ở chế độ phát triển, nên khi có file nào trong thư mục
 * `viewer/` thay đổi thì Bun đẩy lệnh nạp lại trang cho mọi tab đang mở. Mà
 * sửa chính file này cũng là thay đổi một file trong `viewer/` — nên vừa sửa
 * xong rồi chạy ngay thì trang sẽ nạp lại giữa chừng, và mọi thứ đang trỏ
 * tới DOM cũ bỗng thành `undefined`.
 */
async function waitFor(check: string, label: string, tries = 60): Promise<boolean> {
  const ok = async () => {
    try {
      return Boolean(await evaluate(ws, check));
    } catch {
      return false; // trang đang được thay
    }
  };

  for (let attempt = 0; attempt < tries; attempt += 1) {
    await sleep(400);
    if ((await ok()) && (await (async () => (await sleep(800), ok()))())) return true;
  }

  console.error(`Không chờ được: ${label}`);
  return false;
}

// Chế độ mặc định giờ là CHẠY SỐNG. File này nói về chế độ PHÁT LẠI, nên đợi
// trang sống chạy được rồi mới đổi sang.
//
// Hai chế độ có bảng số liệu với số mục khác nhau nhưng cùng hình dạng, nên
// không đếm số mục mà đọc thẳng dấu `data-mode` — đếm thì có ngày đếm nhầm
// sang bảng của chế độ kia mà vẫn thấy "đúng".
if (
  !(await waitFor(
    `return Boolean(document.getElementById("sl-finAreaCm2"));`,
    "chế độ sống",
  ))
) {
  process.exit(1);
}

await sleep(1500); // để cú nạp lại do sửa file kịp tới

await evaluate(ws, `document.getElementById("mode-replay").click(); return 1;`);

if (
  !(await waitFor(
    `return document.getElementById("summary").dataset.mode === "replay"
             && document.querySelectorAll("#summary dd").length === 6;`,
    "chế độ phát lại nạp xong chuyến bay",
  ))
) {
  process.exit(1);
}

const results: [string, boolean, string][] = [];

function expect(name: string, ok: boolean, detail: string) {
  results.push([name, ok, detail]);
}

// ---------------------------------------------------------------------
// 1. Chuyến bay đã nạp
// ---------------------------------------------------------------------

const loaded = await evaluate(
  ws,
  `return {
     flights: document.querySelectorAll("#flight option").length,
     summary: [...document.querySelectorAll("#summary dd")].map(d => d.textContent.trim()),
     decisions: document.querySelectorAll("#decisions .decision").length,
     phases: [...document.querySelectorAll("#decisions .dcmd")].map(d => d.textContent.trim()),
     canvas: (() => { const c = document.querySelector("canvas");
                      return c ? c.width + "x" + c.height : "không có"; })(),
   };`,
);

// Số chuyến bay có thể đổi khi ta ghi thêm, nên lấy từ chính máy chủ mà so.
const advertised = (await (await fetch("http://localhost:4180/api/flights")).json()) as {
  flights: unknown[];
};

expect(
  "nạp được danh sách chuyến bay",
  loaded.flights === advertised.flights.length && loaded.flights > 0,
  `${loaded.flights} chuyến (máy chủ có ${advertised.flights.length})`,
);
expect(
  "bảng số liệu có đủ 6 mục",
  loaded.summary.length === 6,
  loaded.summary.join(" · "),
);
expect(
  "canvas 3D đã được tạo và co giãn theo khung",
  /^\d+x\d+$/.test(loaded.canvas) && Number(loaded.canvas.split("x")[0]) > 500,
  loaded.canvas,
);
expect(
  "danh sách lệnh của model đã dựng",
  loaded.decisions >= 4,
  `${loaded.decisions} lần đổi lệnh: ${loaded.phases.join(" > ")}`,
);

// ---------------------------------------------------------------------
// 2. Phát lại tự chạy
// ---------------------------------------------------------------------

const first = await evaluate(ws, `return document.getElementById("clock").textContent;`);
await sleep(1200);
const second = await evaluate(ws, `return document.getElementById("clock").textContent;`);

expect(
  "bấm phát là máy bay tự bay, đồng hồ chạy",
  parseFloat(second) > parseFloat(first),
  `${first} -> ${second}`,
);

const moving = await evaluate(
  ws,
  `await new Promise(r => setTimeout(r, 400));
   return document.querySelector("#hud .hud-row b").textContent;`,
);
expect("độ cao trên HUD nhúc nhích theo thời gian", parseFloat(moving) > 0, `${moving}`);

// ---------------------------------------------------------------------
// 3. Tạm dừng
// ---------------------------------------------------------------------

await evaluate(ws, `document.getElementById("play").click(); return 1;`);
const pausedLabel = await evaluate(ws, `return document.getElementById("play").textContent;`);

const pauseA = await evaluate(ws, `return document.getElementById("clock").textContent;`);
await sleep(700);
const pauseB = await evaluate(ws, `return document.getElementById("clock").textContent;`);

expect("nút đổi thành Chạy tiếp khi tạm dừng", pausedLabel === "Chạy tiếp", pausedLabel);
expect("tạm dừng thì đồng hồ đứng yên", pauseA === pauseB, `${pauseA} = ${pauseB}`);

// ---------------------------------------------------------------------
// 4. Kéo thanh trượt
// ---------------------------------------------------------------------

const scrubbed = await evaluate(
  ws,
  `const s = document.getElementById("scrub");
   s.value = "600";
   s.dispatchEvent(new Event("input", { bubbles: true }));
   await new Promise(r => setTimeout(r, 200));
   return { clock: document.getElementById("clock").textContent,
            altitude: document.querySelector("#hud .hud-row b").textContent,
            trail: (() => { const l = document.querySelector("canvas"); return l ? 1 : 0; })() };`,
);

expect(
  "kéo thanh trượt thì nhảy tới đúng chỗ đó",
  parseFloat(scrubbed.clock) > parseFloat(pauseB),
  `kéo 60% -> ${scrubbed.clock}`,
);
expect(
  "kéo thanh trượt thì tạm dừng lại",
  (await evaluate(ws, `return document.getElementById("play").textContent;`)) === "Chạy tiếp",
);

// ---------------------------------------------------------------------
// 5. Đổi chuyến bay
// ---------------------------------------------------------------------

const switched = await evaluate(
  ws,
  `const p = document.getElementById("flight");
   p.value = "chuyentay_15m_01";
   p.dispatchEvent(new Event("change", { bubbles: true }));
   await new Promise(r => setTimeout(r, 900));
   return { peak: document.querySelectorAll("#summary dd")[0].textContent.trim(),
            landing: document.querySelectorAll("#summary dd")[1].textContent.trim(),
            clock: document.getElementById("clock").textContent };`,
);

expect(
  "đổi sang chuyến 15 m thì số liệu đổi theo",
  switched.peak.startsWith("16.4"),
  `lên ${switched.peak} · chạm đất ${switched.landing}`,
);

// ---------------------------------------------------------------------
// 6. Chạy lại từ đầu
// ---------------------------------------------------------------------

const restarted = await evaluate(
  ws,
  `document.getElementById("restart").click();
   await new Promise(r => setTimeout(r, 250));
   return { clock: document.getElementById("clock").textContent,
            label: document.getElementById("play").textContent };`,
);

expect(
  "bấm Chạy lại thì về giây 0 và tự chạy tiếp",
  parseFloat(restarted.clock) < 1.5 && restarted.label === "Tạm dừng",
  `${restarted.clock} · nút "${restarted.label}"`,
);

// ---------------------------------------------------------------------
// 7. Các công tắc
// ---------------------------------------------------------------------

const toggles = await evaluate(
  ws,
  `for (const id of ["trail", "spin", "slow"]) {
     const box = document.getElementById(id);
     box.checked = !box.checked;
     box.dispatchEvent(new Event("change", { bubbles: true }));
   }
   await new Promise(r => setTimeout(r, 400));
   for (const id of ["trail", "spin", "slow"]) {
     const box = document.getElementById(id);
     box.checked = !box.checked;
     box.dispatchEvent(new Event("change", { bubbles: true }));
   }
   await new Promise(r => setTimeout(r, 500));
   return document.getElementById("clock").textContent;`,
);

expect("bật/tắt ba công tắc không làm trang hỏng", toggles !== null, `đồng hồ ${toggles}`);

// ---------------------------------------------------------------------
// 8. Bay hết chuyến thì tự dừng
// ---------------------------------------------------------------------

const ended = await evaluate(
  ws,
  `const p = document.getElementById("flight");
   p.value = "chuyentay_10m_01";
   p.dispatchEvent(new Event("change", { bubbles: true }));

   // Phải CHỜ nạp xong. Nạp chuyến bay là việc bất đồng bộ, mà lúc nó xong
   // thì nó đặt lại đồng hồ về 0 — nếu kéo thanh trượt ngay lập tức thì
   // thao tác đó bị chính cú nạp kia xoá đi.
   for (let i = 0; i < 40; i++) {
     await new Promise(r => setTimeout(r, 100));
     const peak = document.querySelectorAll("#summary dd")[0].textContent.trim();
     if (peak.startsWith("11.3")) break;
   }

   const s = document.getElementById("scrub");
   s.value = "995";
   s.dispatchEvent(new Event("input", { bubbles: true }));
   await new Promise(r => setTimeout(r, 100));

   document.getElementById("play").click();
   await new Promise(r => setTimeout(r, 1200));

   return { clock: document.getElementById("clock").textContent,
            label: document.getElementById("play").textContent,
            altitude: document.querySelector("#hud .hud-row b").textContent };`,
);

expect(
  "bay tới cuối chuyến thì tự dừng lại ở mặt đất",
  ended.label === "Chạy tiếp" && parseFloat(ended.altitude) < 0.2,
  `${ended.clock} · cao ${ended.altitude} · nút "${ended.label}"`,
);

// ---------------------------------------------------------------------
// 9. Đổi chuyến bay liên tiếp
// ---------------------------------------------------------------------

// Giữ phím mũi tên trong ô chọn là bắn ra một loạt sự kiện, mà nạp chuyến
// bay thì bất đồng bộ. Không chặn câu trả lời về muộn thì trang sẽ hiện
// số liệu của chuyến này trong khi ô chọn ghi chuyến khác.
const race = await evaluate(
  ws,
  `const p = document.getElementById("flight");
   for (const name of ["chuyentay_15m_03", "chuyentay_10m_02", "chuyentay_12m_01", "chuyentay_15m_01"]) {
     p.value = name;
     p.dispatchEvent(new Event("change", { bubbles: true }));
   }
   await new Promise(r => setTimeout(r, 1500));
   return { chosen: p.value,
            peak: document.querySelectorAll("#summary dd")[0].textContent.trim() };`,
);

expect(
  "đổi chuyến bay liên tiếp thì số liệu khớp với chuyến đang chọn",
  race.chosen === "chuyentay_15m_01" && race.peak.startsWith("16.4"),
  `ô chọn "${race.chosen}" · bảng ghi lên ${race.peak}`,
);

// ---------------------------------------------------------------------
// 10. Vệt bay có thật sự được VẼ ra không
// ---------------------------------------------------------------------

// Đây là bài học đắt nhất của cả trình xem. `setFromPoints` nghe như "thay
// cả danh sách điểm", nhưng thật ra nó ghi vào đệm ĐÃ CÓ và chỉ ghi được
// bằng số điểm mà đệm đó đã chứa. Gọi lần đầu với 1 điểm là đệm chỉ còn 1
// chỗ, và đường vẽ 1 điểm thì không vẽ ra gì — vệt bay không bao giờ hiện,
// dù mã nguồn trông rất hợp lý.
//
// Nên phải kiểm tra bằng ảnh chụp thật: tắt vệt bay đi rồi chụp lại, hai
// ảnh phải KHÁC nhau. Nếu giống hệt thì có nghĩa là vệt bay chưa từng được
// vẽ, và mọi thứ khác vẫn "đạt" một cách vô nghĩa.
const painted = await evaluate(
  ws,
  `const p = document.getElementById("flight");
   p.value = "chuyentay_10m_01";
   p.dispatchEvent(new Event("change", { bubbles: true }));
   for (let i = 0; i < 40; i++) {
     await new Promise(r => setTimeout(r, 100));
     const first = document.querySelector("#summary dd");
     if (first && first.textContent.trim().startsWith("11.3")) break;
   }

   // Tua tới 80% rồi để yên — ảnh chụp phải đứng yên mới so được.
   const s = document.getElementById("scrub");
   s.value = "800";
   s.dispatchEvent(new Event("input", { bubbles: true }));
   await new Promise(r => setTimeout(r, 500));

   const box = document.getElementById("trail");
   box.checked = true;
   box.dispatchEvent(new Event("change", { bubbles: true }));
   await new Promise(r => setTimeout(r, 300));
   return 1;`,
);

const shotWith = await send(ws, "Page.captureScreenshot", { format: "png" });

await evaluate(
  ws,
  `const box = document.getElementById("trail");
   box.checked = false;
   box.dispatchEvent(new Event("change", { bubbles: true }));
   await new Promise(r => setTimeout(r, 300));
   return 1;`,
);

const shotWithout = await send(ws, "Page.captureScreenshot", { format: "png" });

expect(
  "vệt bay thật sự được vẽ ra (tắt đi thì ảnh phải khác)",
  shotWith.data !== shotWithout.data,
  shotWith.data === shotWithout.data
    ? "hai ảnh GIỐNG HỆT NHAU — vệt bay chưa từng được vẽ"
    : `ảnh khác nhau ${Math.abs(shotWith.data.length - shotWithout.data.length)} byte`,
);

// Bật lại cho những lần sau.
await evaluate(
  ws,
  `const box = document.getElementById("trail");
   box.checked = true;
   box.dispatchEvent(new Event("change", { bubbles: true }));
   return 1;`,
);

expect("chụp được ảnh cảnh 3D để so", painted === 1, `${shotWith.data.length} byte`);

// ---------------------------------------------------------------------

console.log("\nThử trình xem 3D bằng cách bấm thật\n" + "=".repeat(58));

let failed = 0;

for (const [name, ok, detail] of results) {
  if (!ok) failed += 1;
  console.log(`  ${ok ? "[đạt] " : "[HỎNG]"} ${name}` + (detail ? `  (${detail})` : ""));
}

if (problems.length) {
  console.log("\nLỗi bắt được trong trang:");
  for (const problem of new Set(problems)) console.log("  - " + problem);
  failed += problems.length;
}

console.log("\n" + "=".repeat(58));
console.log(`${results.length - failed}/${results.length} mục đạt`);

ws.close();
process.exit(failed ? 1 : 0);
