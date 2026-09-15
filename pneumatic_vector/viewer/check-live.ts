/**
 * check-live.ts — Chế độ CHẠY SỐNG có thật sự chạy sống không?
 *
 * Câu hỏi không phải "trang có mở được không" mà là:
 *
 *   1. Vật lý có chạy ĐÚNG 1.000 bước mỗi giây thời gian thật không?
 *   2. Model trong trình duyệt có thật sự viết ra lệnh không?
 *   3. Kéo thanh chỉnh tham số thì con tàu có bay khác đi thật không?
 *   4. Máy có theo kịp không, hay là bị đuối mà vẫn giả vờ?
 *
 * Chạy:  bun run check-live.ts
 */

const PORT = 9333;
const URL = "http://localhost:4180/";

let id = 0;
const pending = new Map<number, (v: any) => void>();
const problems: string[] = [];

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
  if (reply.exceptionDetails) {
    throw new Error(reply.exceptionDetails.exception?.description ?? "lỗi trong trang");
  }
  return reply.result?.value;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const list = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()) as any[];
const page = list.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));

ws.onmessage = (e) => {
  const d = JSON.parse(String(e.data));
  if (d.id && pending.has(d.id)) {
    pending.get(d.id)!(d.result ?? d);
    pending.delete(d.id);
  }
  if (d.method === "Runtime.exceptionThrown") {
    problems.push(d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text);
  }
  if (d.method === "Runtime.consoleAPICalled" && d.params.type === "error") {
    problems.push("console.error: " + d.params.args.map((a: any) => a.value).join(" "));
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

const results: [string, boolean, string][] = [];
const expect = (name: string, ok: boolean, detail: string) => results.push([name, ok, detail]);

// --- Chờ trang sống chạy ---

let ready = false;
for (let attempt = 0; attempt < 50; attempt += 1) {
  await sleep(400);
  try {
    // Kiểm tra theo TÊN thanh trượt, không đếm số lượng — thêm thanh mới là
    // chuyện bình thường, còn thiếu thanh thì mới là hỏng.
    const ok = await evaluate(
      ws,
      `return ["targetAltitude","dryMass","pressureBar","gimbalMaxDeg",
                "aeroTorqueGain","finAreaCm2"]
         .every(k => document.getElementById("sl-" + k));`,
    );
    if (ok) {
      ready = true;
      break;
    }
  } catch {
    // trang đang được thay
  }
}

if (!ready) {
  console.error("Trang không nạp được chế độ sống. Máy chủ đã chạy chưa?");
  process.exit(1);
}

await sleep(1200);

// =====================================================================
// 1. Vật lý có chạy đúng 1.000 Hz không
// =====================================================================

console.log("\nChế độ CHẠY SỐNG — vật lý và model trong trình duyệt");
console.log("=".repeat(64));

/** Đọc số bước vật lý đã chạy từ dòng trạng thái. */
async function steps(): Promise<number> {
  const text = await evaluate(ws, `return document.getElementById("live-status").textContent;`);
  const match = /([\d.]+)\s*bước/.exec(String(text).replace(/\./g, ""));
  return match ? Number(match[1]) : -1;
}

{
  await evaluate(ws, `document.getElementById("live-reset").click(); return 1;`);
  await sleep(400);

  const before = await steps();
  const t0 = Date.now();
  await sleep(3000);
  const after = await steps();
  const elapsed = (Date.now() - t0) / 1000;

  const rate = (after - before) / elapsed;

  expect(
    "vật lý chạy thật trong trình duyệt, đúng 1.000 bước mỗi giây",
    rate > 900 && rate < 1100,
    `${Math.round(rate)} bước/giây thời gian thật (${after - before} bước trong ${elapsed.toFixed(2)} s)`,
  );

  const hud = await evaluate(
    ws,
    `return { t: document.querySelectorAll("#hud .hud-row b")[0].textContent,
              z: document.querySelectorAll("#hud .hud-row b")[1].textContent };`,
  );

  const t = parseFloat(hud.t);
  expect(
    "đồng hồ mô phỏng khớp với thời gian thật",
    Math.abs(t - elapsed - 0.4) < 0.5,
    `mô phỏng ${hud.t} · thật ${elapsed.toFixed(2)} s · cao ${hud.z}`,
  );
}

// =====================================================================
// 2. Kéo thanh chỉnh tham số thì con tàu bay khác đi thật
// =====================================================================

{
  async function setSlider(key: string, value: number | string) {
    await evaluate(
      ws,
      `const s = document.getElementById("sl-${key}");
       s.value = "${value}";
       s.dispatchEvent(new Event("input", { bubbles: true }));
       return 1;`,
    );
    await sleep(300);
  }

  const readHud = () =>
    evaluate(
      ws,
      `return { z: parseFloat(document.querySelectorAll("#hud .hud-row b")[1].textContent),
                pressure: parseFloat(document.querySelectorAll("#hud .hud-row b")[4].textContent),
                status: document.getElementById("live-status").textContent };`,
    );

  /**
   * Bay tới khi chạm đất, trả về độ cao lớn nhất.
   *
   * Phải CHỜ dòng trạng thái hết kết quả cũ trước đã. Sau khi kéo thanh, dòng
   * đó vẫn đang ghi kết quả của chuyến vừa rồi — đọc ngay là đọc nhầm sang
   * chuyến cũ, và bài kiểm tra sẽ tưởng chuyến mới đã bay xong.
   */
  async function waitForFreshFlight() {
    for (let attempt = 0; attempt < 30; attempt += 1) {
      const status = String(
        await evaluate(ws, `return document.getElementById("live-status").textContent;`),
      );
      if (!/ĐẠT|HỎNG/.test(status)) return;
      await sleep(200);
    }
  }

  async function apex(): Promise<number> {
    await waitForFreshFlight();

    let peak = 0;

    for (let attempt = 0; attempt < 70; attempt += 1) {
      await sleep(400);

      const hud = await readHud();
      peak = Math.max(peak, hud.z);

      if (/ĐẠT|HỎNG/.test(String(hud.status))) break;
    }

    return peak;
  }

  // --- Đề bài: đây là phép thử rõ ràng nhất, vì độ cao đỉnh là hệ quả trực
  //     tiếp của đề bài.
  await setSlider("targetAltitude", 8);
  const low = await apex();

  await setSlider("targetAltitude", 17);
  const high = await apex();

  expect(
    "kéo đề bài từ 8 m lên 17 m thì con tàu bay cao hơn hẳn",
    high > low + 5,
    `đề bài 8 m -> đỉnh ${low.toFixed(2)} m · đề bài 17 m -> đỉnh ${high.toFixed(2)} m`,
  );

  // --- Khối lượng vỏ: KHÔNG so độ cao.
  //
  // Bộ điều khiển giữ vận tốc leo ở 1,8 m/s bất kể con tàu nặng bao nhiêu,
  // nên ở giây thứ hai hai con tàu ở gần cùng một độ cao — đó là hành vi
  // ĐÚNG. Cái khác nhau là phải mở van to hơn, tức là tốn khí hơn.
  await setSlider("targetAltitude", 12);
  await setSlider("dryMass", 0.35);
  await sleep(2500);
  const lightTank = (await readHud()).pressure;

  await setSlider("dryMass", 1.15);
  await sleep(2500);
  const heavyTank = (await readHud()).pressure;

  expect(
    "vỏ nặng gấp ba thì tốn khí hơn hẳn trong cùng khoảng thời gian",
    heavyTank < lightTank - 0.3,
    `vỏ 0,35 kg -> còn ${lightTank} bar · vỏ 1,15 kg -> còn ${heavyTank} bar`,
  );

  // --- Áp suất nạp: so lượng khí mang theo, không so độ cao (cùng lý do).
  await setSlider("dryMass", 0.6);
  await setSlider("pressureBar", 4.5);
  await sleep(2500);
  const weakTank = (await readHud()).pressure;

  await setSlider("pressureBar", 15.0);
  await sleep(2500);
  const strongTank = (await readHud()).pressure;

  expect(
    "nạp 15 bar thì bay được lâu hơn hẳn so với 4,5 bar",
    strongTank > weakTank * 2,
    `nạp 4,5 bar -> còn ${weakTank} bar · nạp 15 bar -> còn ${strongTank} bar`,
  );

  await setSlider("pressureBar", 10);
  await setSlider("dryMass", 0.6);
  await setSlider("targetAltitude", 12);
}

// =====================================================================
// 3. Mất ổn định: tắt đi thì con tàu đứng yên hơn hẳn
// =====================================================================

{
  /** Bay hết một chuyến, trả về nghiêng tối đa và độ cao đỉnh. */
  async function fullFlight(aero: number) {
    await evaluate(
      ws,
      `const s = document.getElementById("sl-aeroTorqueGain");
       s.value = "${aero}";
       s.dispatchEvent(new Event("input", { bubbles: true }));
       return 1;`,
    );
    await sleep(200);

    // Chờ kết quả cũ biến mất trước khi chờ kết quả mới — nếu không thì đọc
    // ngay phải kết quả của chuyến trước.
    for (let attempt = 0; attempt < 30; attempt += 1) {
      const status = String(
        await evaluate(ws, `return document.getElementById("live-status").textContent;`),
      );
      if (!/ĐẠT|HỎNG/.test(status)) break;
      await sleep(200);
    }

    for (let attempt = 0; attempt < 70; attempt += 1) {
      await sleep(400);

      const done = await evaluate(
        ws,
        `return /ĐẠT|HỎNG/.test(document.getElementById("live-status").textContent);`,
      );
      if (done) break;
    }

    return evaluate(
      ws,
      `const dd = [...document.querySelectorAll("#summary div")];
       const find = (name) => {
         const row = dd.find(d => d.querySelector("dt")?.textContent === name);
         return parseFloat(row?.querySelector("dd")?.textContent ?? "0");
       };
       return { tilt: find("Nghiêng tối đa"), peak: find("Lên cao nhất"),
                target: document.getElementById("sl-targetAltitude").value,
                landed: find("Chạm đất") };`,
    );
  }

  // Đo cả chuyến chứ không đo hai giây đầu.
  //
  // Bản đầu tiên của bài kiểm tra này đo nghiêng tối đa trong 2,5 giây đầu,
  // và kết quả ngược lại: mất ổn định 3,5 -> 2,00°, mất ổn định 0 -> 2,10°.
  // Không phải vì mô phỏng sai, mà vì con tàu xuất phát đã nghiêng sẵn 2,0°,
  // nên "nghiêng tối đa" trong hai giây đầu chỉ là dao động ban đầu ấy. Mô-men
  // khí động tỉ lệ với bình phương vận tốc, mà lúc leo thì con tàu chỉ đi
  // 1,8 m/s — phải tới lúc rơi tự do 11 m/s nó mới thành đáng kể.
  const stable = await fullFlight(0.0);
  const unstable = await fullFlight(3.5);

  expect(
    "tắt mất ổn định khí động thì cả chuyến bay nghiêng ít hơn",
    unstable.tilt > stable.tilt,
    `đề bài ${stable.target} m · mất ổn định 0 -> nghiêng tối đa ${stable.tilt}° ` +
      `(đỉnh ${stable.peak} m) · mất ổn định 3,5 -> ${unstable.tilt}° (đỉnh ${unstable.peak} m)`,
  );

  await evaluate(
    ws,
    `const s = document.getElementById("sl-aeroTorqueGain");
     s.value = "1.4";
     s.dispatchEvent(new Event("input", { bubbles: true }));
     return 1;`,
  );
}

// =====================================================================
// 3b. Cánh đuôi — câu hỏi thiết kế, trả lời được ngay trên màn hình
// =====================================================================

{
  const readStability = () =>
    evaluate(
      ws,
      `const row = [...document.querySelectorAll("#summary div")]
         .find(d => d.querySelector("dt")?.textContent === "Ổn định");
       return row?.querySelector("dd")?.textContent?.trim() ?? "";`,
    );

  async function setFin(cm2: number) {
    await evaluate(
      ws,
      `const s = document.getElementById("sl-finAreaCm2");
       s.value = "${cm2}";
       s.dispatchEvent(new Event("input", { bubbles: true }));
       return 1;`,
    );
    await sleep(700);
  }

  await setFin(0);
  const bare = await readStability();

  await setFin(200);
  const finned = await readStability();

  expect(
    "không cánh thì bảng báo MẤT ổn định, kèm số cánh cần có",
    /mất ổn định/.test(bare) && /\d+ cm²/.test(bare),
    bare,
  );
  expect(
    "gắn đủ cánh thì bảng báo TỰ DỰNG LẠI",
    /tự dựng lại/.test(finned),
    finned,
  );

  // Cánh phải thật sự đổi con tàu, không chỉ đổi dòng chữ trên bảng.
  //
  // Đo NGHIÊNG TỐI ĐA CẢ CHUYẾN, và đo với ĐỘ MẤT ỔN ĐỊNH CAO.
  //
  // Hai lần đo hỏng trước đó, ghi lại để khỏi lặp lại:
  //
  //   - Lần đầu chỉ lấy mẫu trong lúc ở giai đoạn ROI. Trong chuyến bay CÓ
  //     điều khiển, con tàu vào rơi tự do gần như thẳng (0,2°) và chỉ kịp lệch
  //     tới 1,9°, nên không phân biệt được gì. Con số 178° ấn tượng kia chỉ có
  //     khi THẢ RƠI không điều khiển — chế độ sống không làm được điều đó.
  //   - Lần thứ hai đo nghiêng tối đa cả chuyến ở độ mất ổn định mặc định:
  //     ra 2,4° cả hai bên. Vì con tàu xuất phát đã nghiêng sẵn 2,0°, nên
  //     "nghiêng tối đa" bị chặn dưới bởi chính điều kiện đầu, và cánh không
  //     thể làm nó nhỏ hơn 2,0°.
  //
  // Nên phải đẩy mất ổn định lên 3,5 để nó thành thứ đáng kể, rồi mới so.
  async function flightWithFins(cm2: number) {
    await evaluate(
      ws,
      `const a = document.getElementById("sl-aeroTorqueGain");
       a.value = "3.5";
       a.dispatchEvent(new Event("input", { bubbles: true }));
       return 1;`,
    );
    await setFin(cm2);
    await evaluate(ws, `document.getElementById("live-reset").click(); return 1;`);
    await sleep(300);

    for (let attempt = 0; attempt < 60; attempt += 1) {
      const status = String(
        await evaluate(ws, `return document.getElementById("live-status").textContent;`),
      );
      if (!/ĐẠT|HỎNG/.test(status)) break;
      await sleep(200);
    }

    for (let attempt = 0; attempt < 70; attempt += 1) {
      await sleep(400);
      const done = await evaluate(
        ws,
        `return /ĐẠT|HỎNG/.test(document.getElementById("live-status").textContent);`,
      );
      if (done) break;
    }

    return evaluate(
      ws,
      `const dd = [...document.querySelectorAll("#summary div")];
       const find = (n) => {
         const row = dd.find(d => d.querySelector("dt")?.textContent === n);
         return row?.querySelector("dd")?.textContent?.trim() ?? "";
       };
       return { tilt: parseFloat(find("Nghiêng tối đa")),
                neutral: find("Ổn định"),
                outcome: find("Chạm đất") };`,
    );
  }

  const withoutFins = await flightWithFins(0);
  const withFins = await flightWithFins(200);

  expect(
    "cánh làm con tàu thẳng hơn hẳn khi mất ổn định cao",
    withFins.tilt < withoutFins.tilt - 0.5,
    `mất ổn định 3,5 · không cánh nghiêng ${withoutFins.tilt}° · ` +
      `cánh 200 cm² mỗi bên ${withFins.tilt}° (chạm đất ${withoutFins.outcome} và ${withFins.outcome})`,
  );

  await setFin(0);
  await evaluate(
    ws,
    `const a = document.getElementById("sl-aeroTorqueGain");
     a.value = "1.4";
     a.dispatchEvent(new Event("input", { bubbles: true }));
     return 1;`,
  );
}

// =====================================================================
// 4. Model trong trình duyệt
// =====================================================================

{
  await evaluate(
    ws,
    `const p = document.getElementById("policy");
     p.value = "model";
     p.dispatchEvent(new Event("change", { bubbles: true }));
     return 1;`,
  );

  // Nạp 1,5 MB rồi còn phải chạy forward pass nữa, nên chờ rộng rãi.
  //
  // Không chờ dòng trạng thái báo "Đã nạp model": dòng đó bị vòng vẽ ghi đè
  // sau vài trăm mili-giây. Chờ theo thứ BỀN VỮNG hơn — chính sách đang chọn
  // và việc model có thật sự ra lệnh hay không.
  let loaded = false;
  let status = "";

  for (let attempt = 0; attempt < 60; attempt += 1) {
    await sleep(500);

    status = String(
      await evaluate(ws, `return document.getElementById("live-status").textContent;`),
    );

    const policy = await evaluate(ws, `return document.getElementById("policy").value;`);
    const choices = await evaluate(
      ws,
      `return document.querySelectorAll("#decisions .decision").length;`,
    );

    if (policy === "model" && /model trong trình duyệt/.test(status) && choices > 0) {
      loaded = true;
      break;
    }
  }

  expect(
    "nạp được trọng số model và model thật sự ra lệnh trong trình duyệt",
    loaded,
    status.trim() || "(chưa thấy model ra lệnh)",
  );

  await evaluate(ws, `document.getElementById("live-reset").click(); return 1;`);
  await sleep(3500);

  status = String(await evaluate(ws, `return document.getElementById("live-status").textContent;`));
  const ms = /([\d.]+)\s*ms/.exec(status);

  expect(
    "dòng trạng thái nói rõ đang dùng model, kèm chi phí mỗi lần quyết định",
    /model trong trình duyệt/.test(status) && Number(ms?.[1] ?? 0) > 0,
    status.trim(),
  );

  // 20 lần mỗi giây, và mỗi lần tốn vài mili-giây — nếu chậm quá thì trình
  // duyệt không theo kịp và chế độ sống chỉ là giả vờ.
  const decisionMs = Number(ms?.[1] ?? 0);
  expect(
    "mỗi lần model ra quyết định đủ nhanh để chạy 20 Hz",
    decisionMs > 0 && decisionMs < 50,
    `${decisionMs} ms mỗi lần · 20 Hz cần dưới 50 ms`,
  );

  const phases = await evaluate(
    ws,
    `return [...document.querySelectorAll("#decisions .dcmd")].map(d => d.textContent.trim());`,
  );

  expect(
    "model tự chọn ra các giai đoạn bay, không phải kịch bản có sẵn",
    phases.length >= 2,
    phases.join(" > "),
  );

  const hud = await evaluate(
    ws,
    `return { z: document.querySelectorAll("#hud .hud-row b")[1].textContent,
              cmd: document.querySelector("#hud .cmd").textContent };`,
  );

  expect(
    "con tàu bay lên được dưới sự điều khiển của model",
    parseFloat(hud.z) > 3,
    `cao ${hud.z} · lệnh "${hud.cmd}"`,
  );
}

// =====================================================================
// 5. Bay hết chuyến thì dừng và báo kết quả
// =====================================================================

{
  await evaluate(ws, `document.getElementById("live-reset").click(); return 1;`);

  let finished = false;
  let status = "";

  for (let attempt = 0; attempt < 40; attempt += 1) {
    await sleep(500);
    status = String(await evaluate(ws, `return document.getElementById("live-status").textContent;`));
    if (/ĐẠT|HỎNG/.test(status)) {
      finished = true;
      break;
    }
  }

  expect("bay hết chuyến thì tự dừng và báo kết quả", finished, status.trim());

  const button = await evaluate(ws, `return document.getElementById("live-run").textContent;`);
  expect(
    "bay xong thì nút ghi đúng việc nó làm: Bay lại",
    button === "Bay lại",
    `nút "${button}"`,
  );
}

// =====================================================================

console.log("\n" + "=".repeat(64));

let failed = 0;
for (const [name, ok, detail] of results) {
  if (!ok) failed += 1;
  console.log(`  ${ok ? "[đạt] " : "[HỎNG]"} ${name}` + (detail ? `  (${detail})` : ""));
}

if (problems.length) {
  console.log("\nLỗi bắt được trong trang:");
  for (const p of new Set(problems)) console.log("  - " + p);
  failed += problems.length;
}

console.log("\n" + "=".repeat(64));
console.log(`${results.length - Math.min(failed, results.length)}/${results.length} mục đạt`);

ws.close();
process.exit(failed ? 1 : 0);
