/**
 * check-port.ts — Bản TypeScript có thật sự giống bản Python không?
 *
 * Đây là câu hỏi duy nhất mà file này trả lời. Không phải "trông có vẻ chạy",
 * mà là: chạy đúng cùng một việc ở cả hai bên rồi so từng bước.
 *
 * Bốn tầng, mỗi tầng kiểm tra một phần:
 *
 *     1. vật lý thuần   4.000 bước, so từng bước
 *     2. chuyên gia     cả chuyến bay, so từng bước (gồm cả bộ điều khiển)
 *     3. model          cùng câu, phải viết ra cùng lệnh
 *     4. đầu-cuối       cả chuyến bay do model lái, so từng bước
 *
 * Chạy:  bun run check-port.ts
 * (phải chạy `python -m pneumatic_vector export` trước)
 */

import { readFileSync } from "node:fs";

import {
  CascadedController,
  ExpertPilotAdapter,
  Tank,
  commandFromScript,
  dragArea,
  fly,
  makeMission,
  makeState,
  makeVehicle,
  stability as stabilityOf,
  step,
  type StepRecord,
  type State,
} from "./src/physics";
import { LearnedPilot, PilotModel, type Weights } from "./src/model";

const HERE = import.meta.dir;

/** Đọc một file, và nếu chưa có thì nói rõ phải chạy gì — đừng để lỗi khó hiểu. */
function readJson<T>(path: string, label: string): T {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as T;
  } catch {
    console.error(
      `Chưa có ${label} ở ${path}.\n\n` +
        "Hai file này do Python sinh ra. Chạy trước:\n" +
        "  python -m pneumatic_vector export\n",
    );
    process.exit(1);
  }
}

const weights = readJson<Weights>(`${HERE}/public/model.json`, "trọng số model");
const reference = readJson<any>(`${HERE}/public/reference.json`, "dữ liệu đối chiếu");

let passed = 0;
const failures: string[] = [];

function check(name: string, ok: boolean, detail = "") {
  if (ok) {
    passed += 1;
    console.log(`  [đạt]  ${name}${detail ? `  (${detail})` : ""}`);
  } else {
    failures.push(name);
    console.log(`  [HỎNG] ${name}${detail ? `  (${detail})` : ""}`);
  }
}

/**
 * So hai dãy số theo kiểu `allclose`: đạt khi
 *
 *     |a - b|  <=  atol + rtol · |b|
 *
 * Không dùng riêng sai số tương đối. Có những giá trị gần bằng 0 (ví dụ `y`
 * ở bước đầu, cỡ 1e-7) mà sai số tương đối của chúng luôn trông khủng khiếp
 * dù sai số tuyệt đối chỉ là 5e-9 — chia một số bé cho một số bé thì được
 * một tỉ lệ vô nghĩa. Cũng không dùng riêng sai số tuyệt đối, vì có những
 * giá trị cỡ 1e11.
 */
const ATOL = 1e-6;
const RTOL = 1e-9;

function compare(mine: number[], ref: number[]) {
  let worstRatio = 0;
  let worstAt = 0;
  let worstAbs = 0;
  let worstRel = 0;

  for (let i = 0; i < mine.length; i += 1) {
    const abs = Math.abs(mine[i] - ref[i]);
    const allowed = ATOL + RTOL * Math.abs(ref[i]);
    const ratio = abs / allowed;

    if (ratio > worstRatio) {
      worstRatio = ratio;
      worstAt = i;
    }

    worstAbs = Math.max(worstAbs, abs);

    // Chỉ tính sai số tương đối trên những giá trị đủ lớn để phép chia có
    // nghĩa.
    if (Math.abs(ref[i]) > 1e-3) {
      worstRel = Math.max(worstRel, abs / Math.abs(ref[i]));
    }
  }

  return { worstRatio, worstAt, worstAbs, worstRel, ok: worstRatio <= 1 };
}

/** So hai dãy số, trả về sai lệch tuyệt đối lớn nhất. */
function maxDiff(a: number[], b: number[]): number {
  let worst = 0;
  for (let i = 0; i < a.length; i += 1) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

const fmt = (x: number) => (x === 0 ? "0" : x.toExponential(2));

/** Tên các cột trong bảng vật lý, để báo lỗi chỉ đúng chỗ. */
const SCALAR_COLUMNS = [
  "t", "x", "y", "z", "vx", "vy", "vz",
  "theta", "phi", "omega_theta", "omega_phi", "thrust", "pressure",
];

// =====================================================================
// 1. Vật lý thuần
// =====================================================================

console.log("\nVật lý — 4.000 bước, so từng bước với Python");
console.log("=".repeat(64));

{
  const v = makeVehicle();
  const tank = new Tank(v);
  let state = makeState({ theta: 0.03, phi: -0.02 });

  const rows: number[][] = [];
  const mine: number[][] = [];

  for (let index = 0; index < reference.physics.steps; index += 1) {
    const command = commandFromScript(index, reference.physics.steps, state);
    const record = step(state, v, tank, command);
    state = record.state;

    mine.push([
      state.t, state.x, state.y, state.z,
      state.vx, state.vy, state.vz,
      state.theta, state.phi,
      state.omegaTheta, state.omegaPhi,
      record.thrust, tank.pressure,
    ]);

    rows.push(reference.physics.rows[index]);
  }

  const flatMine = mine.flat();
  const flatRef = rows.flat();

  // So bằng `allclose`, không so riêng sai số tuyệt đối hay tương đối.
  //
  // Sai số tuyệt đối một mình không nói lên điều gì: hệ này bất ổn, nên chỉ
  // cần một chút làm tròn khác nhau là hai bên tách dần. Đo được: khi bài
  // kiểm tra còn lái vòi phun bằng sin cố định, con tàu lộn nhào và tới bước
  // 4.000 thì sai số lên 61 — nhưng đó là 61 trên `omega_theta` cỡ 2,4e11,
  // tức là lệch 2,5e-10.
  //
  // Còn sai số tương đối một mình thì hỏng ở chiều ngược lại: `y` ở bước đầu
  // cỡ 1e-7, lệch 5e-9 đã thành 3e-5 "phần trăm" — nghe ghê mà thật ra chỉ
  // là làm tròn.
  const cmp = compare(flatMine, flatRef);
  const worst = {
    column: SCALAR_COLUMNS[cmp.worstAt % 13],
    step: Math.floor(cmp.worstAt / 13),
  };

  check(
    "40.000 con số của 4.000 bước vật lý khớp với Python",
    cmp.ok,
    `lệch tuyệt đối ${fmt(cmp.worstAbs)} · tương đối ${fmt(cmp.worstRel)}` +
      (cmp.ok ? "" : ` — vượt mức cho phép ở bước ${worst.step}, cột ${worst.column}`),
  );

  check(
    "áp suất bình khí cuối cùng khớp",
    Math.abs(tank.pressure - reference.physics.rows[reference.physics.steps - 1][12]) < 1e-6,
    `${(tank.pressure / 1e5).toFixed(6)} bar`,
  );

  check(
    "cả 4.000 bước đều hữu hạn (không có NaN lọt vào)",
    flatMine.every(Number.isFinite),
  );
}

// =====================================================================
// 1b. Vật lý CÓ CÁNH ĐUÔI
// =====================================================================

console.log("\nVật lý có cánh đuôi — nhánh mô-men đổi dấu");
console.log("=".repeat(64));

{
  const fins = reference.physics_with_fins;
  const v = makeVehicle({ finArea: fins.fin_area, finOffset: fins.fin_offset });

  // Hai con số dẫn xuất phải khớp trước đã — nếu công thức lệch thì so 4.000
  // bước cũng vô nghĩa.
  check(
    "hệ số ổn định tính ra khớp (âm = tự dựng lại)",
    Math.abs(stabilityOf(v) - fins.stability) < 1e-9,
    `TS ${stabilityOf(v).toExponential(4)} · Python ${fins.stability.toExponential(4)}`,
  );
  check(
    "diện tích cản tính ra khớp (cánh làm tăng)",
    Math.abs(dragArea(v) - fins.drag_area) < 1e-9,
    `TS ${(dragArea(v) * 1e4).toFixed(2)} cm² · Python ${(fins.drag_area * 1e4).toFixed(2)} cm²`,
  );
  check(
    "cánh đủ lớn để con tàu TĨNH ỔN ĐỊNH",
    stabilityOf(v) < 0,
    `${stabilityOf(v).toExponential(4)} m³ < 0`,
  );

  const tank = new Tank(v);
  let state = makeState({ theta: 0.03, phi: -0.02 });
  const mine: number[] = [];

  for (let index = 0; index < fins.steps; index += 1) {
    const command = commandFromScript(index, fins.steps, state);
    const record = step(state, v, tank, command);
    state = record.state;

    mine.push(
      state.t, state.x, state.y, state.z,
      state.vx, state.vy, state.vz,
      state.theta, state.phi,
      state.omegaTheta, state.omegaPhi,
      record.thrust, tank.pressure,
    );
  }

  const cmp = compare(mine, fins.rows.flat());

  check(
    "40.000 con số của 4.000 bước CÓ CÁNH khớp với Python",
    cmp.ok,
    `lệch tuyệt đối ${fmt(cmp.worstAbs)} · tương đối ${fmt(cmp.worstRel)}`,
  );
}

// =====================================================================
// 2. Chuyên gia + bộ điều khiển
// =====================================================================

console.log("\nChuyên gia viết tay — cả chuyến bay, gồm cả bộ điều khiển PID");
console.log("=".repeat(64));

{
  const v = makeVehicle();
  const mission = makeMission({ targetAltitude: 12.0 });
  const policy = new ExpertPilotAdapter(v, mission);

  // Dùng ĐÚNG nghiêng ban đầu mà Python đã sinh ra. Không chép lại RNG của
  // PyTorch — chỉ cần hai bên xuất phát từ cùng một chỗ là đủ để so vật lý.
  const result = fly(v, policy, mission, {
    initialTilt: reference.expert.initial_tilt,
    keepSamples: false,
  });

  const mine: number[][] = [];

  {
    // Chạy lại lần nữa nhưng ghi từng bước, để so với bản Python.
    const tank = new Tank(v);
    const controller = new CascadedController(v);
    const pilot = new ExpertPilotAdapter(v, mission);

    let state = makeState({
      theta: reference.expert.initial_tilt,
      phi: -reference.expert.initial_tilt * 0.7,
    });

    let target = { altitude: 12.0, climbRate: 1.8, level: true, cut: false, descentProfile: 0 };
    let decision = { command: "LEN", value: 12.0 };
    let previousZ = state.z;

    const decisionEvery = 50;

    for (let index = 0; index < 25000; index += 1) {
      if (index % decisionEvery === 0) {
        decision = pilot.decide(state, tank);
        target = decisionTargetLocal(decision, 12.0);
      }

      const command = controller.call(state, tank, target);
      const record: StepRecord = step(state, v, tank, command);

      const previousVz = state.vz;
      state = record.state;

      mine.push([
        state.t, state.z, state.vz, state.theta, state.phi,
        record.thrust, record.command.throttle, record.command.gimbalTheta, tank.pressure,
      ]);

      if (state.z <= 0 && previousZ > 0) {
        void previousVz;
        break;
      }

      previousZ = state.z;
    }
  }

  const diff = maxDiff(mine.flat(), reference.expert.rows.flat());

  check(
    "từng bước của chuyến bay chuyên gia khớp với Python",
    diff < 1e-6,
    `${mine.length} bước · sai lệch lớn nhất ${fmt(diff)}`,
  );
  check(
    "độ cao lớn nhất khớp",
    Math.abs(result.peakAltitude - reference.expert.peak) < 1e-6,
    `${result.peakAltitude.toFixed(6)} m`,
  );
  check(
    "vận tốc chạm đất khớp (đọc TRƯỚC bước va chạm)",
    Math.abs(result.landingSpeed - reference.expert.landing) < 1e-6,
    `${result.landingSpeed.toFixed(6)} m/s`,
  );
  check(
    "thời gian bay khớp",
    Math.abs(result.duration - reference.expert.duration) < 1e-6,
    `${result.duration.toFixed(4)} s`,
  );
}

function decisionTargetLocal(d: { command: string; value: number }, altitude: number) {
  if (d.command === "LEN") return { altitude, climbRate: 1.8, level: true, cut: false, descentProfile: 0 };
  if (d.command === "GIU") return { altitude, climbRate: 0.0, level: true, cut: false, descentProfile: 0 };
  if (d.command === "ROI") return { altitude, climbRate: 0.0, level: true, cut: true, descentProfile: 0 };
  return { altitude, climbRate: 0.0, level: true, cut: false, descentProfile: d.value > 0 ? d.value : 7.0 };
}

// =====================================================================
// 3. Model — cùng câu, cùng lệnh
// =====================================================================

console.log("\nModel — cùng một câu đưa vào, phải viết ra cùng một lệnh");
console.log("=".repeat(64));

const model = new PilotModel(weights);
const pilot = new LearnedPilot(model, makeMission({ targetAltitude: 12.0 }));

{
  check(
    "số ký tự trong từ điển khớp",
    model.vocabSize === reference.model.chars.length,
    `${model.vocabSize} ký tự`,
  );

  let agree = 0;
  const mismatches: string[] = [];

  for (const prompt of reference.model.prompts) {
    const answer = pilot.write(prompt.text);
    if (answer === prompt.answer) {
      agree += 1;
    } else {
      mismatches.push(`"${prompt.text}" -> TS "${answer}" / Python "${prompt.answer}"`);
    }
  }

  check(
    `model viết ra đúng ${reference.model.prompts.length} câu giống hệt PyTorch`,
    agree === reference.model.prompts.length,
    agree === reference.model.prompts.length
      ? `${agree}/${reference.model.prompts.length}`
      : `${agree}/${reference.model.prompts.length} — lệch: ${mismatches.slice(0, 3).join(" · ")}`,
  );

  // So cả CON SỐ, không chỉ so ký tự được chọn. Hai bên dùng kiểu số khác
  // nhau (float32 ở Python, float64 ở JavaScript) nên không thể giống tới
  // từng bit — nhưng phải giống tới mức chọn ra cùng một ký tự.
  const probe = reference.model.logit_probe;
  const logits = model.forward(probe.ids);
  const last = Array.from(logits.slice((probe.ids.length - 1) * model.vocabSize));

  const diff = maxDiff(last, probe.logits);

  let argmaxTs = 0;
  let argmaxPy = 0;
  for (let i = 1; i < last.length; i += 1) {
    if (last[i] > last[argmaxTs]) argmaxTs = i;
    if (probe.logits[i] > probe.logits[argmaxPy]) argmaxPy = i;
  }

  const scale = Math.max(...probe.logits.map(Math.abs));

  check(
    "logits khớp trong sai số cho phép của float32",
    diff / scale < 1e-4,
    `lệch ${fmt(diff)} trên thang ${scale.toFixed(2)} = ${((diff / scale) * 100).toFixed(4)}%`,
  );
  check(
    "cả hai bên chọn cùng một ký tự",
    argmaxTs === argmaxPy,
    `TS chọn "${model.chars[argmaxTs]}" · Python chọn "${model.chars[argmaxPy]}"`,
  );
}

// =====================================================================
// 4. Đầu-cuối — cả chuyến bay do model lái
// =====================================================================

console.log("\nĐầu-cuối — cả chuyến bay do model lái, so từng bước");
console.log("=".repeat(64));

{
  for (const flight of reference.end_to_end.flights) {
    const v = makeVehicle();
    const mission = makeMission({ targetAltitude: flight.target });

    const tank = new Tank(v);
    const controller = new CascadedController(v);
    const learner = new LearnedPilot(model, mission);

    let state = makeState({
      theta: flight.initial_tilt,
      phi: -flight.initial_tilt * 0.7,
    });

    let target = { altitude: flight.target, climbRate: 1.8, level: true, cut: false, descentProfile: 0 };
    let decision = { command: "LEN", value: flight.target };
    let previousZ = state.z;

    const rows: number[][] = [];

    for (let index = 0; index < 25000; index += 1) {
      if (index % 50 === 0) {
        decision = learner.decide(state, tank);
        target = decisionTargetLocal(decision, flight.target);
      }

      const command = controller.call(state, tank, target);
      const record = step(state, v, tank, command);

      state = record.state;

      rows.push([
        state.t, state.z, state.vz, state.theta, state.phi,
        record.thrust, record.command.throttle, record.command.gimbalTheta, tank.pressure,
      ]);

      if (state.z <= 0 && previousZ > 0) break;
      previousZ = state.z;
    }

    const diff = maxDiff(rows.flat(), flight.rows.flat());

    check(
      `đề bài ${flight.target} m — model lái, từng bước khớp Python`,
      diff < 1e-6,
      `${rows.length} bước · sai lệch ${fmt(diff)} · lên ${rows[rows.length - 1][1].toFixed(2)} m`,
    );
  }
}

// =====================================================================

console.log("\n" + "=".repeat(64));
const total = passed + failures.length;
console.log(`${passed}/${total} mục đạt`);

if (failures.length) {
  console.log("\nCòn hỏng:");
  for (const name of failures) console.log(`  - ${name}`);
  process.exit(1);
}

console.log("Bản TypeScript khớp bản Python.");
