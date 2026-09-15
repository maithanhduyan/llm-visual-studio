/**
 * main.ts — Hai chế độ, hai cách nhìn cùng một con tàu.
 *
 *   CHẠY SỐNG   vật lý và model chạy thật trong trình duyệt, ngay bây giờ
 *   PHÁT LẠI    đọc lại một chuyến bay mà Python đã mô phỏng và ghi ra file
 *
 * Chế độ sống là chế độ chính. Chế độ phát lại giữ lại vì nó cho xem được
 * những chuyến bay đã đo, kể cả những chuyến hỏng — thứ mà chạy sống không
 * tái hiện lại được.
 */

import { createScene, type Sample } from "./scene";
import {
  type LiveSettings,
  type LiveSnapshot,
  DEFAULT_SETTINGS,
  LiveFlight,
  loadModel,
} from "./live";


interface Flight {
  meta: Record<string, unknown>;
  summary: {
    peak_altitude: number;
    landing_speed: number;
    duration: number;
    max_tilt_deg: number;
    air_used_g: number;
    crashed: boolean;
  };
  phases: { t: number; name: string }[];
  samples: Sample[];
}

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const sceneBox = el("scene");
const hud = el("hud");
const decisions = el("decisions");
const decisionsTitle = el("decisions-title");
const summaryBox = el("summary");

const liveBar = el("live-bar");
const replayBar = el("replay-bar");
const modeLive = el<HTMLButtonElement>("mode-live");
const modeReplay = el<HTMLButtonElement>("mode-replay");
const modeNote = el("mode-note");

const policyPicker = el<HTMLSelectElement>("policy");
const liveRun = el<HTMLButtonElement>("live-run");
const liveReset = el<HTMLButtonElement>("live-reset");
const liveStatus = el("live-status");
const sliders = el("sliders");

const picker = el<HTMLSelectElement>("flight");
const playButton = el<HTMLButtonElement>("play");
const restartButton = el<HTMLButtonElement>("restart");
const scrub = el<HTMLInputElement>("scrub");
const clock = el("clock");
const slowBox = el<HTMLInputElement>("slow");
const trailBox = el<HTMLInputElement>("trail");
const spinBox = el<HTMLInputElement>("spin");

const view = createScene(sceneBox);

let mode: "live" | "replay" = "live";

// ---------------------------------------------------------------------
// Chế độ SỐNG
// ---------------------------------------------------------------------

// Model được nạp muộn, chỉ khi người dùng chọn "Model đã huấn luyện" — file
// 1,5 MB, không nên bắt ai cũng tải.
let live: LiveFlight | null = null;
let liveRunning = true;
let modelLoading = false;
let lastLiveStatus: LiveSnapshot | null = null;

interface SliderSpec {
  key: keyof LiveSettings;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  hint: string;
}

const SLIDERS: SliderSpec[] = [
  {
    key: "targetAltitude",
    label: "Đề bài",
    min: 8,
    max: 18,
    step: 0.5,
    unit: "m",
    hint: "phóng lên cao bao nhiêu",
  },
  {
    key: "pressureBar",
    label: "Áp suất nạp",
    min: 4,
    max: 16,
    step: 0.5,
    unit: "bar",
    hint: "càng nhiều khí thì hãm càng khoẻ, nhưng nặng hơn",
  },
  {
    key: "dryMass",
    label: "Khối lượng vỏ",
    min: 0.3,
    max: 1.2,
    step: 0.05,
    unit: "kg",
    hint: "nặng quá thì lực đẩy không đủ",
  },
  {
    key: "gimbalMaxDeg",
    label: "Góc vòi tối đa",
    min: 2,
    max: 12,
    step: 0.5,
    unit: "°",
    hint: "nghiêng được bao nhiêu — lái vector nằm ở đây",
  },
  {
    key: "aeroTorqueGain",
    label: "Độ mất ổn định",
    min: 0,
    max: 4,
    step: 0.1,
    unit: "",
    hint: "0 = con tàu trung tính, càng to càng khó giữ",
  },
  {
    key: "finAreaCm2",
    label: "Cánh đuôi",
    min: 0,
    max: 250,
    step: 5,
    unit: " cm²",
    hint:
      "diện tích MỖI cánh. Cánh kéo tâm khí động xuống dưới trọng tâm, " +
      "làm con tàu tự dựng lại — nhưng cũng chống lại việc lái",
  },
];

function buildSliders() {
  sliders.innerHTML = SLIDERS.map(
    (spec) => `
    <label class="slider" title="${spec.hint}">
      <span class="slabel">${spec.label}</span>
      <input type="range" id="sl-${spec.key}"
             min="${spec.min}" max="${spec.max}" step="${spec.step}"
             value="${DEFAULT_SETTINGS[spec.key]}" />
      <output id="out-${spec.key}">${DEFAULT_SETTINGS[spec.key]}${spec.unit}</output>
    </label>`,
  ).join("");

  for (const spec of SLIDERS) {
    const input = el<HTMLInputElement>(`sl-${spec.key}`);
    const output = el<HTMLOutputElement>(`out-${spec.key}`);

    input.addEventListener("input", () => {
      output.textContent = `${input.value}${spec.unit}`;

      if (!live) return;
      live.reset(readSettings());
      liveRunning = true;
      syncRunButton();
      drawDecisions([]);

      // Phải xoá dòng trạng thái cũ. Không xoá thì sau khi kéo thanh, trang
      // vẫn đang ghi "Chạm đất 1,93 m/s — ĐẠT" của chuyến VỪA RỒI, trong khi
      // con tàu mới còn chưa cất cánh. Người đọc tưởng đó là kết quả của
      // tham số mới.
      liveStatus.textContent = "Đang bay lại với tham số mới...";
    });
  }
}

function readSettings(): LiveSettings {
  const settings: LiveSettings = { ...DEFAULT_SETTINGS, policy: policyPicker.value as "expert" | "model" };

  for (const spec of SLIDERS) {
    settings[spec.key] = Number(el<HTMLInputElement>(`sl-${spec.key}`).value) as never;
  }

  return settings;
}

function syncRunButton() {
  // Bay xong rồi thì nút không còn là "chạy tiếp" nữa — bấm vào là bay lại
  // từ đầu. Ghi đúng việc nó làm, không ghi "Chạy tiếp" rồi restart.
  if (live?.finished) {
    liveRun.textContent = "Bay lại";
    return;
  }

  liveRun.textContent = liveRunning ? "Tạm dừng" : "Chạy tiếp";
}

/** Tên bộ ra quyết định đang dùng, để dòng trạng thái nói rõ đang là cái gì. */
function policyName(): string {
  return policyPicker.value === "model" ? "model trong trình duyệt" : "chuyên gia viết tay";
}

function resetLive() {
  live?.reset(readSettings());
  liveRunning = true;
  syncRunButton();
  view.fitCamera(readSettings().targetAltitude + 5);
  view.setTrail([], 0, false);
  drawDecisions([]);
  drawLiveSummary(null);
  liveStatus.textContent = "Đang bay...";
}

async function startLive() {
  buildSliders();

  const settings = readSettings();
  live = new LiveFlight(settings, null);

  view.fitCamera(settings.targetAltitude + 5);
  view.setTrail([], 0, false);
  drawDecisions([]);

  // Trạng thái ban đầu, trước khi bay.
  lastLiveStatus = live.snapshot();
  drawHudFromLive(lastLiveStatus);
  drawLiveSummary(lastLiveStatus);

  liveStatus.textContent = "Sẵn sàng: vật lý đang chạy trong trình duyệt.";
}

async function useModelPolicy() {
  if (live?.hasModel || modelLoading) return;

  modelLoading = true;
  liveStatus.textContent = "Đang nạp 121.220 thông số của model...";
  liveRunning = false;
  syncRunButton();

  try {
    const loaded = await loadModel();
    live?.setModel(loaded);

    liveStatus.textContent =
      `Đã nạp model trong trình duyệt — ${loaded.vocabSize} ký tự. ` +
      "Model thật sự viết ra từng lệnh, không đọc từ file nào.";
  } catch (error) {
    liveStatus.textContent = String(error);
    policyPicker.value = "expert";
  } finally {
    modelLoading = false;
    liveRunning = true;
    syncRunButton();
  }
}

// ---------------------------------------------------------------------
// Vòng chạy sống
// ---------------------------------------------------------------------

function frame(time: number) {
  requestAnimationFrame(frame);

  if (mode !== "live" || !live) {
    replayFrame(time);
    return;
  }

  const delta = lastLiveFrame ? (time - lastLiveFrame) / 1000 : 0;
  lastLiveFrame = time;

  if (liveRunning && !live.finished) {
    live.advance(delta);
  }

  const status = live.snapshot();
  lastLiveStatus = status;

  // Ba chiều: cảnh 3D vẽ mỗi khung hình, còn chữ thì thôi đừng dựng lại 60
  // lần mỗi giây — mắt người không đọc kịp, mà `innerHTML` thì tốn.
  view.update(sampleFromLive(status));
  view.setTrail(live.samples, live.samples.length - 1, status.stepsRun > 0);

  if (time - lastTextUpdate > 200) {
    lastTextUpdate = time;
    drawHudFromLive(status);
    drawLiveSummary(status);
    drawDecisions(live.phases);

    if (status.finished) {
      liveStatus.textContent = status.success
        ? `Đã hạ cánh sau ${status.stepsRun.toLocaleString("vi-VN")} bước. ` +
          `Chạm đất ${status.landingSpeed?.toFixed(2)} m/s — ĐẠT.`
        : `Chạm đất ${status.landingSpeed?.toFixed(2)} m/s — HỎNG. ` +
          "Kéo thanh chỉnh tham số rồi bay lại.";
    } else {
      liveStatus.textContent =
        `${policyName()} · ${status.stepsRun.toLocaleString("vi-VN")} bước · ` +
        `${status.decisionsMade} lần ra lệnh · ` +
        `mỗi lần quyết định tốn ${status.decisionMs.toFixed(2)} ms`;
    }
  }

  if (status.finished && liveRunning) {
    liveRunning = false;
    syncRunButton();
  }
}

let lastLiveFrame = 0;
let lastTextUpdate = 0;

function sampleFromLive(status: LiveSnapshot): Sample {
  return {
    t: status.t,
    x: 0,
    y: 0,
    z: status.z,
    vz: status.vz,
    tilt: status.tilt,
    throttle: status.throttle,
    gimbal: 0,
    pressure: status.pressure,
    thrust: status.thrust,
    phase: status.phase,
    decision: status.decision,
  };
}

function drawHudFromLive(status: LiveSnapshot) {
  hud.innerHTML = `
    <div class="hud-row"><span>thời gian</span><b>${status.t.toFixed(2)} s</b></div>
    <div class="hud-row"><span>độ cao</span><b>${status.z.toFixed(2)} m</b></div>
    <div class="hud-row"><span>vận tốc</span><b>${status.vz >= 0 ? "▲" : "▼"} ${Math.abs(status.vz).toFixed(2)} m/s</b></div>
    <div class="hud-row"><span>nghiêng</span><b>${status.tilt.toFixed(1)}°</b></div>
    <div class="hud-row"><span>khí còn</span><b>${status.pressure.toFixed(2)} bar · ${status.airLeft.toFixed(0)} g</b></div>
    <div class="hud-row"><span>van mở</span><b>${(status.throttle * 100).toFixed(0)}%</b></div>
    <div class="hud-row"><span>lực đẩy</span><b>${status.thrust.toFixed(1)} N</b></div>
    <div class="hud-row wide"><span>giai đoạn</span><b class="phase">${status.phase}</b></div>
    <div class="hud-row wide"><span>model ra lệnh</span><b class="cmd">${status.decision}</b></div>
  `;
}

function drawLiveSummary(status: LiveSnapshot | null) {
  // Đánh dấu bảng này là của chế độ nào. Bộ kiểm tra cần biết chắc đang đọc
  // số của chế độ sống hay của chuyến bay phát lại — hai bảng có cùng số mục.
  summaryBox.dataset.mode = "live";

  if (!status) {
    summaryBox.innerHTML = "";
    return;
  }

  const done = status.finished;

  summaryBox.innerHTML = `
    <div><dt>Đang chạy</dt><dd>${status.stepsRun.toLocaleString("vi-VN")} bước</dd></div>
    <div><dt>Model ra lệnh</dt><dd>${status.decisionsMade} lần</dd></div>
    <div><dt>Mỗi lần quyết định</dt><dd>${status.decisionMs.toFixed(2)} ms</dd></div>
    <div><dt>Lên cao nhất</dt><dd>${status.peak.toFixed(2)} m</dd></div>
    <div><dt>Nghiêng tối đa</dt><dd id="live-max-tilt">${status.maxTilt.toFixed(1)}°</dd></div>
    <div><dt>Ổn định</dt><dd class="${stabilityClass(status)}">${stabilityLabel(status)}</dd></div>
    <div><dt>Chạm đất</dt><dd class="${done ? (status.success ? "good" : "bad") : ""}">${
      status.landingSpeed === null ? "chưa" : `${status.landingSpeed.toFixed(2)} m/s`
    }</dd></div>
    <div><dt>Khí còn lại</dt><dd>${status.airLeft.toFixed(0)} g</dd></div>
  `;
}

/**
 * Nói thẳng con tàu đang ổn định hay không, và mốc để so.
 *
 * Đây là chỗ trả lời câu hỏi thiết kế cánh đuôi ngay trên màn hình: kéo thanh
 * cánh lên và đọc dòng này, thay vì phải tin vào lời giải thích.
 */
function stabilityLabel(status: LiveSnapshot): string {
  const neutral = status.finNeutralCm2;

  if (status.stability > 1e-5) {
    return `mất ổn định · cần ${neutral.toFixed(0)} cm² mỗi cánh để trung tính`;
  }

  if (status.stability < -1e-5) {
    return "tự dựng lại (tĩnh ổn định)";
  }

  return "trung tính";
}

function stabilityClass(status: LiveSnapshot): string {
  if (status.stability > 1e-5) return "bad";
  if (status.stability < -1e-5) return "warn";
  return "";
}

function drawDecisions(phases: { t: number; name: string }[]) {
  decisionsTitle.textContent =
    mode === "live" ? "Model đang ra lệnh gì" : "Những lần đổi lệnh trong chuyến bay";

  if (!phases.length) {
    decisions.innerHTML =
      '<p class="muted">Chưa có lệnh nào. Bấm Bay lại để bắt đầu.</p>';
    return;
  }

  decisions.innerHTML = phases
    .map(
      (phase) =>
        `<div class="decision"><span class="dt">${phase.t.toFixed(1)}s</span>` +
        `<span class="dcmd">${phase.name}</span></div>`,
    )
    .join("");
}

// ---------------------------------------------------------------------
// Chế độ PHÁT LẠI
// ---------------------------------------------------------------------

let flight: Flight | null = null;
let index = 0;
let playing = true;
let lastReplayFrame = 0;
let ticket = 0;

function setPlaying(value: boolean) {
  playing = value;
  playButton.textContent = value ? "Tạm dừng" : "Chạy tiếp";
}

function show(sample: Sample) {
  view.update(sample);
  view.setTrail(flight!.samples, index, trailBox.checked);

  clock.textContent = `${sample.t.toFixed(2)} s`;
  scrub.value = `${Math.round((index / Math.max(flight!.samples.length - 1, 1)) * 1000)}`;

  hud.innerHTML = `
    <div class="hud-row"><span>độ cao</span><b>${sample.z.toFixed(2)} m</b></div>
    <div class="hud-row"><span>vận tốc</span><b>${sample.vz >= 0 ? "▲" : "▼"} ${Math.abs(sample.vz).toFixed(2)} m/s</b></div>
    <div class="hud-row"><span>nghiêng</span><b>${sample.tilt.toFixed(1)}°</b></div>
    <div class="hud-row"><span>khí còn</span><b>${sample.pressure.toFixed(2)} bar</b></div>
    <div class="hud-row"><span>van mở</span><b>${(sample.throttle * 100).toFixed(0)}%</b></div>
    <div class="hud-row"><span>lực đẩy</span><b>${sample.thrust.toFixed(1)} N</b></div>
    <div class="hud-row wide"><span>giai đoạn</span><b class="phase">${sample.phase}</b></div>
    <div class="hud-row wide"><span>model ra lệnh</span><b class="cmd">${sample.decision}</b></div>
  `;
}

function drawReplayDecisions() {
  if (!flight) return;

  const rows: string[] = [];
  let previous = "";

  flight.samples.forEach((sample) => {
    if (sample.decision === previous) return;
    previous = sample.decision;

    rows.push(
      `<div class="decision"><span class="dt">${sample.t.toFixed(1)}s</span>` +
        `<span class="dz">${sample.z.toFixed(1)} m</span>` +
        `<span class="dcmd">${sample.decision}</span></div>`,
    );
  });

  decisions.innerHTML = rows.join("");
}

function drawReplaySummary() {
  if (!flight) return;

  summaryBox.dataset.mode = "replay";

  const s = flight.summary;

  summaryBox.innerHTML = `
    <div><dt>Lên cao nhất</dt><dd>${s.peak_altitude.toFixed(2)} m</dd></div>
    <div><dt>Chạm đất</dt><dd class="${s.landing_speed > 2 ? "bad" : "good"}">${s.landing_speed.toFixed(2)} m/s</dd></div>
    <div><dt>Nghiêng tối đa</dt><dd>${s.max_tilt_deg.toFixed(1)}°</dd></div>
    <div><dt>Khí đã dùng</dt><dd>${s.air_used_g.toFixed(0)} g</dd></div>
    <div><dt>Thời gian</dt><dd>${s.duration.toFixed(1)} s</dd></div>
    <div><dt>Kết quả</dt><dd class="${s.crashed ? "bad" : "good"}">${s.crashed ? "HỎNG" : "ĐẠT"}</dd></div>
  `;
}

async function loadFlight(name: string) {
  const mine = ++ticket;
  const response = await fetch(`/api/flight/${name}`);
  const data = (await response.json()) as Flight;

  if (mine !== ticket) return; // đã có người đổi sang chuyến khác rồi

  flight = data;
  index = 0;
  setPlaying(true);

  view.fitCamera(flight.summary.peak_altitude);
  drawReplaySummary();
  drawReplayDecisions();
  show(flight.samples[0]);
}

function replayFrame(time: number) {
  if (mode !== "replay" || !flight || !playing) {
    lastReplayFrame = time;
    return;
  }

  const delta = lastReplayFrame ? (time - lastReplayFrame) / 1000 : 0;
  lastReplayFrame = time;

  const perSecond = flight.samples.length / Math.max(flight.summary.duration, 0.5);
  const speed = slowBox.checked ? 0.25 : 1;
  index += delta * perSecond * speed;

  if (index >= flight.samples.length - 1) {
    index = flight.samples.length - 1;
    setPlaying(false);
  }

  show(flight.samples[Math.floor(index)]);
}

async function bootReplay() {
  const response = await fetch("/api/flights");
  const { flights } = (await response.json()) as {
    flights: { name: string; summary: Flight["summary"] }[];
  };

  if (!flights.length) {
    picker.innerHTML = '<option value="">chưa có chuyến bay nào</option>';
    return;
  }

  picker.innerHTML = flights
    .map(
      (f) =>
        `<option value="${f.name}">${f.name} · lên ${(f.summary.peak_altitude ?? 0).toFixed(1)} m · ` +
        `chạm đất ${(f.summary.landing_speed ?? 0).toFixed(2)} m/s</option>`,
    )
    .join("");

  await loadFlight(flights[0].name);
}

// ---------------------------------------------------------------------
// Đổi chế độ
// ---------------------------------------------------------------------

function setMode(next: "live" | "replay") {
  mode = next;

  modeLive.classList.toggle("on", next === "live");
  modeReplay.classList.toggle("on", next === "replay");

  liveBar.hidden = next !== "live";
  replayBar.hidden = next !== "replay";

  lastLiveFrame = 0;
  lastReplayFrame = 0;

  if (next === "live") {
    modeNote.textContent = "vật lý + model chạy trong trình duyệt";
    view.setTrail(live?.samples ?? [], (live?.samples.length ?? 0) - 1, true);
    resetLive();
  } else {
    modeNote.textContent = "đọc lại chuyến bay Python đã ghi";
    if (flight) {
      index = 0;
      setPlaying(true);
      view.fitCamera(flight.summary.peak_altitude);
      drawReplayDecisions();
      drawReplaySummary();
      show(flight.samples[0]);
    } else {
      drawDecisions([]);
    }
  }
}

// ---------------------------------------------------------------------
// Nối sự kiện
// ---------------------------------------------------------------------

modeLive.addEventListener("click", () => setMode("live"));
modeReplay.addEventListener("click", () => setMode("replay"));

policyPicker.addEventListener("change", () => {
  if (policyPicker.value === "model") {
    void useModelPolicy();
  } else {
    live?.setPolicy("expert");
    liveRunning = true;
    syncRunButton();
    drawDecisions([]);
  }
});

liveRun.addEventListener("click", () => {
  if (live?.finished) {
    resetLive();
    return;
  }

  liveRunning = !liveRunning;
  syncRunButton();
});

liveReset.addEventListener("click", resetLive);

picker.addEventListener("change", () => void loadFlight(picker.value));
playButton.addEventListener("click", () => setPlaying(!playing));
restartButton.addEventListener("click", () => {
  index = 0;
  setPlaying(true);
});

scrub.addEventListener("input", () => {
  if (!flight) return;
  index = Math.round((Number(scrub.value) / 1000) * (flight.samples.length - 1));
  setPlaying(false);
  show(flight.samples[index]);
});

trailBox.addEventListener("change", () => {
  if (flight) view.setTrail(flight.samples, index, trailBox.checked);
});

spinBox.addEventListener("change", () => view.setAutoRotate(spinBox.checked));

// ---------------------------------------------------------------------

requestAnimationFrame(frame);
void startLive();
void bootReplay();
