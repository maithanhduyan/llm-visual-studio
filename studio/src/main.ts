/**
 * main.ts — Ghép bốn phần trực quan lại, và nối với máy chủ.
 *
 * Trang này xem được cả ba cấp trong repo. Đổi model thì cả bốn panel
 * đều đổi theo.
 */

import type {
  AnalyzeResult,
  ArchData,
  ArchSelection,
  ModelOption,
  SpaceData,
  StudioMeta,
} from "../types";
import { createArch3D, edgeColor, kindColor } from "./arch3d";
import { createAttentionView, matrixFor } from "./attention";
import { renderExperts } from "./experts";
import { escapeHtml, expertColor, fmt } from "./format";
import { renderLoss, type LossOptions } from "./loss";
import { createSpace3D } from "./space3d";

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const modelSeg = el("model-seg");
const modelWhat = el("model-what");
const metaBox = el<HTMLDListElement>("meta");
const promptInput = el<HTMLInputElement>("prompt");
const goButton = el<HTMLButtonElement>("go");
const examplesBox = el("examples");
const statusBox = el("status");

const spaceBox = el("space3d");
const spaceLegend = el("space-legend");
const spaceNote = el("space-note");
const dModelLabel = el("dmodel");
const labelsBox = el<HTMLInputElement>("space-labels");
const spinBox = el<HTMLInputElement>("space-spin");
const resetButton = el<HTMLButtonElement>("space-reset");
const tooltip = el("tooltip");

const archBox = el("arch3d");
const archLegend = el("arch-legend");
const archDetail = el("arch-detail");
const archSummary = el("arch-summary");
const archLabels = el<HTMLInputElement>("arch-labels");
const archFlow = el<HTMLInputElement>("arch-flow");
const archSpin = el<HTMLInputElement>("arch-spin");
const archReset = el<HTMLButtonElement>("arch-reset");

const layerSeg = el("layer-seg");
const headSeg = el("head-seg");
const canvas = el<HTMLCanvasElement>("attn");
const readout = el("attn-readout");

const legendBox = el("expert-legend");
const gridBox = el("expert-grid");
const expertEmpty = el("expert-empty");
const panelExperts = el("panel-experts");
const topkLabel = el("topk");
const nexpertsLabel = el("nexperts");

const chartBox = el("loss-chart");
const logBox = el<HTMLInputElement>("loss-log");
const extraBox = el<HTMLInputElement>("loss-extra");

const EXAMPLES = [
  "Học mãi thì giỏi",
  "Chuyên gia giỏi toán trả lời câu hỏi",
  "Máy tính không hiểu như con người",
];

let meta: StudioMeta | null = null;
let current: AnalyzeResult | null = null;
let modelId = "";
let layerIndex = 0;
let headIndex = -1; // -1 = trung bình mọi đầu

const lossOptions: LossOptions = { log: false, showExtra: false };

const space = createSpace3D(spaceBox, tooltip);
const arch = createArch3D(archBox, tooltip, renderArchDetail);
const view = createAttentionView(canvas, tooltip, readout);

const describe = (error: unknown) => (error instanceof Error ? error.message : String(error));

function setStatus(text: string, isError = false) {
  statusBox.textContent = text;
  statusBox.classList.toggle("err", isError);
}

async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error ?? `Lỗi ${response.status}`);
  return data as T;
}

// ---------------------------------------------------------------------
// Chọn model
// ---------------------------------------------------------------------

function buildModelButtons(models: ModelOption[]) {
  modelSeg.innerHTML = models
    .map(
      (m) =>
        `<button type="button" data-id="${m.id}"${m.id === modelId ? ' class="on"' : ""}` +
        `${m.available ? "" : " disabled"} title="${escapeHtml(m.what)}">${escapeHtml(m.short)}</button>`,
    )
    .join("");

  modelSeg.onclick = (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button");
    if (!button || button.disabled) return;
    const id = button.dataset.id;
    if (id && id !== modelId) void selectModel(id);
  };
}

async function selectModel(id: string) {
  modelId = id;
  setStatus("Đang nạp model...");

  try {
    const [nextMeta, nextSpace, nextArch] = await Promise.all([
      getJSON<StudioMeta>(`/api/meta?model=${encodeURIComponent(id)}`),
      getJSON<SpaceData>(`/api/space?model=${encodeURIComponent(id)}`),
      getJSON<ArchData>(`/api/architecture?model=${encodeURIComponent(id)}`),
    ]);

    meta = nextMeta;
    buildModelButtons(nextMeta.models);
    renderHeader();
    renderSpace(nextSpace);
    renderArchitecture(nextArch);
    paintLoss();

    await run(promptInput.value);
  } catch (error) {
    setStatus(describe(error), true);
  }
}

// ---------------------------------------------------------------------
// Các panel
// ---------------------------------------------------------------------

function renderHeader() {
  if (!meta) return;

  modelWhat.textContent = meta.what;

  const rows: [string, string][] = [
    ["Thông số", fmt(meta.nParams)],
    ["Tầng", `${meta.nLayers}`],
    ["Đầu Q / K-V", meta.nKvHeads === meta.nHeads ? `${meta.nHeads} / ${meta.nHeads}` : `${meta.nHeads} / ${meta.nKvHeads}`],
    ["Từ điển", `${meta.vocabSize} ký tự`],
    ["Chuyên gia", meta.nExperts ? `${meta.nExperts} · chọn ${meta.topK}` : "không có"],
    ["Dòng suy nghĩ", `${meta.nStreams}`],
  ];

  if (meta.window) rows.push(["Cửa sổ trượt", `${meta.window} token`]);

  metaBox.innerHTML = rows
    .map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`)
    .join("");

  topkLabel.textContent = `${meta.topK || "?"}`;
  nexpertsLabel.textContent = `${meta.nExperts || "?"}`;

  // Cấp 1 chưa có chuyên gia -> giấu panel đó đi thay vì hiện bảng rỗng.
  if (meta.nExperts === 0) {
    legendBox.innerHTML = "";
    gridBox.innerHTML = "";
    expertEmpty.hidden = false;
    expertEmpty.textContent =
      "Model Cấp 1 chưa có chuyên gia. Mọi token đều đi qua cùng một MLP, " +
      "nên không có gì để chia. Lên Cấp 2 và Cấp 3 sẽ thấy phần này.";
  } else {
    expertEmpty.hidden = true;
  }
}

function renderSpace(data: SpaceData) {
  space.set(data);
  dModelLabel.textContent = `${data.dModel}`;

  const percent = (v: number) => `${(v * 100).toFixed(1)}%`;
  const total = data.explained.reduce((a, b) => a + b, 0);
  const detail = data.explained.map(percent).join(" + ");

  if (data.nExperts > 0) {
    spaceLegend.innerHTML = Array.from(
      { length: data.nExperts },
      (_, e) => `<span class="lg"><i style="background:${expertColor(e)}"></i>chuyên gia ${e}</span>`,
    ).join("");

    spaceNote.textContent =
      `Ba hướng đầu giữ được ${percent(total)} thông tin của bảng ${data.dModel} chiều ` +
      `(${detail}). Màu là chuyên gia mà ký tự đó hay hỏi nhất, khi model đọc ` +
      `chính bài học của nó. Cụm cùng màu = vùng ký tự mà cùng một chuyên gia phụ trách.`;
  } else {
    spaceLegend.innerHTML =
      `<span class="lg"><i style="background:${expertColor(-1)}"></i>ký tự</span>`;

    spaceNote.textContent =
      `Ba hướng đầu giữ được ${percent(total)} thông tin của bảng ${data.dModel} chiều ` +
      `(${detail}). Model này chưa có chuyên gia nên không tô màu được — ` +
      `nhưng vẫn thấy được ký tự nào được xếp gần nhau.`;
  }
}

function paintLoss() {
  if (meta) renderLoss(chartBox, meta.loss, lossOptions);
}

// ---------------------------------------------------------------------
// Sơ đồ kiến trúc
// ---------------------------------------------------------------------

function renderArchitecture(data: ArchData) {
  arch.set(data);

  const item = (color: string, what: string) =>
    `<span class="lg"><i style="background:${color}"></i>${escapeHtml(what)}</span>`;

  archLegend.innerHTML =
    data.kinds.map((k) => item(kindColor(k.kind), k.what)).join("") +
    data.edgeKinds.map((k) => item(edgeColor(k.kind), k.what)).join("");

  archSummary.textContent = `${data.nodes.length} bộ phận, ${data.edges.length} liên hệ · ${data.summary}`;
}

function renderArchDetail(selection: ArchSelection | null) {
  if (!selection) {
    archDetail.innerHTML =
      "Bấm vào một ô trong sơ đồ để xem nó ăn khớp với những bộ phận nào.";
    return;
  }

  const { node, incoming, outgoing } = selection;

  const tag = node.params
    ? `${fmt(node.params)} thông số`
    : "không có thông số";

  const pill = (names: string[]) =>
    names.length
      ? names.map((n) => `<span class="pill">${escapeHtml(n)}</span>`).join("")
      : '<span class="pill muted">không có</span>';

  archDetail.innerHTML =
    `<h3><i class="dot" style="background:${kindColor(node.kind)}"></i>${escapeHtml(node.label)}` +
    `<span class="tag">${tag}</span></h3>` +
    `<p>${escapeHtml(node.detail)}</p>` +
    `<div class="io"><span class="io-label">nhận từ</span><div>${pill(incoming)}</div></div>` +
    `<div class="io"><span class="io-label">gửi tới</span><div>${pill(outgoing)}</div></div>`;
}

function chips(
  container: HTMLElement,
  options: { label: string; value: number }[],
  active: number,
  onPick: (value: number) => void,
) {
  container.innerHTML = options
    .map(
      (option) =>
        `<button type="button" data-v="${option.value}"${option.value === active ? ' class="on"' : ""}>${escapeHtml(option.label)}</button>`,
    )
    .join("");

  container.onclick = (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button");
    if (!button) return;
    onPick(Number(button.dataset.v));
  };
}

function buildSegments() {
  if (!meta) return;

  chips(
    layerSeg,
    Array.from({ length: meta.nLayers }, (_, i) => ({ label: `tầng ${i}`, value: i })),
    layerIndex,
    (value) => {
      layerIndex = value;
      buildSegments();
      paintAttention();
    },
  );

  chips(
    headSeg,
    [
      { label: "trung bình", value: -1 },
      ...Array.from({ length: meta.nHeads }, (_, i) => ({ label: `đầu ${i}`, value: i })),
    ],
    headIndex,
    (value) => {
      headIndex = value;
      buildSegments();
      paintAttention();
    },
  );
}

function paintAttention() {
  if (current) view.set(current.chars, matrixFor(current.layers, layerIndex, headIndex));
}

// ---------------------------------------------------------------------
// Phân tích một câu
// ---------------------------------------------------------------------

async function run(text: string) {
  const question = text.trim();
  if (!question) {
    setStatus("Gõ một câu đã nhé.", true);
    return;
  }

  goButton.disabled = true;
  setStatus("Đang chạy model...");

  try {
    const data = await getJSON<AnalyzeResult>("/api/analyze", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: modelId, text: question }),
    });

    current = data;
    layerIndex = 0;
    headIndex = -1;

    buildSegments();
    paintAttention();
    if (meta && meta.nExperts > 0) renderExperts(gridBox, legendBox, current, meta, tooltip);

    const truncated = current.chars.length < [...question].length;
    setStatus(`${current.chars.length} token${truncated ? " · đã cắt bớt cho vừa hình" : ""}`);
  } catch (error) {
    setStatus(describe(error), true);
  } finally {
    goButton.disabled = false;
  }
}

// ---------------------------------------------------------------------
// Khởi động và nối sự kiện
// ---------------------------------------------------------------------

goButton.addEventListener("click", () => void run(promptInput.value));

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") void run(promptInput.value);
});

examplesBox.addEventListener("click", (event) => {
  const chip = (event.target as HTMLElement).closest<HTMLButtonElement>("button.chip");
  if (!chip) return;
  promptInput.value = chip.textContent ?? "";
  void run(promptInput.value);
});

labelsBox.addEventListener("change", () => space.setLabels(labelsBox.checked));
spinBox.addEventListener("change", () => space.setSpinning(spinBox.checked));
resetButton.addEventListener("click", () => space.resetView());

archLabels.addEventListener("change", () => arch.setLabels(archLabels.checked));
archFlow.addEventListener("change", () => arch.setFlow(archFlow.checked));
archSpin.addEventListener("change", () => arch.setSpinning(archSpin.checked));
archReset.addEventListener("click", () => arch.resetView());

// Xoay 3D tốn pin, nên chỉ vẽ khi panel còn nhìn thấy.
const watch = (box: HTMLElement, target: { setEnabled: (on: boolean) => void }) =>
  new IntersectionObserver(([entry]) => target.setEnabled(entry.isIntersecting), {
    threshold: 0.05,
  }).observe(box);

watch(archBox, arch);
watch(spaceBox, space);

logBox.addEventListener("change", () => {
  lossOptions.log = logBox.checked;
  paintLoss();
});

extraBox.addEventListener("change", () => {
  lossOptions.showExtra = extraBox.checked;
  paintLoss();
});

async function boot() {
  examplesBox.innerHTML = EXAMPLES.map(
    (text) => `<button type="button" class="chip">${escapeHtml(text)}</button>`,
  ).join("");

  try {
    const { models } = await getJSON<{ models: ModelOption[] }>("/api/models");

    const first = models.find((m) => m.id === "deepseek_lite" && m.available)
      ?? models.find((m) => m.available);

    if (!first) throw new Error("Chưa học model nào cả. Xem README ở thư mục gốc.");

    buildModelButtons(models);
    await selectModel(first.id);
  } catch (error) {
    setStatus(describe(error), true);
    renderLoss(chartBox, { step: [], loss: [], extra: [], extraName: "" }, lossOptions);
  }
}

void boot();

// Tay nắm gỡ lỗi: mở Console gõ __studio.space.stats() để xem three.js
// đang vẽ bao nhiêu điểm. Cũng dùng cho bài kiểm tra tự động.
(window as unknown as Record<string, unknown>).__studio = {
  space,
  arch,
  meta: () => meta,
  current: () => current,
  selectModel,
};
