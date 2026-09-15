/**
 * model.ts — Bản dịch của `deepseek_lite` (attention, MoE, hyper-connections,
 * RoPE, tokenizer) và của `policy.py` sang TypeScript.
 *
 * Đây là cùng một kiến trúc, viết lại để chạy trong trình duyệt. Không có gì
 * mới: GQA, RoPE, cửa sổ trượt, chuyên gia dùng chung, hyper-connections —
 * tất cả đều giống bản Python, chỉ nhỏ hơn (121 nghìn thông số).
 *
 * Đối chiếu với PyTorch bằng `check-port.ts`: cùng một câu, cả hai bên phải
 * viết ra cùng một lệnh. Đó là điều kiện để được gọi là "model thật sự lái".
 *
 * Mọi phép tính ở đây dùng float64 của JavaScript. Python dùng float32, nên
 * hai bên không giống nhau tới từng bit — chúng chỉ cần giống tới mức chọn ra
 * cùng một ký tự. Sai khác đó được đo và in ra, không giấu.
 */

// --- Cấu hình (policy.PILOT_CONFIG) -----------------------------------

export interface ModelConfig {
  vocabSize: number;
  dModel: number;
  nHeads: number;
  nKvHeads: number;
  nLayers: number;
  maxSeqLen: number;
  window: number;
  nExperts: number;
  topK: number;
  nShared: number;
  expertHidden: number;
  nStreams: number;
}

export const PILOT_CONFIG: Omit<ModelConfig, "vocabSize"> = {
  dModel: 64,
  nHeads: 4,
  nKvHeads: 2,
  nLayers: 2,
  maxSeqLen: 48,
  window: 0,
  nExperts: 4,
  topK: 2,
  nShared: 1,
  expertHidden: 48,
  nStreams: 1,
};

// --- Trọng số ---------------------------------------------------------

export interface Tensor {
  shape: number[];
  data: number[];
}

export interface Weights {
  config: ModelConfig;
  chars: string[];
  tensors: Record<string, Tensor>;
}

/** Một tầng Linear: y = x·Wᵀ + b. */
function linear(x: Float64Array, rows: number, inDim: number, w: Tensor, b?: Tensor): Float64Array {
  const outDim = w.shape[0];
  const out = new Float64Array(rows * outDim);
  const wd = w.data;
  const bd = b?.data;

  for (let r = 0; r < rows; r += 1) {
    const xOff = r * inDim;
    const oOff = r * outDim;

    for (let o = 0; o < outDim; o += 1) {
      let sum = bd ? bd[o] : 0;
      const wOff = o * inDim;

      for (let i = 0; i < inDim; i += 1) {
        sum += x[xOff + i] * wd[wOff + i];
      }

      out[oOff + o] = sum;
    }
  }

  return out;
}

/** RMSNorm: chia x cho "độ to trung bình" của chính nó. */
function rmsNorm(x: Float64Array, rows: number, dim: number, weight: Tensor): Float64Array {
  const out = new Float64Array(rows * dim);
  const wd = weight.data;

  for (let r = 0; r < rows; r += 1) {
    const off = r * dim;

    let sumSq = 0;
    for (let i = 0; i < dim; i += 1) sumSq += x[off + i] * x[off + i];

    const rms = Math.sqrt(sumSq / dim + 1e-6);

    for (let i = 0; i < dim; i += 1) {
      out[off + i] = (x[off + i] / rms) * wd[i];
    }
  }

  return out;
}

/** SiLU: x · sigmoid(x). */
const silu = (x: number) => x / (1 + Math.exp(-x));

/** SwiGLU: cửa mở thì nội dung mới đi qua. */
function swiglu(
  x: Float64Array,
  rows: number,
  inDim: number,
  prefix: string,
  t: Record<string, Tensor>,
): Float64Array {
  const hidden = t[`${prefix}.gate.weight`].shape[0];

  const gate = linear(x, rows, inDim, t[`${prefix}.gate.weight`], t[`${prefix}.gate.bias`]);
  const value = linear(x, rows, inDim, t[`${prefix}.value.weight`], t[`${prefix}.value.bias`]);

  const mixed = new Float64Array(rows * hidden);
  for (let i = 0; i < rows * hidden; i += 1) {
    mixed[i] = silu(gate[i]) * value[i];
  }

  return linear(mixed, rows, hidden, t[`${prefix}.out.weight`], t[`${prefix}.out.bias`]);
}

// --- RoPE -------------------------------------------------------------

/**
 * Bảng cos/sin tính sẵn (rope.py). Cặp chiều đầu tiên xoay nhanh, cặp cuối
 * xoay rất chậm.
 */
function ropeTables(headDim: number, maxSeqLen: number, base = 10000.0) {
  const half = headDim / 2;
  const cos = new Float64Array(maxSeqLen * half);
  const sin = new Float64Array(maxSeqLen * half);
  const freq = new Float64Array(half);

  for (let i = 0; i < half; i += 1) {
    freq[i] = 1 / base ** (i / half);
  }

  for (let p = 0; p < maxSeqLen; p += 1) {
    for (let i = 0; i < half; i += 1) {
      const angle = p * freq[i];
      cos[p * half + i] = Math.cos(angle);
      sin[p * half + i] = Math.sin(angle);
    }
  }

  return { cos, sin, half };
}

// --- Một tầng ---------------------------------------------------------

interface LayerWeights {
  norm1: Tensor;
  norm2: Tensor;
  qW: Tensor;
  qB: Tensor;
  kW: Tensor;
  kB: Tensor;
  vW: Tensor;
  vB: Tensor;
  oW: Tensor;
  oB: Tensor;
  routerW: Tensor;
  expertBias: Tensor;
  experts: string[];
  shared: string[];
  alpha: Float64Array;
  beta: Float64Array;
}

export class PilotModel {
  config: ModelConfig;
  chars: string[];
  stoi: Map<string, number>;
  private tensors: Record<string, Tensor>;
  private layers: LayerWeights[];
  private embed: Tensor;
  private finalNorm: Tensor;
  private rope: ReturnType<typeof ropeTables>;

  constructor(weights: Weights) {
    this.config = weights.config;
    this.chars = weights.chars;
    this.tensors = weights.tensors;
    this.stoi = new Map(this.chars.map((ch, i) => [ch, i]));

    const c = this.config;
    const t = this.tensors;

    this.embed = t["embedding.weight"];
    this.finalNorm = t["norm.weight"];
    this.rope = ropeTables(c.dModel / c.nHeads, c.maxSeqLen);

    this.layers = [];
    for (let i = 0; i < c.nLayers; i += 1) {
      const p = `blocks.${i}`;

      this.layers.push({
        norm1: t[`${p}.norm1.weight`],
        norm2: t[`${p}.norm2.weight`],
        qW: t[`${p}.attention.q_proj.weight`],
        qB: t[`${p}.attention.q_proj.bias`],
        kW: t[`${p}.attention.k_proj.weight`],
        kB: t[`${p}.attention.k_proj.bias`],
        vW: t[`${p}.attention.v_proj.weight`],
        vB: t[`${p}.attention.v_proj.bias`],
        oW: t[`${p}.attention.out_proj.weight`],
        oB: t[`${p}.attention.out_proj.bias`],
        routerW: t[`${p}.moe.router.weight`],
        expertBias: t[`${p}.moe.expert_bias`],
        experts: Array.from({ length: c.nExperts }, (_, e) => `${p}.moe.experts.${e}`),
        shared: Array.from({ length: c.nShared }, (_, s) => `${p}.moe.shared_experts.${s}`),
        alpha: Float64Array.from(t[`${p}.hyper.alpha`].data),
        beta: Float64Array.from(t[`${p}.hyper.beta`].data),
      });
    }
  }

  get vocabSize(): number {
    return this.config.vocabSize;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    for (const ch of text) {
      const id = this.stoi.get(ch);
      if (id !== undefined) ids.push(id);
    }
    return ids;
  }

  decode(ids: number[]): string {
    return ids.map((id) => this.chars[id] ?? "").join("");
  }

  /**
   * Chạy cả câu qua model, trả về điểm cho ký tự tiếp theo ở VỊ TRÍ CUỐI.
   *
   * Bản Python (`LearnedPilot.write`) gọi `self.model(tokens)` không có đệm
   * K/V, nên mỗi ký tự lại tính lại cả câu. Ở đây làm y hệt — câu chỉ dài
   * tối đa 47 ký tự nên không đáng để tối ưu, mà giữ giống bản gốc thì mới
   * đối chiếu được.
   */
  forward(tokens: number[]): Float64Array {
    const c = this.config;
    const T = tokens.length;
    const C = c.dModel;
    const nStreams = c.nStreams;

    // 1. Bảng tra: mỗi chữ có một vector riêng.
    const x = new Float64Array(T * C);
    for (let i = 0; i < T; i += 1) {
      const off = tokens[i] * C;
      for (let j = 0; j < C; j += 1) x[i * C + j] = this.embed.data[off + j];
    }

    // Nhân thành n dòng suy nghĩ giống hệt nhau lúc đầu. n_streams = 1 thì
    // đây chỉ là một dòng.
    let streams: Float64Array[] = [];
    for (let s = 0; s < nStreams; s += 1) streams.push(x.slice());

    for (const layer of this.layers) {
      streams = this.block(streams, layer, T);
    }

    // Gộp n dòng lại thành một câu trả lời.
    const merged = new Float64Array(T * C);
    for (const stream of streams) {
      for (let i = 0; i < T * C; i += 1) merged[i] += stream[i] / nStreams;
    }

    const normed = rmsNorm(merged, T, C, this.finalNorm);

    // LM Head dùng chung bảng tra với embedding (weight tying).
    return linear(normed, T, C, this.embed);
  }

  /** Một tầng: attention rồi tới MoE, mỗi việc đọc/ghi qua các dòng. */
  private block(streams: Float64Array[], layer: LayerWeights, T: number): Float64Array[] {
    const C = this.config.dModel;
    const n = streams.length;

    // Việc 1: các token nói chuyện với nhau.
    let h = new Float64Array(T * C);
    for (let s = 0; s < n; s += 1) {
      for (let i = 0; i < T * C; i += 1) h[i] += layer.alpha[s] * streams[s][i];
    }

    const attended = this.attention(rmsNorm(h, T, C, layer.norm1), layer, T);

    for (let s = 0; s < n; s += 1) {
      const target = streams[s];
      for (let i = 0; i < T * C; i += 1) target[i] += layer.beta[s] * attended[i];
    }

    // Việc 2: suy nghĩ riêng nhờ các chuyên gia.
    h = new Float64Array(T * C);
    for (let s = 0; s < n; s += 1) {
      for (let i = 0; i < T * C; i += 1) h[i] += layer.alpha[s] * streams[s][i];
    }

    const thought = this.moe(rmsNorm(h, T, C, layer.norm2), layer, T);

    for (let s = 0; s < n; s += 1) {
      const target = streams[s];
      for (let i = 0; i < T * C; i += 1) target[i] += layer.beta[s] * thought[i];
    }

    return streams;
  }

  /** GQA + RoPE + mặt nạ nhân quả. */
  private attention(x: Float64Array, layer: LayerWeights, T: number): Float64Array {
    const c = this.config;
    const C = c.dModel;
    const H = c.nHeads;
    const KV = c.nKvHeads;
    const D = C / H;
    const repeat = H / KV;
    const { cos, sin, half } = this.rope;

    const q = linear(x, T, C, layer.qW, layer.qB);
    const k = linear(x, T, C, layer.kW, layer.kB);
    const v = linear(x, T, C, layer.vW, layer.vB);

    // Xoay theo vị trí. `start_pos` luôn là 0 ở đây vì không dùng đệm K/V.
    const rotate = (buf: Float64Array, heads: number) => {
      for (let t = 0; t < T; t += 1) {
        for (let h = 0; h < heads; h += 1) {
          const off = (t * heads + h) * D;
          for (let i = 0; i < half; i += 1) {
            const c0 = cos[t * half + i];
            const s0 = sin[t * half + i];
            const a = buf[off + i];
            const b = buf[off + half + i];
            buf[off + i] = a * c0 - b * s0;
            buf[off + half + i] = a * s0 + b * c0;
          }
        }
      }
    };

    rotate(q, H);
    rotate(k, KV);

    // Nhân bản K, V lên cho đủ số đầu Q (GQA).
    const kFull = new Float64Array(T * H * D);
    const vFull = new Float64Array(T * H * D);

    for (let t = 0; t < T; t += 1) {
      for (let h = 0; h < H; h += 1) {
        const src = (t * KV + Math.floor(h / repeat)) * D;
        const dst = (t * H + h) * D;
        for (let i = 0; i < D; i += 1) {
          kFull[dst + i] = k[src + i];
          vFull[dst + i] = v[src + i];
        }
      }
    }

    const scale = 1 / Math.sqrt(D);
    const y = new Float64Array(T * H * D);

    // Mặt nạ nhân quả: token chỉ nhìn được chính nó và những token trước.
    // (`window = 0` nên không có cửa sổ trượt.)
    const allowed = (t: number, s: number) => {
      if (s > t) return false;
      if (c.window > 0 && s <= t - c.window) return false;
      return true;
    };

    for (let t = 0; t < T; t += 1) {
      for (let h = 0; h < H; h += 1) {
        const qOff = (t * H + h) * D;

        let maxScore = -Infinity;
        const scores: number[] = [];

        for (let s = 0; s <= t; s += 1) {
          if (!allowed(t, s)) {
            scores.push(-Infinity);
            continue;
          }

          const kOff = (s * H + h) * D;
          let dot = 0;
          for (let i = 0; i < D; i += 1) dot += q[qOff + i] * kFull[kOff + i];

          const score = dot * scale;
          scores.push(score);
          if (score > maxScore) maxScore = score;
        }

        let sum = 0;
        for (let s = 0; s < scores.length; s += 1) {
          scores[s] = scores[s] === -Infinity ? 0 : Math.exp(scores[s] - maxScore);
          sum += scores[s];
        }

        const yOff = (t * H + h) * D;
        for (let s = 0; s < scores.length; s += 1) {
          const weight = scores[s] / sum;
          if (weight === 0) continue;

          const vOff = (s * H + h) * D;
          for (let i = 0; i < D; i += 1) y[yOff + i] += weight * vFull[vOff + i];
        }
      }
    }

    return linear(y, T, C, layer.oW, layer.oB);
  }

  /**
   * MoE: một chuyên gia dùng chung + top_k chuyên gia riêng do router chọn.
   */
  private moe(x: Float64Array, layer: LayerWeights, T: number): Float64Array {
    const c = this.config;
    const C = c.dModel;
    const rows = T;
    const t = this.tensors;

    // 1. Chuyên gia dùng chung: ai cũng hỏi, không cần chọn.
    const out = new Float64Array(rows * C);
    for (const prefix of layer.shared) {
      const part = swiglu(x, rows, C, prefix, t);
      for (let i = 0; i < rows * C; i += 1) out[i] += part[i];
    }

    // 2. Router cho điểm từng chuyên gia riêng, cộng điểm thiên vị.
    const scores = linear(x, rows, C, layer.routerW);
    const bias = layer.expertBias.data;

    const probs = new Float64Array(rows * c.nExperts);
    for (let r = 0; r < rows; r += 1) {
      for (let e = 0; e < c.nExperts; e += 1) {
        probs[r * c.nExperts + e] = 1 / (1 + Math.exp(-(scores[r * c.nExperts + e] + bias[e])));
      }
    }

    // 3. Chọn top_k, rồi chuẩn hoá để tổng trọng số của mỗi token = 1.
    for (let r = 0; r < rows; r += 1) {
      const off = r * c.nExperts;
      const order = Array.from({ length: c.nExperts }, (_, e) => e).sort(
        (a, b) => probs[off + b] - probs[off + a],
      );

      const chosen = order.slice(0, c.topK);
      let sum = 0;
      for (const e of chosen) sum += probs[off + e];

      // 4. Hỏi từng chuyên gia được chọn, cộng kết quả theo trọng số.
      for (const e of chosen) {
        const part = swiglu(x.subarray(r * C, (r + 1) * C), 1, C, layer.experts[e], t);

        const weight = probs[off + e] / sum;
        for (let i = 0; i < C; i += 1) out[r * C + i] += weight * part[i];
      }
    }

    return out;
  }
}

// --- Bộ ra quyết định đã học (policy.py) ------------------------------

import { COMMANDS, type Decision, type Mission, type Pilot, type State, type Tank, makeMission, stateText } from "./physics";

export class LearnedPilot implements Pilot {
  model: PilotModel;
  mission: Mission;
  maxNew: number;
  phase = "?";

  /** Đếm xem model viết ra chữ không hiểu được bao nhiêu lần. */
  invalid = 0;
  total = 0;

  private newline: number | undefined;

  constructor(model: PilotModel, mission: Mission = makeMission(), maxNew = 16) {
    this.model = model;
    this.mission = mission;
    this.maxNew = maxNew;
    this.newline = this.model.stoi.get("\n");
  }

  /** Cho model viết tiếp, dừng ở xuống dòng. */
  write(prompt: string): string {
    const limit = this.model.config.maxSeqLen - 1;
    let tokens = this.model.encode(prompt);
    if (tokens.length > limit) tokens = tokens.slice(tokens.length - limit);

    if (!tokens.length) return "";

    const written: number[] = [];

    for (let i = 0; i < this.maxNew; i += 1) {
      const logits = this.model.forward(tokens);
      const vocab = this.model.vocabSize;

      let best = 0;
      let bestScore = -Infinity;
      for (let v = 0; v < vocab; v += 1) {
        const score = logits[(tokens.length - 1) * vocab + v];
        if (score > bestScore) {
          bestScore = score;
          best = v;
        }
      }

      if (best === this.newline) break;

      written.push(best);
      tokens = [...tokens, best];
    }

    return this.model.decode(written);
  }

  decide(state: State, tank: Tank): Decision {
    const target = this.mission?.targetAltitude ?? 12.0;
    const answer = this.write(`${stateText(state, tank, target)} -> `);

    this.total += 1;

    let decision = parse(answer);

    // Đếm xem model có viết ra chữ không hiểu được không. Phải xem CHỮ model
    // viết, không phải lệnh đã đọc ra: `parse` đã tự đổi mọi thứ lạ thành
    // `ROI` rồi, nên đọc lệnh thì lúc nào cũng thấy hợp lệ.
    const first = answer.trim().split(/\s+/)[0]?.toUpperCase() ?? "";
    if (!(COMMANDS as readonly string[]).includes(first)) this.invalid += 1;

    // Kiểm tra lệnh trước khi đưa xuống PID: `HAM 999` là lệnh vô nghĩa.
    if (decision.command === "HAM" && !(decision.value >= 1.0 && decision.value <= 15.0)) {
      this.invalid += 1;
      decision = { command: "HAM", value: 6.5 };
    }

    this.phase = decision.command;
    return decision;
  }
}

function parse(text: string): Decision {
  const parts = text.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return { command: "ROI", value: 0 };

  let name = parts[0].toUpperCase();
  if (!(COMMANDS as readonly string[]).includes(name)) name = "ROI";

  let value = 0;
  if (parts.length > 1) {
    const parsed = Number(parts[1]);
    value = Number.isFinite(parsed) ? parsed : 0;
  }

  return { command: name, value };
}
