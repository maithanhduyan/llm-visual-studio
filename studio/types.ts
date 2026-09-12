/**
 * types.ts — Kiểu dữ liệu dùng chung giữa server.ts (Bun) và trang web.
 *
 * Chỉ có kiểu, không có code. Nên file này biến mất khi Bun đóng gói.
 */

/** Dữ liệu của một tầng: bản đồ attention + đường đi của chuyên gia. */
export interface LayerData {
  /** attention[head][query][key] — hàng cộng lại = 1 */
  attention: number[][][];
  /** experts[token][0..top_k-1] — token này chọn chuyên gia nào */
  experts: number[][];
  /** weights[token][0..top_k-1] — chọn với mức độ bao nhiêu */
  weights: number[][];
}

/** Kết quả phân tích một câu. */
export interface AnalyzeResult {
  text: string;
  chars: string[];
  ids: number[];
  layers: LayerData[];
}

/** Loss theo từng bước huấn luyện. */
export interface LossData {
  step: number[];
  loss: number[];
  /** Đường thứ hai, mỗi cấp một nghĩa (xem extraName). */
  extra: number[];
  extraName: string;
}

/** Một model mà Studio mở được. */
export interface ModelOption {
  id: string;
  label: string;
  short: string;
  what: string;
  available: boolean;
}

/** Thông tin về model đang xem. */
export interface StudioMeta {
  id: string;
  label: string;
  what: string;
  models: ModelOption[];

  vocabSize: number;
  nParams: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  nExperts: number;
  topK: number;
  nStreams: number;
  window: number;
  maxSeqLen: number;
  cacheBytes: number;

  loss: LossData;
}

/** Một ký tự trong không gian embedding. */
export interface SpacePoint {
  ch: string;
  x: number;
  y: number;
  z: number;
  /** Chuyên gia mà ký tự này hay hỏi nhất. -1 nếu model không có chuyên gia. */
  expert: number;
  /** Tin cậy: chuyên gia đó chiếm bao nhiêu phần. */
  share: number;
  /** Ký tự này xuất hiện bao nhiêu lần trong bài học. */
  count: number;
}

/** Không gian embedding đã chiếu xuống 3 chiều. */
export interface SpaceData {
  model: string;
  nExperts: number;
  dModel: number;
  /** 3 thành phần đầu giữ được bao nhiêu phần thông tin. */
  explained: number[];
  points: SpacePoint[];
}

/** Một bộ phận trong sơ đồ kiến trúc. */
export interface ArchNode {
  id: string;
  label: string;
  /** Loại bộ phận — quyết định màu và hình dáng. */
  kind: string;
  x: number;
  y: number;
  z: number;
  /** Giải thích bộ phận này làm gì và ăn khớp với ai. */
  detail: string;
  params: number;
}

/** Một mối liên hệ giữa hai bộ phận. */
export interface ArchEdge {
  from: string;
  to: string;
  /** data = dòng chảy chính, residual = đường tắt, route/combine = chuyên gia. */
  kind: string;
}

export interface ArchLegendItem {
  kind: string;
  what: string;
}

/** Sơ đồ kiến trúc của một model. */
export interface ArchData {
  model: string;
  nodes: ArchNode[];
  edges: ArchEdge[];
  /** Dãy nút mà dòng chảy chính đi qua, theo đúng thứ tự. */
  spine: string[];
  kinds: ArchLegendItem[];
  edgeKinds: ArchLegendItem[];
  summary: string;
}

/** Khi bấm vào một bộ phận. */
export interface ArchSelection {
  node: ArchNode;
  incoming: ArchNode[];
  outgoing: ArchNode[];
}
