/**
 * live.ts — Con tàu bay SỐNG trong trình duyệt.
 *
 * Khác hẳn chế độ phát lại. Ở chế độ phát lại, Python đã mô phỏng xong rồi
 * trình duyệt chỉ đọc file ra mà vẽ. Ở đây thì **không có file nào cả**: vật
 * lý chạy thật trong trình duyệt, 1.000 bước mỗi giây, và model thật sự ra
 * lệnh 20 lần mỗi giây. Kéo thanh chỉnh khối lượng hay áp suất thì con tàu
 * phản ứng ngay, vì không có gì được tính sẵn.
 *
 * Vật lý và model ở đây là bản dịch từ Python, đã đối chiếu từng bước bằng
 * `check-port.ts` — không phải bản mô phỏng viết lại cho vui.
 *
 * ---
 *
 * ## Nhịp thời gian
 *
 * Vật lý chạy 1.000 Hz, màn hình vẽ 60 Hz. Không thể vẽ 1.000 lần mỗi giây,
 * mà cũng không được bỏ bước vật lý — bỏ bước là đổi bài toán.
 *
 * Nên dùng **bộ tích luỹ**: mỗi khung hình cộng thêm khoảng thời gian thật đã
 * trôi qua, rồi tiêu nó bằng đúng số bước 1 ms. Thừa bao nhiêu để lại cho
 * khung hình sau.
 *
 * Đây cũng chính là cách mọi trò chơi có vật lý làm việc.
 */

import {
  type Mission,
  type Pilot,
  type Sample,
  type State,
  type Vehicle,
  CascadedController,
  DECISION_HZ,
  ExpertPilotAdapter,
  LOG_HZ,
  Tank,
  decisionText,
  decisionTarget,
  finAreaForNeutral,
  makeMission,
  makeState,
  makeTarget,
  makeVehicle,
  stability,
  step,
  success,
  tiltDeg,
  type FlightResult,
} from "./physics";
import { LearnedPilot, PilotModel, type Weights } from "./model";

export type PolicyKind = "expert" | "model";

export interface LiveSettings {
  targetAltitude: number;
  dryMass: number;
  pressureBar: number;
  gimbalMaxDeg: number;
  aeroTorqueGain: number;
  /** Diện tích MỖI cánh đuôi, cm². 0 = không gắn cánh. */
  finAreaCm2: number;
  policy: PolicyKind;
}

export const DEFAULT_SETTINGS: LiveSettings = {
  targetAltitude: 12.0,
  dryMass: 0.6,
  pressureBar: 10.0,
  gimbalMaxDeg: 12.0,
  aeroTorqueGain: 1.4,
  finAreaCm2: 0,
  policy: "expert",
};

/** Cánh gắn cách trọng tâm bao xa, m. */
export const FIN_ARM = 0.5;

export interface LiveSnapshot {
  t: number;
  z: number;
  vz: number;
  tilt: number;
  pressure: number;
  throttle: number;
  thrust: number;
  phase: string;
  decision: string;
  airLeft: number;
  landingSpeed: number | null;
  finished: boolean;
  success: boolean;
  stepsRun: number;
  decisionsMade: number;
  /** Thời gian model tốn cho mỗi lần ra quyết định, mili-giây. */
  decisionMs: number;
  peak: number;
  maxTilt: number;
  /** Hệ số ổn định: dương = mất ổn định, âm = tự dựng lại (m³). */
  stability: number;
  /** Diện tích mỗi cánh để vừa đủ trung tính, cm². */
  finNeutralCm2: number;
}

/** Nạp trọng số model từ file. Chỉ gọi khi thật sự cần — file 1,5 MB. */
let cached: PilotModel | null = null;

export async function loadModel(url = "./public/model.json"): Promise<PilotModel> {
  if (cached) return cached;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Không nạp được ${url} (${response.status}). Chạy trước:\n` +
        "  python -m pneumatic_vector export",
    );
  }

  cached = new PilotModel((await response.json()) as Weights);
  return cached;
}

export class LiveFlight {
  settings: LiveSettings;
  vehicle: Vehicle;
  mission: Mission;

  private tank!: Tank;
  private controller!: CascadedController;
  private policy!: Pilot;

  private state!: State;
  private target = makeTarget();
  private decision = { command: "LEN", value: 0 };
  private lastPhase = "LEN";

  private accumulator = 0;
  private index = 0;
  private previousZ = 0;
  private airStart = 0;

  private model: PilotModel | null = null;

  phases: { t: number; name: string }[] = [];
  samples: Sample[] = [];

  stepsRun = 0;
  decisionsMade = 0;
  decisionMs = 0;
  landingSpeed: number | null = null;
  finished = false;
  timeout = false;

  constructor(settings: LiveSettings = DEFAULT_SETTINGS, model: PilotModel | null = null) {
    this.settings = { ...settings };
    this.model = model;
    this.vehicle = makeVehicle();
    this.mission = makeMission();
    this.reset(settings);
  }

  /** Dựng lại từ đầu với tham số mới. Đang bay mà kéo thanh thì bay lại. */
  reset(settings: LiveSettings = this.settings) {
    this.settings = { ...settings };

    this.vehicle = makeVehicle({
      dryMass: settings.dryMass,
      pressureBar: settings.pressureBar,
      gimbalMaxDeg: settings.gimbalMaxDeg,
      aeroTorqueGain: settings.aeroTorqueGain,
      // Thanh trượt tính bằng cm² MỖI cánh, và mô hình cũng tính bằng m² mỗi
      // cánh. Từng viết `* 2` ở đây vì tưởng `finArea` là tổng hai cánh —
      // thành ra trình xem gửi diện tích gấp đôi nhãn ghi.
      finArea: settings.finAreaCm2 / 1e4,
      finOffset: settings.finAreaCm2 > 0 ? FIN_ARM : 0,
    });

    this.mission = makeMission({ targetAltitude: settings.targetAltitude });

    this.tank = new Tank(this.vehicle);
    this.controller = new CascadedController(this.vehicle);
    this.airStart = this.tank.airMass;

    this.policy =
      settings.policy === "model" && this.model
        ? new LearnedPilot(this.model, this.mission)
        : new ExpertPilotAdapter(this.vehicle, this.mission);

    // Nghiêng ban đầu cố định: ở chế độ sống, mỗi lần chạy lại phải giống
    // nhau, nếu không thì không so sánh được hai bộ tham số với nhau.
    const tilt0 = 0.035;
    this.state = makeState({ z: 0, theta: tilt0, phi: -tilt0 * 0.7 });

    this.target = makeTarget({ altitude: settings.targetAltitude, climbRate: 1.8 });
    this.decision = { command: "LEN", value: settings.targetAltitude };
    this.lastPhase = this.policy.phase ?? "LEN";

    this.phases = [{ t: 0, name: String(this.lastPhase) }];
    this.samples = [];

    this.accumulator = 0;
    this.index = 0;
    this.previousZ = 0;
    this.stepsRun = 0;
    this.decisionsMade = 0;
    this.decisionMs = 0;
    this.landingSpeed = null;
    this.finished = false;
    this.timeout = false;

    // Hai con số này PHẢI trả về 0 cùng với mọi thứ khác.
    //
    // Chỗ này từng thiếu, và hậu quả rất khó thấy: sau khi kéo thanh chỉnh
    // tham số, bảng số liệu vẫn ghi "Lên cao nhất 18,43 m" trong khi chuyến
    // mới chỉ lên 13,41 m — vì nó đang giữ kỷ lục của chuyến TRƯỚC. Bài
    // kiểm tra cũng bị lừa theo, và so nhầm hai chuyến khác nhau.
    this.peak = 0;
    this.maxTilt = 0;
  }

  setModel(model: PilotModel) {
    this.model = model;
    if (this.settings.policy === "model") this.reset(this.settings);
  }

  /** Đã nạp trọng số chưa — để không nạp lại file 1,5 MB lần thứ hai. */
  get hasModel(): boolean {
    return this.model !== null;
  }

  setPolicy(kind: PolicyKind) {
    this.reset({ ...this.settings, policy: kind });
  }

  /**
   * Tiến mô phỏng thêm `elapsed` giây thời gian thật.
   *
   * Trả về số bước vật lý đã chạy. Nếu máy chậm, số bước vẫn đúng bằng thời
   * gian đã trôi — chậm thì bị đuối chứ không bỏ bước, vì bỏ bước là đổi
   * bài toán.
   */
  advance(elapsed: number): number {
    if (this.finished) return 0;

    // Kẹp lại: tab bị ẩn rồi quay lại có thể cho `elapsed` cỡ vài giây, và
    // chạy bù chừng đó sẽ treo trình duyệt.
    this.accumulator += Math.min(elapsed, 0.1);

    const dt = this.vehicle.dt;
    const decisionEvery = Math.max(1, Math.round(1 / (DECISION_HZ * dt)));
    const logEvery = Math.max(1, Math.round(1 / (LOG_HZ * dt)));

    let ran = 0;

    while (this.accumulator >= dt) {
      this.accumulator -= dt;

      if (this.index % decisionEvery === 0) {
        const started = performance.now();
        this.decision = this.policy.decide(this.state, this.tank);
        this.decisionMs = performance.now() - started;
        this.decisionsMade += 1;

        this.target = decisionTarget(this.decision, this.settings.targetAltitude);

        const phase = String((this.policy as { phase?: string }).phase ?? this.decision.command);
        if (phase !== this.lastPhase) {
          this.phases.push({ t: this.state.t, name: phase });
          this.lastPhase = phase;
        }
      }

      const command = this.controller.call(this.state, this.tank, this.target);
      const record = step(this.state, this.vehicle, this.tank, command);

      const previousVz = this.state.vz;
      this.state = record.state;
      this.stepsRun += 1;

      // Ghi lại lệnh THẬT SỰ đã đưa xuống, để HUD và luồng khí phụt hiển thị
      // đúng cái đang xảy ra chứ không phải tính lại rồi đoán.
      this.lastThrottle = record.command.throttle;
      this.lastThrust = record.thrust;

      this.peak = Math.max(this.peak, this.state.z);
      this.maxTilt = Math.max(this.maxTilt, tiltDeg(this.state));

      if (this.index % logEvery === 0) {
        this.samples.push({
          t: round(this.state.t, 3),
          x: round(this.state.x, 3),
          y: round(this.state.y, 3),
          z: round(this.state.z, 3),
          vz: round(this.state.vz, 2),
          tilt: round(tiltDeg(this.state), 2),
          throttle: round(record.command.throttle, 3),
          gimbal: round(record.command.gimbalTheta, 4),
          pressure: round(record.pressure / 1e5, 2),
          thrust: round(record.thrust, 1),
          phase: this.lastPhase,
          decision: decisionText(this.decision),
        });
      }

      if (this.state.z <= 0 && this.previousZ > 0) {
        this.landingSpeed = Math.abs(previousVz);
        this.finished = true;
        break;
      }

      if (this.state.t >= 25.0) {
        this.timeout = true;
        this.finished = true;
        break;
      }

      this.previousZ = this.state.z;
      this.index += 1;
      ran += 1;
    }

    return ran;
  }

  get result(): FlightResult {
    return {
      peakAltitude: this.peak,
      landingSpeed: this.landingSpeed ?? 0,
      duration: this.state.t,
      airUsed: this.airStart - this.tank.airMass,
      maxTiltDeg: this.maxTilt,
      phases: this.phases,
      samples: this.samples,
      crashed: this.landingSpeed !== null && this.landingSpeed > this.mission.maxLandingSpeed,
      timeout: this.timeout,
    };
  }

  peak = 0;
  maxTilt = 0;

  snapshot(): LiveSnapshot {
    return {
      t: this.state.t,
      z: this.state.z,
      vz: this.state.vz,
      tilt: tiltDeg(this.state),
      pressure: this.tank.pressure / 1e5,
      throttle: this.lastThrottle,
      thrust: this.lastThrust,
      phase: this.lastPhase,
      decision: decisionText(this.decision),
      airLeft: this.tank.airMass * 1000,
      landingSpeed: this.landingSpeed,
      finished: this.finished,
      success: this.finished && success(this.result, this.mission),
      stepsRun: this.stepsRun,
      decisionsMade: this.decisionsMade,
      decisionMs: this.decisionMs,
      peak: this.peak,
      maxTilt: this.maxTilt,
      stability: stability(this.vehicle),
      finNeutralCm2: (finAreaForNeutral(this.vehicle, FIN_ARM) * 1e4),
    };
  }

  private lastThrottle = 0;
  private lastThrust = 0;
}

const round = (value: number, digits: number) => {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
};
