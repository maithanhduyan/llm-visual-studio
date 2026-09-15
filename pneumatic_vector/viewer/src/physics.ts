/**
 * physics.ts — Bản dịch của `vehicle.py`, `physics.py`, `controller.py`,
 * `expert.py` sang TypeScript.
 *
 * Bản dịch này KHÔNG phải bản mô phỏng mới. Nó là cùng một mô hình, viết lại
 * để chạy được trong trình duyệt. Từng công thức đều đối chiếu với bản Python
 * bằng `check-port.ts`: chạy cùng một chuyến bay ở cả hai bên rồi so từng
 * bước. Lệch một bước là hỏng.
 *
 * Vì sao phải chép lại thay vì gọi Python: để con tàu bay SỐNG trong trình
 * duyệt — chỉnh khối lượng, áp suất, góc vòi và thấy nó phản ứng ngay, thay
 * vì xem lại một chuyến bay đã ghi sẵn.
 *
 * Hệ toạ độ: z là độ cao (giống Python). Trình xem tự đổi sang y của three.js.
 */

// --- Hằng số vật lý (vehicle.py) -------------------------------------

export const G = 9.81; // m/s²
export const R_AIR = 287.0; // J/(kg·K)
export const GAMMA = 1.4; // tỉ số nhiệt dung của không khí
export const P_ATM = 101325.0; // Pa
export const RHO_AIR = 1.225; // kg/m³ ở mặt đất
export const T_GAS = 293.0; // K

// --- Con tàu ----------------------------------------------------------

export interface Vehicle {
  length: number;
  diameter: number;
  pressureBar: number;
  dryMass: number;
  throatDiameter: number;
  dischargeCoefficient: number;
  thrustCoefficient: number;
  gimbalMaxDeg: number;
  nozzleOffset: number;
  dragCoefficient: number;
  crossSection: number;
  aeroTorqueGain: number;

  /**
   * Cánh đuôi (tùy chọn). Hai cánh nhỏ hai bên, gắn phía đuôi.
   *
   * Chúng làm đúng một việc: kéo tâm khí động xuống DƯỚI trọng tâm, làm
   * mô-men khí động đổi dấu thành mô-men TỰ DỰNG LẠI — như đuôi mũi tên.
   *
   * `finArea` là diện tích MỖI cánh (m²), `finOffset` là khoảng cách sau
   * trọng tâm (m). Tĩnh ổn định khi `2·A·L > aeroTorqueGain·crossSection`.
   *
   * Đo được: cánh làm con tàu thẳng ra rất nhiều (rơi tự do 178° → 3,5°)
   * nhưng KHÔNG đổi quỹ đạo rơi (0,6%), và cánh đủ lớn để tĩnh ổn định thì
   * làm hỏng việc lái — xem `python -m pneumatic_vector fins`.
   */
  finArea: number;
  finOffset: number;
  finDragFactor: number;

  spinDamping: number;
  inertia: number;
  dt: number;
}

export function makeVehicle(overrides: Partial<Vehicle> = {}): Vehicle {
  return {
    length: 1.2,
    diameter: 0.1,
    pressureBar: 10.0,
    dryMass: 0.6,
    throatDiameter: 0.012,
    dischargeCoefficient: 0.85,
    thrustCoefficient: 1.25,
    gimbalMaxDeg: 12.0,
    nozzleOffset: 0.62,
    dragCoefficient: 0.55,
    crossSection: 0.0079,
    aeroTorqueGain: 1.4,
    finArea: 0.0,
    finOffset: 0.0,
    finDragFactor: 0.1,
    spinDamping: 0.006,
    inertia: 0.035,
    dt: 0.001,
    ...overrides,
  };
}

export const volume = (v: Vehicle) => Math.PI * (v.diameter / 2) ** 2 * v.length;
export const initialPressure = (v: Vehicle) => v.pressureBar * 1e5;
export const airMass = (v: Vehicle) => (initialPressure(v) * volume(v)) / (R_AIR * T_GAS);
export const throatArea = (v: Vehicle) => Math.PI * (v.throatDiameter / 2) ** 2;
export const gimbalMax = (v: Vehicle) => (v.gimbalMaxDeg * Math.PI) / 180;
export const totalMass = (v: Vehicle) => v.dryMass + airMass(v);

/** Cánh kéo tâm khí động xuống được bao nhiêu, m³ (vehicle.fin_moment). */
export const finMoment = (v: Vehicle) => 2.0 * v.finArea * v.finOffset;

/**
 * Số dương = MẤT ổn định, số âm = TỰ DỰNG LẠI, đơn vị m³ (vehicle.stability).
 *
 * Hiệu giữa "tâm khí động đẩy ngã thêm" và "cánh kéo lại".
 */
export const stability = (v: Vehicle) => v.aeroTorqueGain * v.crossSection - finMoment(v);

/** Diện tích cản hiệu dụng, m² — cánh làm tăng lên (vehicle.drag_area). */
export const dragArea = (v: Vehicle) => v.crossSection + v.finDragFactor * 2.0 * v.finArea;

/**
 * Diện tích mỗi cánh cần có để vừa đủ trung tính, m².
 *
 * `offset` để hỏi "nếu gắn cánh cách trọng tâm chừng này thì cần bao nhiêu" —
 * hữu ích vì lúc đang thiết kế thì chưa gắn cánh, nên `v.finOffset` còn bằng 0.
 */
export function finAreaForNeutral(v: Vehicle, offset?: number): number {
  const arm = offset ?? v.finOffset;
  if (arm <= 0) return Infinity;
  return (v.aeroTorqueGain * v.crossSection) / (2.0 * arm);
}

/** Khí phụt ra bao nhiêu kg mỗi giây (vehicle.mass_flow). */
export function massFlow(v: Vehicle, pressure: number, throttle: number): number {
  const throat = throatArea(v) * Math.max(0, Math.min(1, throttle));

  if (throat <= 0 || pressure <= P_ATM) return 0;

  if (pressure > 1.9 * P_ATM) {
    // Tắt nghẽn: dòng chảy đạt tốc độ âm thanh ở cổ vòi.
    const choke = Math.sqrt(GAMMA * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (GAMMA - 1)));
    return (v.dischargeCoefficient * throat * pressure * choke) / Math.sqrt(R_AIR * T_GAS);
  }

  // Áp suất đã yếu: dòng chảy dưới tốc độ âm thanh.
  //
  // Tỉ số ở đây là NGOÀI/TRONG, tức là một số nhỏ hơn 1. Bản Python từng viết
  // ngược lại, thành ra số hạng trong căn luôn âm và động cơ tắt ngóm ngay
  // khi áp suất xuống dưới 1,9 bar — đúng lúc con tàu cần hãm nhất.
  const ratio = P_ATM / pressure;
  const term =
    ((2 * GAMMA) / (GAMMA - 1)) *
    (ratio ** (2 / GAMMA) - ratio ** ((GAMMA + 1) / GAMMA));

  if (term <= 0) return 0;

  return (v.dischargeCoefficient * throat * pressure * Math.sqrt(term)) / Math.sqrt(R_AIR * T_GAS);
}

/**
 * Khí phụt ra nhanh bao nhiêu, m/s.
 *
 * Giữ tham số `_v` cho giống chữ ký của bản Python (`exhaust_velocity(vehicle,
 * pressure)`) — bản Python cũng không dùng tới con tàu ở đây. Đổi chữ ký thì
 * hai bên khó đối chiếu với nhau.
 */
export function exhaustVelocity(_v: Vehicle, pressure: number): number {
  if (pressure <= P_ATM) return 0;

  const ratio = P_ATM / pressure;
  const term = ((2 * GAMMA) / (GAMMA - 1)) * (1 - ratio ** ((GAMMA - 1) / GAMMA));

  return Math.sqrt(Math.max(term, 0) * R_AIR * T_GAS);
}

export function maximumThrust(v: Vehicle): number {
  const pressure = initialPressure(v);
  return massFlow(v, pressure, 1) * exhaustVelocity(v, pressure) * v.thrustCoefficient;
}

// --- Bình khí ---------------------------------------------------------

export class Tank {
  vehicle: Vehicle;
  pressure: number;

  constructor(vehicle: Vehicle) {
    this.vehicle = vehicle;
    this.pressure = initialPressure(vehicle);
  }

  get airMass(): number {
    return (this.pressure * volume(this.vehicle)) / (R_AIR * T_GAS);
  }

  get fillFraction(): number {
    return this.pressure / initialPressure(this.vehicle);
  }

  /** Phụt khí trong `dt` giây. Trả về lực đẩy sinh ra (N). */
  burn(throttle: number, dt: number): number {
    const mass = this.airMass;

    if (mass <= 1e-6) {
      this.pressure = P_ATM;
      return 0;
    }

    const flow = massFlow(this.vehicle, this.pressure, throttle);

    // Van ĐÓNG thì không phụt gì cả — và áp suất phải giữ nguyên.
    if (flow <= 0) return 0;

    const before = this.pressure;
    const used = Math.min(flow * dt, mass);
    this.pressure = Math.max(
      before - (used * R_AIR * T_GAS) / volume(this.vehicle),
      P_ATM,
    );

    const realFlow = used / dt;

    // Vận tốc phụt lấy ở áp suất TRUNG BÌNH trong bước, không phải áp suất
    // cuối bước. Với bước 1 ms thì hai số gần như bằng nhau, nhưng nếu xả hết
    // khí trong một bước dài thì áp suất cuối bước đúng bằng 1 bar — mà ở
    // 1 bar thì không còn lực đẩy nào cả.
    const velocity = exhaustVelocity(this.vehicle, 0.5 * (before + this.pressure));

    return realFlow * velocity * this.vehicle.thrustCoefficient;
  }
}

// --- Trạng thái và một bước vật lý ------------------------------------

export interface State {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  theta: number;
  phi: number;
  omegaTheta: number;
  omegaPhi: number;
  t: number;
}

export function makeState(overrides: Partial<State> = {}): State {
  return {
    x: 0, y: 0, z: 0,
    vx: 0, vy: 0, vz: 0,
    theta: 0, phi: 0,
    omegaTheta: 0, omegaPhi: 0,
    t: 0,
    ...overrides,
  };
}

export const tilt = (s: State) => Math.hypot(s.theta, s.phi);
export const tiltDeg = (s: State) => (tilt(s) * 180) / Math.PI;
export const speed = (s: State) => Math.sqrt(s.vx ** 2 + s.vy ** 2 + s.vz ** 2);
export const horizontalSpeed = (s: State) => Math.hypot(s.vx, s.vy);

export interface Command {
  throttle: number;
  gimbalTheta: number;
  gimbalPhi: number;
}

export function clipped(command: Command, v: Vehicle): Command {
  const limit = gimbalMax(v);
  return {
    throttle: Math.max(0, Math.min(1, command.throttle)),
    gimbalTheta: Math.max(-limit, Math.min(limit, command.gimbalTheta)),
    gimbalPhi: Math.max(-limit, Math.min(limit, command.gimbalPhi)),
  };
}

/** Hướng lực đẩy trong hệ toạ độ thế giới, khi thân tàu nghiêng (θ, φ). */
export function thrustDirection(theta: number, phi: number): [number, number, number] {
  return [
    Math.sin(theta),
    -Math.sin(phi) * Math.cos(theta),
    Math.cos(phi) * Math.cos(theta),
  ];
}

export interface StepRecord {
  t: number;
  state: State;
  command: Command;
  thrust: number;
  mass: number;
  pressure: number;
}

/** Tiến con tàu thêm `dt` giây (physics.step). */
export function step(state: State, v: Vehicle, tank: Tank, command: Command): StepRecord {
  const dt = v.dt;
  const cmd = clipped(command, v);

  const mass = v.dryMass + tank.airMass;

  // 1. Đốt khí: lực đẩy sinh ra, áp suất tụt xuống.
  const thrust = tank.burn(cmd.throttle, dt);

  // 2. Hướng phụt = hướng thân tàu, xoay thêm một góc bằng góc nghiêng vòi.
  const [ux, uy, uz] = thrustDirection(
    state.theta + cmd.gimbalTheta,
    state.phi + cmd.gimbalPhi,
  );

  // 3. Các lực tác dụng.
  let fx = thrust * ux;
  let fy = thrust * uy;
  let fz = thrust * uz - mass * G;

  // Lực cản không khí, ngược chiều chuyển động.
  const sp = speed(state);
  if (sp > 1e-6) {
    const drag = 0.5 * RHO_AIR * v.dragCoefficient * dragArea(v) * sp;
    fx -= drag * state.vx;
    fy -= drag * state.vy;
    fz -= drag * state.vz;
  }

  const ax = fx / mass;
  const ay = fy / mass;
  const az = fz / mass;

  // 4. Mô-men xoay.
  //    - Lái vector: chỉ phụ thuộc góc nghiêng vòi phun.
  //    - Khí động: `stability(v)` gộp hai thứ đối nghịch nhau:
  //         + tâm khí động trên trọng tâm -> ĐỒNG DẤU (tự ngã thêm)
  //         - cánh đuôi                   -> NGƯỢC DẤU (tự dựng lại)
  //    - Cản quay: làm chậm tốc độ xoay.
  const dynamicPressure = 0.5 * RHO_AIR * sp ** 2;
  const aero = stability(v) * dynamicPressure;

  let tauTheta = -v.nozzleOffset * thrust * cmd.gimbalTheta;
  tauTheta += aero * state.theta;
  tauTheta -= v.spinDamping * state.omegaTheta;

  let tauPhi = -v.nozzleOffset * thrust * cmd.gimbalPhi;
  tauPhi += aero * state.phi;
  tauPhi -= v.spinDamping * state.omegaPhi;

  const alphaTheta = tauTheta / v.inertia;
  const alphaPhi = tauPhi / v.inertia;

  // 5. Cộng dồn (Euler nửa ẩn — ổn định hơn Euler thường).
  const next: State = {
    t: state.t + dt,
    vx: state.vx + ax * dt,
    vy: state.vy + ay * dt,
    vz: state.vz + az * dt,
    x: 0, y: 0, z: 0,
    omegaTheta: state.omegaTheta + alphaTheta * dt,
    omegaPhi: state.omegaPhi + alphaPhi * dt,
    theta: 0,
    phi: 0,
  };

  next.x = state.x + next.vx * dt;
  next.y = state.y + next.vy * dt;
  next.z = state.z + next.vz * dt;

  next.theta = state.theta + next.omegaTheta * dt;
  next.phi = state.phi + next.omegaPhi * dt;

  // 6. Chạm đất thì dừng lại, không xuyên qua mặt đất.
  if (next.z <= 0) {
    next.z = 0;
    next.vz = Math.max(next.vz, 0);
  }

  return {
    t: next.t,
    state: next,
    command: cmd,
    thrust,
    mass: v.dryMass + tank.airMass,
    pressure: tank.pressure,
  };
}

// --- Bộ điều khiển PID (controller.py) --------------------------------

/** Lực đẩy sẽ có nếu mở van ở mức `throttle`, với áp suất hiện tại. */
export function thrustAt(v: Vehicle, pressure: number, throttle = 1.0): number {
  return massFlow(v, pressure, throttle) * exhaustVelocity(v, pressure) * v.thrustCoefficient;
}

export interface Target {
  altitude: number;
  climbRate: number;
  level: boolean;
  cut: boolean;
  descentProfile: number;
}

export function makeTarget(overrides: Partial<Target> = {}): Target {
  return { altitude: 12.0, climbRate: 0.0, level: true, cut: false, descentProfile: 0, ...overrides };
}

export interface Gains {
  attKp: number;
  attKd: number;
  altKp: number;
  altKd: number;
  landKd: number;
  latKp: number;
  latKd: number;
  maxTilt: number;
  minThrottleForSteering: number;
}

export const DEFAULT_GAINS: Gains = {
  attKp: 9.0,
  attKd: 5.4,
  altKp: 0.55,
  altKd: 1.35,
  landKd: 3.2,
  latKp: 0.02,
  latKd: 0.09,
  maxTilt: (9.0 * Math.PI) / 180,
  minThrottleForSteering: 4.0,
};

export class CascadedController {
  vehicle: Vehicle;
  gains: Gains;

  constructor(vehicle: Vehicle, gains: Gains = DEFAULT_GAINS) {
    this.vehicle = vehicle;
    this.gains = gains;
  }

  steeringTilt(state: State): [number, number] {
    const g = this.gains;

    const thetaTarget = -g.latKp * state.x - g.latKd * state.vx;
    const phiTarget = g.latKp * state.y + g.latKd * state.vy;

    const limit = g.maxTilt;
    return [
      Math.max(-limit, Math.min(limit, thetaTarget)),
      Math.max(-limit, Math.min(limit, phiTarget)),
    ];
  }

  attitudeGimbal(
    state: State,
    thrust: number,
    thetaTarget: number,
    phiTarget: number,
  ): [number, number] {
    const g = this.gains;
    const v = this.vehicle;

    // Chia cho lực đẩy là chỗ quan trọng nhất: mô-men xoay tỉ lệ với lực đẩy,
    // mà lực đẩy thì cạn dần. Không chia thì bộ điều khiển chỉ chạy được
    // trong khoảng nửa đầu chuyến bay.
    const authority = v.nozzleOffset * Math.max(thrust, g.minThrottleForSteering);

    const dynamicPressure = 0.5 * RHO_AIR * speed(state) ** 2;

    // Dùng `stability(v)`, KHÔNG dùng `aeroTorqueGain · crossSection`. Hai chỗ
    // này từng tính khác nhau: `step` biết về cánh đuôi còn chỗ này thì không,
    // nên khi gắn cánh vào thì bộ điều khiển vẫn tưởng con tàu mất ổn định
    // như cũ và đẩy vòi phun quá mạnh — theo đúng chiều ngược lại.
    const aero = stability(v) * dynamicPressure;

    const errorTheta = state.theta - thetaTarget;
    const errorPhi = state.phi - phiTarget;

    const deltaTheta =
      ((aero + v.inertia * g.attKp) * errorTheta +
        (v.inertia * g.attKd - v.spinDamping) * state.omegaTheta) /
      authority;

    const deltaPhi =
      ((aero + v.inertia * g.attKp) * errorPhi +
        (v.inertia * g.attKd - v.spinDamping) * state.omegaPhi) /
      authority;

    return [deltaTheta, deltaPhi];
  }

  throttleFor(state: State, tank: Tank, mass: number, target: Target): number {
    const g = this.gains;
    const v = this.vehicle;

    let accelWanted: number;

    if (target.descentProfile > 0) {
      // Hạ cánh theo giản đồ: ở độ cao h cho phép rơi nhanh tối đa
      // sqrt(2·a·h). Rơi nhanh hơn thì hãm; chậm hơn thì thả cho rơi.
      const allowed = -Math.sqrt(2 * target.descentProfile * Math.max(state.z, 0));
      const rateWanted = Math.max(allowed, -2.0);
      accelWanted = g.landKd * (rateWanted - state.vz);
    } else {
      let rateWanted = target.climbRate + g.altKp * (target.altitude - state.z);
      rateWanted = Math.max(-3.0, Math.min(5.0, rateWanted));
      accelWanted = g.altKd * (rateWanted - state.vz);
    }

    const tiltNow = tilt(state);
    const verticalShare = Math.max(Math.cos(tiltNow), 0.3);

    const thrustWanted = (mass * (accelWanted + G)) / verticalShare;

    if (thrustWanted <= 0) return 0;

    const ceiling = thrustAt(v, tank.pressure, 1.0);
    if (ceiling <= 0) return 0;

    return Math.max(0, Math.min(1, thrustWanted / ceiling));
  }

  /** Ba tầng lồng nhau: vị trí ngang -> góc nghiêng -> góc vòi phun. */
  call(state: State, tank: Tank, target: Target): Command {
    if (target.cut) {
      // Rơi tự do. Không có lực đẩy thì lái vô hiệu, nên trả về 0.
      return { throttle: 0, gimbalTheta: 0, gimbalPhi: 0 };
    }

    const mass = this.vehicle.dryMass + tank.airMass;
    const throttle = this.throttleFor(state, tank, mass, target);
    const thrust = thrustAt(this.vehicle, tank.pressure, throttle);

    let [thetaTarget, phiTarget] = this.steeringTilt(state);

    if (!target.level) {
      thetaTarget = 0;
      phiTarget = 0;
    }

    const [deltaTheta, deltaPhi] = this.attitudeGimbal(state, thrust, thetaTarget, phiTarget);

    return clipped(
      { throttle, gimbalTheta: deltaTheta, gimbalPhi: deltaPhi },
      this.vehicle,
    );
  }
}

// --- Lệnh cấp cao và chuyên gia (expert.py) ---------------------------

export const COMMANDS = ["LEN", "GIU", "ROI", "HAM"] as const;
export type CommandName = (typeof COMMANDS)[number];

export interface Decision {
  command: string;
  value: number;
}

export function decisionText(d: Decision): string {
  // `LEN`, `GIU`, `ROI` KHÔNG kèm con số: model nhỏ chép lại số rất kém, mà
  // hệ thống đã biết độ cao mục tiêu rồi (nó nằm trong đề bài).
  if (d.command === "LEN" || d.command === "GIU" || d.command === "ROI") return d.command;
  return `${d.command} ${d.value.toFixed(1)}`;
}

export function decisionTarget(d: Decision, targetAltitude = 12.0): Target {
  if (d.command === "LEN") return makeTarget({ altitude: targetAltitude, climbRate: 1.8 });
  if (d.command === "GIU") return makeTarget({ altitude: targetAltitude, climbRate: 0.0 });
  if (d.command === "ROI") return makeTarget({ cut: true });
  if (d.command === "HAM") {
    return makeTarget({ descentProfile: d.value > 0 ? d.value : 7.0 });
  }
  throw new Error(`Lệnh lạ: ${d.command}`);
}

export function parseDecision(text: string): Decision {
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

export interface Mission {
  targetAltitude: number;
  maxLandingSpeed: number;
  brakeAccel: number;
  margin: number;
}

export function makeMission(overrides: Partial<Mission> = {}): Mission {
  return {
    targetAltitude: 12.0,
    maxLandingSpeed: 2.0,
    brakeAccel: 11.0,
    margin: 1.15,
    ...overrides,
  };
}

/** Trạng thái viết thành chữ, để model đọc. Con số ĐẦU TIÊN là đề bài. */
export function stateText(state: State, tank: Tank, targetAltitude = 12.0): string {
  return (
    `${targetAltitude.toFixed(1)} ${state.z.toFixed(1)} ${state.vz.toFixed(1)} ` +
    `${tiltDeg(state).toFixed(1)} ${(tank.pressure / 1e5).toFixed(1)}`
  );
}

export class ExpertPilot {
  vehicle: Vehicle;
  mission: Mission;
  phase = "LEN";
  peakAltitude = 0;

  constructor(vehicle: Vehicle, mission: Mission = makeMission()) {
    this.vehicle = vehicle;
    this.mission = mission;
  }

  /** Còn hãm được mạnh nhất bao nhiêu m/s², với áp suất hiện tại. */
  availableDeceleration(tank: Tank): number {
    const mass = this.vehicle.dryMass + tank.airMass;
    const thrust = thrustAt(this.vehicle, tank.pressure, 1.0);
    return Math.max((thrust - mass * G) / mass, 0);
  }

  allowedSpeed(altitude: number, tank?: Tank): number {
    let decel = this.mission.brakeAccel;

    if (tank) {
      decel = Math.min(decel, this.availableDeceleration(tank) * 0.75);
    }

    return Math.sqrt(2 * Math.max(decel, 0.5) * Math.max(altitude, 0));
  }

  mustBrake(state: State, tank: Tank): boolean {
    return Math.abs(state.vz) >= this.allowedSpeed(state.z, tank) * 0.95;
  }

  brakeDeceleration(tank: Tank): number {
    const available = this.availableDeceleration(tank);
    const wanted = tank.fillFraction > 0.4 ? 8.0 : 6.5;
    return Math.max(Math.min(wanted, available * 0.7), 2.0);
  }

  decide(state: State, tank: Tank): Decision {
    const m = this.mission;
    this.peakAltitude = Math.max(this.peakAltitude, state.z);

    if (this.phase === "LEN") {
      if (state.z >= m.targetAltitude * 0.98) {
        this.phase = "GIU";
      } else if (state.vz < -0.5) {
        // Đang tụt mà chưa tới nơi -> hết khí rồi, đành chuyển sang hãm.
        this.phase = "HAM";
      } else {
        return { command: "LEN", value: 0 };
      }
    }

    if (this.phase === "GIU") {
      if (state.vz <= 0.15 && state.z >= m.targetAltitude * 0.9) {
        this.phase = "ROI";
      } else {
        return { command: "GIU", value: 0 };
      }
    }

    if (this.phase === "ROI") {
      if (this.mustBrake(state, tank) || state.z < 1.0) {
        this.phase = "HAM";
      } else {
        return { command: "ROI", value: 0 };
      }
    }

    if (this.phase === "HAM") {
      const decel = this.brakeDeceleration(tank);

      if (state.z <= 0.05 && Math.abs(state.vz) <= m.maxLandingSpeed) {
        this.phase = "XONG";
        return { command: "ROI", value: 0 };
      }

      return { command: "HAM", value: decel };
    }

    return { command: "ROI", value: 0 };
  }
}

// --- Giao diện chung cho bộ ra quyết định -----------------------------

export interface Pilot {
  decide(state: State, tank: Tank): Decision;
  phase: string;
  mission: Mission;
}

// --- Chạy một chuyến bay (simulate.py) --------------------------------

export const DECISION_HZ = 20;
export const LOG_HZ = 50;

export interface Sample {
  t: number;
  x: number;
  y: number;
  z: number;
  vz: number;
  tilt: number;
  throttle: number;
  gimbal: number;
  pressure: number;
  thrust: number;
  phase: string;
  decision: string;
}

export interface FlightResult {
  peakAltitude: number;
  landingSpeed: number;
  duration: number;
  airUsed: number;
  maxTiltDeg: number;
  phases: { t: number; name: string }[];
  samples: Sample[];
  crashed: boolean;
  timeout: boolean;
}

/** Chuyến bay có đạt yêu cầu không. */
export function success(result: FlightResult, mission: Mission): boolean {
  if (result.crashed || result.timeout) return false;

  return (
    result.peakAltitude >= 10.0 &&
    Math.abs(result.peakAltitude - mission.targetAltitude) <= 2.0 &&
    result.landingSpeed <= mission.maxLandingSpeed
  );
}

export interface FlyOptions {
  /** Nghiêng ban đầu, thay cho RNG của PyTorch. Bản Python sinh ra nó bằng
   *  `torch.Generator().manual_seed(seed)`; ở đây truyền thẳng con số vào để
   *  hai bên xuất phát từ ĐÚNG cùng một trạng thái. */
  initialTilt: number;
  maxTime?: number;
  keepSamples?: boolean;
}

/**
 * Cho một bộ ra quyết định lái con tàu (simulate.fly).
 *
 * `onStep` được gọi mỗi bước vật lý — dùng cho chế độ chạy sống, để vẽ ngay
 * chứ không phải chờ bay xong. Nó nhận thẳng state/record chứ không nhận mẫu
 * đã ghi, vì chế độ sống chạy 1000 bước mỗi giây và không cần dựng object
 * ở mỗi bước.
 */
export function fly(
  vehicle: Vehicle,
  policy: Pilot,
  mission: Mission = makeMission(),
  options: FlyOptions = { initialTilt: 0.04 },
  onStep?: (state: State, record: StepRecord, decision: Decision, phase: string) => void,
): FlightResult {
  const maxTime = options.maxTime ?? 25.0;
  const keepSamples = options.keepSamples ?? true;

  const tank = new Tank(vehicle);
  const controller = new CascadedController(vehicle);

  policy.mission = mission;

  const tilt0 = options.initialTilt;
  let state = makeState({ z: 0, theta: tilt0, phi: -tilt0 * 0.7 });

  const result: FlightResult = {
    peakAltitude: 0,
    landingSpeed: 0,
    duration: 0,
    airUsed: 0,
    maxTiltDeg: 0,
    phases: [],
    samples: [],
    crashed: false,
    timeout: false,
  };

  const airStart = tank.airMass;
  const decisionEvery = Math.max(1, Math.round(1 / (DECISION_HZ * vehicle.dt)));
  const logEvery = Math.max(1, Math.round(1 / (LOG_HZ * vehicle.dt)));

  let target = makeTarget({ altitude: mission.targetAltitude, climbRate: 1.8 });
  let decision: Decision = { command: "LEN", value: mission.targetAltitude };
  let lastPhase = policy.phase ?? "?";
  result.phases.push({ t: 0, name: String(lastPhase) });

  let landed = false;
  let previousZ = state.z;

  const steps = Math.round(maxTime / vehicle.dt);

  for (let index = 0; index < steps; index += 1) {
    // --- Vòng chậm: hỏi bộ ra quyết định ---
    if (index % decisionEvery === 0) {
      decision = policy.decide(state, tank);
      target = decisionTarget(decision, mission.targetAltitude);

      const phase = String((policy as { phase?: string }).phase ?? decision.command);
      if (phase !== lastPhase) {
        result.phases.push({ t: state.t, name: phase });
        lastPhase = phase;
      }
    }

    // --- Vòng nhanh: PID rồi tới vật lý ---
    const command = controller.call(state, tank, target);
    const record = step(state, vehicle, tank, command);

    const previousVz = state.vz;
    state = record.state;

    result.peakAltitude = Math.max(result.peakAltitude, state.z);
    result.maxTiltDeg = Math.max(result.maxTiltDeg, tiltDeg(state));

    if (keepSamples && index % logEvery === 0) {
      result.samples.push({
        t: round(state.t, 3),
        x: round(state.x, 3),
        y: round(state.y, 3),
        z: round(state.z, 3),
        vz: round(state.vz, 2),
        tilt: round(tiltDeg(state), 2),
        throttle: round(record.command.throttle, 3),
        gimbal: round(record.command.gimbalTheta, 4),
        pressure: round(record.pressure / 1e5, 2),
        thrust: round(record.thrust, 1),
        phase: lastPhase,
        decision: decisionText(decision),
      });
    }

    onStep?.(state, record, decision, lastPhase);

    // --- Chạm đất ---
    if (state.z <= 0 && previousZ > 0) {
      // Phải lấy vận tốc TRƯỚC bước va chạm: `step` ghim vz về 0 khi chạm
      // đất để con tàu không xuyên qua mặt đất, nên đọc sau đó thì chuyến
      // bay nào cũng "hạ cánh 0,00 m/s".
      result.landingSpeed = Math.abs(previousVz);
      landed = true;
      break;
    }

    previousZ = state.z;
  }

  result.duration = state.t;
  result.airUsed = airStart - tank.airMass;
  result.timeout = !landed;

  if (!landed) {
    result.landingSpeed = Math.abs(state.vz);
    result.crashed = true;
  } else if (result.landingSpeed > mission.maxLandingSpeed) {
    result.crashed = true;
  }

  return result;
}

const round = (value: number, digits: number) => {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
};

/**
 * Lệnh cố định dùng cho việc ĐỐI CHIẾU với Python (`export._scripted_command`).
 *
 * Chỉ phụ thuộc số bước và trạng thái, không có gì ngẫu nhiên, nên hai bên
 * chạy ra đúng cùng một chuỗi lệnh. Cố tình đi qua đủ các nhánh: van mở hết
 * (dòng tắt nghẽn), van hé (dòng dưới âm thanh), cắt ga hoàn toàn, và vòi
 * nghiêng đổi dấu liên tục.
 *
 * Chỗ giữ thăng bằng ở đây không phải để bắt chước bộ điều khiển thật. Nó chỉ
 * để con tàu khỏi lộn nhào — xem ghi chú dài ở bản Python: lái vòi phun bằng
 * sin cố định thì con lắc ngược sẽ lộn nhào, và ở đó sai số làm tròn bị khuếch
 * đại tới mức bài kiểm tra báo hỏng oan.
 */
export function commandFromScript(index: number, total: number, state: State): Command {
  const phase = index / total;

  let throttle: number;
  if (phase < 0.35) throttle = 1.0;
  else if (phase < 0.6) throttle = 0.25;
  else if (phase < 0.75) throttle = 0.0;
  else throttle = 0.6;

  const limit = (12.0 * Math.PI) / 180;
  let gimbal = -2.0 * state.theta - 0.5 * state.omegaTheta;
  gimbal = Math.max(-limit, Math.min(limit, gimbal));
  gimbal += ((1.5 * Math.PI) / 180) * Math.sin(index / 173.0);

  return { throttle, gimbalTheta: gimbal, gimbalPhi: -gimbal * 0.5 };
}

// --- Bộ ra quyết định chuyên gia, bọc theo giao diện Pilot -------------
export class ExpertPilotAdapter implements Pilot {
  inner: ExpertPilot;
  mission: Mission;

  constructor(vehicle: Vehicle, mission: Mission = makeMission()) {
    this.inner = new ExpertPilot(vehicle, mission);
    this.mission = mission;
  }

  get phase(): string {
    return this.inner.phase;
  }

  decide(state: State, tank: Tank): Decision {
    return this.inner.decide(state, tank);
  }
}
