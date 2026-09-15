/**
 * scene.ts — Cảnh 3D: con tàu, luồng khí phụt, vệt bay.
 *
 * Hệ toạ độ: mô phỏng dùng z làm độ cao, còn three.js dùng y. Nên đổi trục
 * một lần cho gọn:
 *
 *     three.js (x, y, z)  =  mô phỏng (x, z, y)
 *                  ↑              ↑
 *              độ cao        bay ngang
 *
 * Đơn vị tính bằng mét. Con tàu cao 1,2 m, bay lên 15 m — nên cảnh rộng
 * khoảng 40 m là vừa.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

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

const BODY_LENGTH = 1.2;
const BODY_RADIUS = 0.05;

export function createScene(container: HTMLElement) {
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0b1017, 40, 120);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
  camera.position.set(14, 11, 20);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 7, 0);
  controls.minDistance = 4;
  controls.maxDistance = 120;
  controls.autoRotateSpeed = 0.6;

  // ---- Mặt đất ----
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(24, 64),
    new THREE.MeshBasicMaterial({ color: 0x121a24 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const grid = new THREE.GridHelper(48, 48, 0x263140, 0x1a222c);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.9;
  scene.add(grid);

  // ---- Vạch chia độ cao: mỗi 5 m một vòng ----
  for (const height of [5, 10, 15, 20]) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(2.4, 2.45, 64),
      new THREE.MeshBasicMaterial({
        color: height === 15 ? 0x3d8b53 : 0x2a3441,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: height === 15 ? 0.9 : 0.7,
      }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = height;
    scene.add(ring);
  }

  // ---- Con tàu ----
  const ship = new THREE.Group();

  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(BODY_RADIUS, BODY_RADIUS, BODY_LENGTH, 20),
    new THREE.MeshBasicMaterial({ color: 0xd7dee6 }),
  );
  ship.add(body);

  // Viền cho dễ nhìn thấy con tàu ở xa
  const shell = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.CylinderGeometry(BODY_RADIUS, BODY_RADIUS, BODY_LENGTH, 12)),
    new THREE.LineBasicMaterial({ color: 0x4a9eff }),
  );
  ship.add(shell);

  // Vòi phun ở đáy
  const nozzle = new THREE.Mesh(
    new THREE.CylinderGeometry(BODY_RADIUS * 0.45, BODY_RADIUS * 0.8, 0.12, 16, 1, true),
    new THREE.MeshBasicMaterial({ color: 0x8b98a8, side: THREE.DoubleSide }),
  );
  nozzle.position.y = -BODY_LENGTH / 2 - 0.05;
  ship.add(nozzle);

  // Mũi nhọn ở đỉnh
  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(BODY_RADIUS, 0.22, 20),
    new THREE.MeshBasicMaterial({ color: 0x4a9eff }),
  );
  nose.position.y = BODY_LENGTH / 2 + 0.11;
  ship.add(nose);

  // ---- Luồng khí phụt ----
  const plume = new THREE.Mesh(
    new THREE.ConeGeometry(0.16, 1, 16, 1, true),
    new THREE.MeshBasicMaterial({
      color: 0xff9f45,
      transparent: true,
      opacity: 0.75,
      side: THREE.DoubleSide,
    }),
  );
  plume.rotation.x = Math.PI; // chúc mũi xuống
  ship.add(plume);

  scene.add(ship);

  // ---- Vệt bay ----
  const trailGeometry = new THREE.BufferGeometry();
  const trailMaterial = new THREE.LineBasicMaterial({ color: 0x56d364, transparent: true, opacity: 0.85 });
  const trail = new THREE.Line(trailGeometry, trailMaterial);
  scene.add(trail);

  // Đệm chứa điểm của vệt bay, cấp phát một lần cho cả chuyến.
  let trailCapacity = 0;

  // ---- Vòng sáng dưới đất, cho biết con tàu đang ở trên đầu ----
  const shadow = new THREE.Mesh(
    new THREE.RingGeometry(0.12, 0.3, 24),
    new THREE.MeshBasicMaterial({ color: 0x4a9eff, transparent: true, opacity: 0.4, side: THREE.DoubleSide }),
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.02;
  scene.add(shadow);

  // ------------------------------------------------------------------

  function place(sample: Sample) {
    ship.position.set(sample.x, sample.z, sample.y);

    const theta = (sample.tilt * Math.PI) / 180;
    ship.rotation.set(-theta * 0.7, 0, -theta);
  }

  function setPlume(sample: Sample) {
    // Không có lực đẩy thì không có luồng khí.
    if (sample.throttle <= 0.02) {
      plume.visible = false;
      return;
    }

    plume.visible = true;

    // Càng mở van thì luồng càng dài và càng to.
    const length = 0.4 + sample.throttle * 2.4;
    plume.scale.set(0.6 + sample.throttle * 1.1, length, 0.6 + sample.throttle * 1.1);
    plume.position.y = -BODY_LENGTH / 2 - length / 2 - 0.1;

    // Nghiêng vòi phun thì luồng cũng nghiêng theo.
    plume.rotation.z = -sample.gimbal * 6;
  }

  function setTrail(samples: Sample[], upTo: number, visible: boolean) {
    trail.visible = visible;

    if (!visible) return;

    // Tự cấp phát đệm, rồi tự ghi từng điểm vào.
    //
    // KHÔNG dùng `geometry.setFromPoints()` ở đây. Tên của nó nghe như "thay
    // cả danh sách điểm", nhưng không phải: nó ghi vào đệm ĐÃ CÓ, và chỉ ghi
    // được tối đa bằng số điểm mà đệm đó đã chứa. Gọi nó lần đầu với 1 điểm
    // là đệm chỉ còn đúng 1 chỗ, và mọi lần gọi sau chỉ ghi được 1 điểm —
    // mà đường vẽ bằng 1 điểm thì không vẽ ra gì cả.
    //
    // Hậu quả: vệt bay KHÔNG BAO GIỜ hiện, ở bất kỳ thời điểm nào, dù mã
    // nguồn trông rất hợp lý. Nó chỉ cảnh báo một dòng ra console.
    const count = Math.min(upTo + 1, samples.length);

    if (trailCapacity !== samples.length) {
      trailCapacity = samples.length;
      trailGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(samples.length * 3), 3),
      );
    }

    const position = trailGeometry.getAttribute("position") as THREE.BufferAttribute;

    for (let i = 0; i < count; i += 1) {
      const s = samples[i];
      position.setXYZ(i, s.x, s.z, s.y);
    }

    position.needsUpdate = true;

    // Đệm chứa cả chuyến bay, nhưng chỉ vẽ tới chỗ đang phát.
    trailGeometry.setDrawRange(0, count);

    // Bán kính bao phải tính lại, vì ba chiều dùng nó để quyết định có cắt
    // vật thể đi không. Để nguyên số cũ thì có lúc cả vệt bay bị cắt oan.
    trailGeometry.computeBoundingSphere();
  }

  function update(sample: Sample) {
    place(sample);
    setPlume(sample);
    shadow.position.set(sample.x, 0.02, sample.y);
  }

  function fitCamera(maxAltitude: number) {
    const height = Math.max(maxAltitude, 6);
    controls.target.set(0, height * 0.45, 0);
    camera.position.set(height * 0.9, height * 0.75, height * 1.35);
    controls.update();
  }

  function resize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return;

    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  new ResizeObserver(resize).observe(container);
  resize();

  let running = true;

  function animate() {
    if (!running) return;
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return {
    update,
    setTrail,
    fitCamera,
    setAutoRotate(value: boolean) {
      controls.autoRotate = value;
    },
  };
}
