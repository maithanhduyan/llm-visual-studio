/**
 * space3d.ts — Không gian embedding, nhìn bằng 3 chiều.
 *
 * Bảng embedding của model rộng 96–192 chiều. Con người không nhìn được
 * 192 chiều, nên `analyze.py` đã chiếu nó xuống 3 chiều bằng PCA — giữ lại
 * 3 hướng mà bảng embedding "trải ra" nhiều nhất.
 *
 * Mỗi điểm là một ký tự. Kéo chuột để xoay, lăn chuột để phóng to.
 * Tô màu theo chuyên gia mà ký tự đó hay hỏi nhất.
 *
 * Vì sao phải 3D chứ không phải 2D: khoảng cách giữa các ký tự trong đầu
 * model là thứ ba chiều. Chiếu xuống 2D thì hai ký tự khác nhau hoàn toàn
 * có thể chồng lên nhau, và ta mất luôn thông tin.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { SpaceData, SpacePoint } from "../types";
import { expertColor } from "./format";

export interface Space3DView {
  set(data: SpaceData): void;
  setLabels(visible: boolean): void;
  setSpinning(spinning: boolean): void;
  resetView(): void;
  setEnabled(enabled: boolean): void;
  /** Số liệu thật lấy từ three.js — dùng để gỡ lỗi và để kiểm tra tự động. */
  stats(): {
    points: number;
    labels: number;
    drawCalls: number;
    camera: [number, number, number];
    spinning: boolean;
  };
}

export function createSpace3D(container: HTMLElement, tooltip: HTMLElement): Space3DView {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(2.4, 1.7, 2.8);

  let renderer: THREE.WebGLRenderer;
  try {
    // alpha: true để nền trong suốt, cho màu nền của CSS hiện ra. Nếu tự
    // đặt màu nền trong three.js thì phải qua quy đổi không gian màu, và
    // nó sẽ lệch so với màu panel.
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch {
    container.innerHTML =
      '<p class="empty">Máy này không mở được WebGL nên không xem được phần 3D.</p>';
    return {
      set: () => {},
      setLabels: () => {},
      setSpinning: () => {},
      resetView: () => {},
      setEnabled: () => {},
      stats: () => ({ points: 0, labels: 0, drawCalls: 0, camera: [0, 0, 0], spinning: false }),
    };
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.9;
  controls.zoomSpeed = 0.9;
  controls.minDistance = 1.2;
  controls.maxDistance = 14;
  controls.autoRotateSpeed = 1.1;
  controls.autoRotate = true;

  // Khung hộp mờ, để mắt còn thấy chiều sâu khi xoay.
  const frame = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(2.4, 2.4, 2.4)),
    new THREE.LineBasicMaterial({ color: 0x263140, transparent: true, opacity: 0.9 }),
  );
  scene.add(frame);

  // Ba trục mờ của không gian.
  const axes = new THREE.AxesHelper(1.35);
  (axes.material as THREE.Material).opacity = 0.28;
  (axes.material as THREE.Material).transparent = true;
  scene.add(axes);

  let data: SpaceData | null = null;
  let points: THREE.Points | null = null;
  let labelGroup = new THREE.Group();
  scene.add(labelGroup);

  let labelsWanted = false;
  let enabled = true;

  // ---- Dựng các điểm ------------------------------------------------
  function clearPoints() {
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      (points.material as THREE.Material).dispose();
      points = null;
    }
    for (const child of labelGroup.children) {
      const sprite = child as THREE.Sprite;
      sprite.material.map?.dispose();
      sprite.material.dispose();
    }
    labelGroup.clear();
  }

  function makeLabel(text: string, color: string): THREE.Sprite {
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = color;
    ctx.font = "bold 46px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text === " " ? "␣" : text, size / 2, size / 2 + 2);

    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: new THREE.CanvasTexture(canvas),
        transparent: true,
        depthWrite: false,
      }),
    );
    sprite.scale.set(0.11, 0.11, 0.11);
    return sprite;
  }

  function buildLabels() {
    for (const child of labelGroup.children) {
      const sprite = child as THREE.Sprite;
      sprite.material.map?.dispose();
      sprite.material.dispose();
    }
    labelGroup.clear();

    if (!data || !labelsWanted) return;

    for (const point of data.points) {
      const sprite = makeLabel(point.ch, expertColor(point.expert));
      sprite.position.set(point.x, point.y, point.z);
      labelGroup.add(sprite);
    }
  }

  function set(next: SpaceData) {
    data = next;
    clearPoints();

    const positions = new Float32Array(next.points.length * 3);
    const colors = new Float32Array(next.points.length * 3);
    const colour = new THREE.Color();

    next.points.forEach((point, i) => {
      positions[i * 3] = point.x;
      positions[i * 3 + 1] = point.y;
      positions[i * 3 + 2] = point.z;

      colour.set(expertColor(point.expert));
      colors[i * 3] = colour.r;
      colors[i * 3 + 1] = colour.g;
      colors[i * 3 + 2] = colour.b;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ size: 0.075, vertexColors: true, sizeAttenuation: true }),
    );
    scene.add(points);

    buildLabels();
  }

  // ---- Rê chuột để xem ký tự ----------------------------------------
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: 0.07 };
  const pointer = new THREE.Vector2();

  function describe(point: SpacePoint): string {
    const head = `<b>${point.ch === " " ? "␣ (dấu cách)" : point.ch}</b>`;
    const times = `xuất hiện ${point.count} lần`;

    if (point.expert < 0) return `${head} · ${times}`;

    const swatch = `<span style="color:${expertColor(point.expert)}">■</span>`;
    return `${head} · ${times}<br>hay hỏi ${swatch} chuyên gia ${point.expert} (${Math.round(point.share * 100)}%)`;
  }

  renderer.domElement.addEventListener("pointermove", (event) => {
    if (!points || !data) return;

    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(points, false)[0];

    if (hit?.index !== undefined && hit.index < data.points.length) {
      tooltip.hidden = false;
      tooltip.innerHTML = describe(data.points[hit.index]);
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
    } else {
      tooltip.hidden = true;
    }
  });

  renderer.domElement.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
  });

  // ---- Vòng vẽ ------------------------------------------------------
  let running = true;

  function animate() {
    if (!running) return;
    requestAnimationFrame(animate);

    // Không nhìn thấy thì thôi không vẽ, đỡ tốn pin.
    if (!enabled) return;

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  const resize = new ResizeObserver(() => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return;

    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });
  resize.observe(container);

  return {
    set,
    setLabels(visible: boolean) {
      labelsWanted = visible;
      buildLabels();
    },
    setSpinning(spinning: boolean) {
      controls.autoRotate = spinning;
    },
    setEnabled(next: boolean) {
      enabled = next;
    },
    resetView() {
      camera.position.set(2.4, 1.7, 2.8);
      controls.target.set(0, 0, 0);
      controls.update();
    },
    stats() {
      return {
        points: points?.geometry.getAttribute("position").count ?? 0,
        labels: labelGroup.children.length,
        drawCalls: renderer.info.render.calls,
        camera: [camera.position.x, camera.position.y, camera.position.z],
        spinning: controls.autoRotate,
      };
    },
  };
}
