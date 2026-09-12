/**
 * arch3d.ts — Sơ đồ kiến trúc model, nhìn bằng 3 chiều.
 *
 * Ba trục ở đây đều có nghĩa, không phải bày cho đẹp:
 *
 *     trục X  = dòng chảy, đọc từ trái sang phải
 *     trục Z  = tầng thứ mấy — mỗi tầng lùi ra sau một quãng,
 *               nên nhìn nghiêng là thấy ngay model sâu bao nhiêu
 *     trục Y  = nhánh: đường tắt vòng LÊN TRÊN, chuyên gia toả XUỐNG DƯỚI
 *
 * Bấm vào một bộ phận để xem nó ăn khớp với những bộ phận nào: phần
 * xung quanh được giữ sáng, phần không liên quan bị mờ đi.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ArchData, ArchNode, ArchSelection } from "../types";

const KIND_COLORS: Record<string, string> = {
  embed: "#4a9eff",
  norm: "#6b7a8d",
  attention: "#ff9f45",
  add: "#bc8cff",
  router: "#7ee787",
  expert: "#3d8b53",
  shared: "#56d364",
  ffn: "#f778ba",
  head: "#ff7b72",
};

const EDGE_COLORS: Record<string, string> = {
  data: "#4a6b8a",
  residual: "#8b6fd4",
  route: "#3d8b53",
  combine: "#2f6b40",
};

export const kindColor = (kind: string) => KIND_COLORS[kind] ?? "#8b98a8";
export const edgeColor = (kind: string) => EDGE_COLORS[kind] ?? "#4a6b8a";

/** Nhãn ngắn để vẽ trong không gian 3D — tên đầy đủ nằm ở bảng giải thích. */
const SHORT_LABELS: Record<string, string> = {
  "Token Embedding": "Embed",
  "Position Embedding": "Pos",
  RMSNorm: "Norm",
  LayerNorm: "Norm",
  Attention: "Attn",
  "Cộng": "+",
  "Cộng lại": "+",
  Router: "Router",
  "Dùng chung": "Shared",
  MLP: "MLP",
  "LM Head": "Head",
};

const shortLabel = (label: string) => SHORT_LABELS[label] ?? label;

export interface Arch3DView {
  set(data: ArchData): void;
  setLabels(visible: boolean): void;
  setFlow(visible: boolean): void;
  setSpinning(spinning: boolean): void;
  resetView(): void;
  setEnabled(enabled: boolean): void;
  select(id: string | null): void;
  stats(): {
    nodes: number;
    edges: number;
    labels: number;
    selected: string | null;
    drawCalls: number;
    camera: [number, number, number];
  };
}

export function createArch3D(
  container: HTMLElement,
  tooltip: HTMLElement,
  onSelect: (selection: ArchSelection | null) => void,
): Arch3DView {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);

  let renderer: THREE.WebGLRenderer;
  try {
    // alpha: true để nền trong suốt, cho màu nền của CSS hiện ra. Nếu tự
    // đặt màu nền trong three.js thì phải qua quy đổi không gian màu, và
    // nó sẽ lệch so với màu panel.
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch {
    container.innerHTML =
      '<p class="empty">Máy này không mở được WebGL nên không xem được sơ đồ 3D.</p>';
    return {
      set: () => {},
      setLabels: () => {},
      setFlow: () => {},
      setSpinning: () => {},
      resetView: () => {},
      setEnabled: () => {},
      select: () => {},
      stats: () => ({ nodes: 0, edges: 0, labels: 0, selected: null, drawCalls: 0, camera: [0, 0, 0] }),
    };
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.85;
  controls.autoRotateSpeed = 0.5;
  controls.autoRotate = true;

  const root = new THREE.Group();
  scene.add(root);

  const nodeLayer = new THREE.Group();
  const edgeLayer = new THREE.Group();
  const labelLayer = new THREE.Group();
  const flowLayer = new THREE.Group();
  root.add(edgeLayer, nodeLayer, labelLayer, flowLayer);

  type NodeVisual = {
    node: ArchNode;
    mesh: THREE.Mesh;
    outline: THREE.LineSegments;
    label: THREE.Sprite | null;
  };

  let data: ArchData | null = null;
  let visuals: NodeVisual[] = [];
  let edgeVisuals: { from: string; to: string; line: THREE.Line }[] = [];
  let selected: string | null = null;
  let labelsWanted = true;
  let flowWanted = true;
  let enabled = true;

  let flowCurve: THREE.CatmullRomCurve3 | null = null;
  const pulses: THREE.Mesh[] = [];

  // ---- Dọn sạch trước khi dựng cái mới ------------------------------
  function clear(group: THREE.Group) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.traverse((object) => {
        const any = object as THREE.Mesh;
        any.geometry?.dispose();
        const material = any.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(material)) material.forEach((m) => m.dispose());
        else material?.dispose();
        const sprite = object as THREE.Sprite;
        sprite.material?.map?.dispose();
      });
    }
  }

  function resetAll() {
    clear(nodeLayer);
    clear(edgeLayer);
    clear(labelLayer);
    clear(flowLayer);
    visuals = [];
    edgeVisuals = [];
    pulses.length = 0;
    flowCurve = null;
    selected = null;
  }

  // ---- Nhãn chữ ------------------------------------------------------
  function makeLabel(text: string, color: string, width: number): THREE.Sprite {
    const canvasWidth = 256;
    const canvasHeight = 80;

    const canvas = document.createElement("canvas");
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const ctx = canvas.getContext("2d")!;
    ctx.font = "bold 44px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = color;
    ctx.fillText(text, canvasWidth / 2, canvasHeight / 2 + 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;

    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }),
    );
    sprite.scale.set(width, (width * canvasHeight) / canvasWidth, 1);
    return sprite;
  }

  // ---- Kích thước hộp theo số thông số -------------------------------
  function nodeScale(node: ArchNode, maxParams: number): number {
    if (node.params <= 0) return 0.5; // nút cộng, không có thông số nào
    const t = Math.log10(1 + node.params) / Math.log10(1 + maxParams);
    return 0.5 + 0.85 * t;
  }

  // ---- Dựng sơ đồ ----------------------------------------------------
  function set(next: ArchData) {
    data = next;
    resetAll();
    resizeRenderer();

    const maxParams = Math.max(...next.nodes.map((n) => n.params), 1);
    const position = new Map<string, THREE.Vector3>();

    // Đo cả sơ đồ trước, để nhãn chữ và camera cùng ăn theo một kích thước.
    const bounds = new THREE.Box3();
    for (const node of next.nodes) {
      bounds.expandByPoint(new THREE.Vector3(node.x, node.y, node.z));
    }
    const modelSize = bounds.getSize(new THREE.Vector3());
    const labelWidth = Math.max(modelSize.x, modelSize.y, modelSize.z, 1) * 0.09;

    for (const node of next.nodes) {
      const scale = nodeScale(node, maxParams);
      const geometry = new THREE.BoxGeometry(0.52 * scale, 0.4 * scale, 0.46 * scale);
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({
          color: kindColor(node.kind),
          transparent: true,
          opacity: 0.85,
        }),
      );
      mesh.position.set(node.x, node.y, node.z);
      nodeLayer.add(mesh);

      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.22,
        }),
      );
      outline.position.copy(mesh.position);
      nodeLayer.add(outline);

      // Chuyên gia không gắn nhãn: có tới 24 cái, gắn hết thì rối mắt.
      // Tên của chúng hiện khi rê chuột và ở bảng giải thích bên dưới.
      const label =
        node.kind === "expert"
          ? null
          : makeLabel(shortLabel(node.label), kindColor(node.kind), labelWidth);

      if (label) {
        label.position.set(node.x, node.y + 0.42 * scale + labelWidth * 0.45, node.z);
        labelLayer.add(label);
      }

      visuals.push({ node, mesh, outline, label });
      position.set(node.id, mesh.position.clone());
    }

    for (const edge of next.edges) {
      const a = position.get(edge.from);
      const b = position.get(edge.to);
      if (!a || !b) continue;

      // Đường tắt thì vẽ cong lên, để thấy nó đi vòng qua chứ không
      // chạy xuyên qua các bộ phận.
      let points: THREE.Vector3[];
      if (edge.kind === "residual") {
        const lift = Math.max(a.y, b.y) + 0.55;
        const curve = new THREE.QuadraticBezierCurve3(
          a.clone(),
          new THREE.Vector3((a.x + b.x) / 2, lift, (a.z + b.z) / 2),
          b.clone(),
        );
        points = curve.getPoints(24);
      } else {
        points = [a.clone(), b.clone()];
      }

      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({
          color: edgeColor(edge.kind),
          transparent: true,
          opacity: edge.kind === "data" ? 0.75 : 0.5,
        }),
      );
      edgeLayer.add(line);
      edgeVisuals.push({ from: edge.from, to: edge.to, line });
    }

    // ---- Hạt sáng chạy dọc đường trục, cho thấy chiều dòng chảy ----
    const spinePoints = next.spine
      .map((id) => position.get(id))
      .filter((p): p is THREE.Vector3 => Boolean(p));

    if (spinePoints.length > 1) {
      flowCurve = new THREE.CatmullRomCurve3(spinePoints);

      for (let i = 0; i < 4; i += 1) {
        const pulse = new THREE.Mesh(
          new THREE.SphereGeometry(0.13, 12, 12),
          new THREE.MeshBasicMaterial({ color: 0x8fd4ff, transparent: true, opacity: 0.95 }),
        );
        pulse.userData.offset = i / 4;
        flowLayer.add(pulse);
        pulses.push(pulse);
      }
    }

    applyLabels();
    applyFlow();
    fitCamera();
    applyHighlight();
    onSelect(null); // đổi model thì bỏ lựa chọn cũ
  }

  // ---- Camera nhìn vừa cả sơ đồ --------------------------------------
  //
  // Phải tính theo CẢ bề ngang lẫn bề dọc. Sơ đồ này dài mà thấp, nếu chỉ
  // lấy chiều dài nhất làm chuẩn thì camera lùi quá xa và mọi thứ bé tí.
  function resizeRenderer(): boolean {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return false;

    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    return true;
  }

  function fitCamera() {
    const box = new THREE.Box3().setFromObject(nodeLayer);
    if (box.isEmpty()) return;

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    const halfVertical = (camera.fov * Math.PI) / 360;
    const halfHorizontal = Math.atan(Math.tan(halfVertical) * Math.max(camera.aspect, 0.25));

    const distance =
      Math.max(
        size.y / 2 / Math.tan(halfVertical),
        size.x / 2 / Math.tan(halfHorizontal),
        size.z / 2 / Math.tan(halfVertical),
      ) * 1.3;

    camera.near = Math.max(0.05, distance * 0.01);
    camera.far = distance * 12;
    camera.position.set(
      center.x + distance * 0.08,
      center.y + distance * 0.5,
      center.z + distance * 0.86,
    );
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.minDistance = distance * 0.12;
    controls.maxDistance = distance * 2.6;
    controls.update();
  }

  // ---- Bật/tắt nhãn và hạt sáng --------------------------------------
  function applyLabels() {
    for (const visual of visuals) if (visual.label) visual.label.visible = labelsWanted;
  }

  function applyFlow() {
    for (const pulse of pulses) pulse.visible = flowWanted;
  }

  // ---- Bấm vào một bộ phận: giữ sáng hàng xóm, mờ phần còn lại -------
  function neighbours(id: string): Set<string> {
    const keep = new Set<string>([id]);
    if (!data) return keep;

    for (const edge of data.edges) {
      if (edge.from === id) keep.add(edge.to);
      if (edge.to === id) keep.add(edge.from);
    }
    return keep;
  }

  function applyHighlight() {
    const keep = selected ? neighbours(selected) : null;

    for (const visual of visuals) {
      const on = !keep || keep.has(visual.node.id);
      const isSelected = visual.node.id === selected;

      (visual.mesh.material as THREE.MeshBasicMaterial).opacity = on ? 0.85 : 0.07;
      (visual.outline.material as THREE.LineBasicMaterial).opacity = isSelected ? 0.95 : on ? 0.22 : 0.05;

      if (visual.label) {
        (visual.label.material as THREE.SpriteMaterial).opacity = on ? 1 : 0.1;
      }
    }

    for (const visual of edgeVisuals) {
      const touches = !keep || keep.has(visual.from) || keep.has(visual.to);
      (visual.line.material as THREE.LineBasicMaterial).opacity = touches ? 0.75 : 0.05;
    }
  }

  function select(id: string | null) {
    selected = id;
    applyHighlight();
    if (!id) {
      onSelect(null);
      return;
    }

    const node = data?.nodes.find((n) => n.id === id);
    if (!node || !data) return;

    const label = (nodeId: string) => data!.nodes.find((n) => n.id === nodeId)?.label ?? nodeId;

    onSelect({
      node,
      incoming: data.edges.filter((e) => e.to === id).map((e) => label(e.from)),
      outgoing: data.edges.filter((e) => e.from === id).map((e) => label(e.to)),
    });
  }

  // ---- Rê chuột ------------------------------------------------------
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  function nodeAt(event: PointerEvent): ArchNode | null {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);
    const meshes = visuals.map((v) => v.mesh);
    const hit = raycaster.intersectObjects(meshes, false)[0];
    if (!hit) return null;

    return visuals.find((v) => v.mesh === hit.object)?.node ?? null;
  }

  renderer.domElement.addEventListener("pointermove", (event) => {
    const node = nodeAt(event);

    if (!node) {
      tooltip.hidden = true;
      renderer.domElement.style.cursor = "grab";
      return;
    }

    tooltip.hidden = false;
    tooltip.innerHTML =
      `<b>${node.label}</b>` +
      (node.params ? `<br>${node.params.toLocaleString("vi-VN")} thông số` : "") +
      `<br><span style="color:#8b98a8">bấm để xem liên hệ</span>`;
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
    renderer.domElement.style.cursor = "pointer";
  });

  renderer.domElement.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
  });

  // Phân biệt bấm thật với kéo xoay: chỉ chọn khi chuột gần như đứng yên.
  let downAt: { x: number; y: number } | null = null;

  renderer.domElement.addEventListener("pointerdown", (event) => {
    downAt = { x: event.clientX, y: event.clientY };
  });

  renderer.domElement.addEventListener("pointerup", (event) => {
    if (!downAt) return;
    const moved = Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y);
    downAt = null;
    if (moved > 5) return; // vừa kéo xoay, không phải bấm chọn

    const node = nodeAt(event);
    select(node && node.id !== selected ? node.id : null);
  });

  // ---- Vòng vẽ -------------------------------------------------------
  let running = true;

  function animate(time: number) {
    if (!running) return;
    requestAnimationFrame(animate);
    if (!enabled) return;

    controls.update();

    if (flowCurve && flowWanted) {
      for (const pulse of pulses) {
        const t = (time * 0.00008 + (pulse.userData.offset as number)) % 1;
        pulse.position.copy(flowCurve.getPointAt(t));
      }
    }

    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);

  const resize = new ResizeObserver(() => {
    // Chỉ chỉnh tỉ lệ khung, không chỉnh lại camera — nếu không thì mỗi lần
    // đổi cỡ cửa sổ là mất góc nhìn người dùng vừa xoay.
    resizeRenderer();
  });
  resize.observe(container);

  return {
    set,
    setLabels(visible: boolean) {
      labelsWanted = visible;
      applyLabels();
    },
    setFlow(visible: boolean) {
      flowWanted = visible;
      applyFlow();
    },
    setSpinning(spinning: boolean) {
      controls.autoRotate = spinning;
    },
    resetView() {
      select(null);
      fitCamera();
    },
    setEnabled(next: boolean) {
      enabled = next;
    },
    select,
    stats() {
      return {
        nodes: visuals.length,
        edges: edgeVisuals.length,
        labels: visuals.filter((v) => v.label?.visible).length,
        selected,
        drawCalls: renderer.info.render.calls,
        camera: [camera.position.x, camera.position.y, camera.position.z],
      };
    },
  };
}
