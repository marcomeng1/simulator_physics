// Three.js replay of exported traces: one body hierarchy per engine, overlay or side by side.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const mjQuat = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

function canvasTexture(w, h, draw) {
  const c = Object.assign(document.createElement("canvas"), { width: w, height: h });
  draw(c.getContext("2d"));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// MuJoCo's default look: gradient skybox and a two-tone checker ground.
const sky = () => canvasTexture(2, 256, (g) => {
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, "#3b4a5e");
  grad.addColorStop(0.55, "#161b22");
  grad.addColorStop(1, "#07090c");
  g.fillStyle = grad;
  g.fillRect(0, 0, 2, 256);
});

const checker = () => {
  const tex = canvasTexture(128, 128, (g) => {
    g.fillStyle = "#2a3340";
    g.fillRect(0, 0, 128, 128);
    g.fillStyle = "#1c232c";
    g.fillRect(0, 0, 64, 64);
    g.fillRect(64, 64, 64, 64);
    g.strokeStyle = "rgba(160,180,205,0.18)";
    g.strokeRect(0.5, 0.5, 127, 127);
  });
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
};

/** Distinct ground textures so terrain materials read immediately in the replay. */
function terrainTexture(look) {
  const tex = canvasTexture(256, 256, (g) => {
    if (look === "glass") {
      g.fillStyle = "#6a9bb8";
      g.fillRect(0, 0, 256, 256);
      g.strokeStyle = "rgba(220,240,255,0.55)";
      g.lineWidth = 3;
      for (let i = 0; i <= 8; i++) {
        g.beginPath(); g.moveTo(i * 32, 0); g.lineTo(i * 32, 256); g.stroke();
        g.beginPath(); g.moveTo(0, i * 32); g.lineTo(256, i * 32); g.stroke();
      }
      g.fillStyle = "rgba(255,255,255,0.18)";
      g.fillRect(8, 8, 40, 12);
    } else if (look === "sand") {
      g.fillStyle = "#c9a05a";
      g.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 900; i++) {
        g.fillStyle = i % 3 ? "#b8893f" : "#d9b574";
        g.fillRect((i * 47) % 256, (i * 91) % 256, 2, 2);
      }
      g.strokeStyle = "rgba(120,80,30,0.25)";
      for (let y = 10; y < 256; y += 18) {
        g.beginPath();
        g.moveTo(0, y);
        for (let x = 0; x < 256; x += 8) g.lineTo(x, y + Math.sin(x * 0.08 + y) * 4);
        g.stroke();
      }
    } else if (look === "mud") {
      g.fillStyle = "#3a2516";
      g.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 40; i++) {
        g.fillStyle = i % 2 ? "#2a180e" : "#4a3020";
        g.beginPath();
        g.ellipse((i * 67) % 256, (i * 99) % 256, 18 + (i % 5) * 4, 10 + (i % 4) * 3, i, 0, 7);
        g.fill();
      }
    } else if (look === "sponge") {
      g.fillStyle = "#e87888";
      g.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 120; i++) {
        g.fillStyle = i % 2 ? "#f0a0a8" : "#d06070";
        g.beginPath();
        g.arc((i * 53) % 256, (i * 97) % 256, 4 + (i % 5), 0, 7);
        g.fill();
      }
    } else { // concrete / flat
      g.fillStyle = "#8a8e94";
      g.fillRect(0, 0, 256, 256);
      g.fillStyle = "#7a7e84";
      g.fillRect(0, 0, 128, 128);
      g.fillRect(128, 128, 128, 128);
      g.strokeStyle = "rgba(40,40,45,0.35)";
      g.strokeRect(0.5, 0.5, 127, 127);
      g.strokeRect(128.5, 128.5, 127, 127);
      for (let i = 0; i < 200; i++) {
        g.fillStyle = "rgba(60,60,65,0.25)";
        g.fillRect((i * 37) % 256, (i * 73) % 256, 1, 1);
      }
    }
  });
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  return tex;
}

function terrainLook(name = "") {
  const m = /floor_(\w+)/.exec(name || "");
  return m ? m[1] : "concrete";
}

export function frameAt(t, time) {
  let lo = 0, hi = t.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (t[mid] <= time + 1e-9) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function geomMesh(g, meshes, cache) {
  let geo;
  const s = g.size;
  switch (g.type) {
    case "box": geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    case "sphere": geo = new THREE.SphereGeometry(s[0], 32, 16); break;
    case "capsule":
      geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2); break;
    case "cylinder":
      geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2); break;
    case "plane": geo = new THREE.PlaneGeometry(2 * (s[0] || 10), 2 * (s[1] || 10)); break;
    case "mesh": {
      if (!cache.has(g.mesh)) {
        const m = meshes[g.mesh];
        const b = new THREE.BufferGeometry();
        b.setAttribute("position", new THREE.BufferAttribute(m.vert, 3));
        b.setIndex(new THREE.BufferAttribute(m.face, 1));
        b.computeVertexNormals();
        cache.set(g.mesh, b);
      }
      geo = cache.get(g.mesh);
      break;
    }
    default: return null;
  }
  return geo;
}

export class Viewer {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    el.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.005, 50);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envMap = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    this.sky = sky();
    this.checker = checker();
    this.hemi = new THREE.HemisphereLight(0xdfe8ff, 0x20242a, 1.6);
    this.scene.add(this.hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(1.5, -1.2, 2.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0002;
    sun.shadow.normalBias = 0.002;
    Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1 });
    this.scene.add(sun, sun.target);
    this.sun = sun;
    this.setStudio(true);
    this.mode = "overlay";
    this.showForces = true;
    this.showTrails = true;
    this.showPoints = true;
    this.opacity = 1;
    this.forceScale = 0.01;
    this.pointGeo = new THREE.SphereGeometry(1, 10, 8);
    this.labels = [];
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => {
      this.controls.update();
      this.render();
      requestAnimationFrame(loop);
    };
    loop();
  }

  /** Image-based lighting, filmic tone mapping, sky and checker floor; off is the flat look. */
  setStudio(on) {
    this.studio = on;
    const r = this.renderer;
    r.toneMapping = on ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    r.toneMappingExposure = on ? 1.05 : 1;
    this.scene.environment = on ? this.envMap : null;
    this.scene.environmentIntensity = 0.55;
    this.scene.background = on ? this.sky : new THREE.Color(0x0b0e11);
    this.hemi.intensity = on ? 0.5 : 1.6;
    this.sun.intensity = on ? 2.6 : 2.2;
    this.floors?.forEach((m) => {
      m.map = on ? m.userData.map : null;
      // Keep material tint from the terrain texture; only fall back to dark when map is off.
      m.color.set(on ? 0xffffff : 0x1b2027);
      m.needsUpdate = true;
    });
    this.scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.el;
    this.renderer.setSize(w, h);
  }

  load(replay, colors) {
    if (this.root) this.scene.remove(this.root);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.replay = replay;
    this.colors = colors;
    const cache = new Map();
    const statics = new THREE.Group();
    this.floors = [];
    for (const g of replay.geoms.filter((g) => g.body < 0)) {
      const geo = geomMesh(g, replay.meshes, cache);
      if (!geo) continue;
      const plane = g.type === "plane";
      const look = terrainLook(g.name);
      const base = new THREE.Color(...g.rgba.slice(0, 3));
      const mat = new THREE.MeshStandardMaterial({
        color: plane ? 0xffffff : base,
        roughness: look === "glass" ? 0.12 : look === "mud" ? 0.95 : 0.85,
        metalness: look === "glass" ? 0.35 : 0.02,
        transparent: g.rgba[3] < 0.99, opacity: g.rgba[3] });
      if (!plane) mat.userData = { alpha: g.rgba[3], fixed: true };
      if (plane) {
        const map = terrainTexture(look);
        mat.userData = { map, extent: [2 * (g.size[0] || 10), 2 * (g.size[1] || 10)] };
        this.floors.push(mat);
      }
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.fromArray(g.pos);
      mesh.quaternion.copy(mjQuat(g.quat));
      mesh.receiveShadow = true;
      statics.add(mesh);
    }
    this.root.add(statics);
    this.statics = statics;

    this.routes = {};
    for (const name of Object.keys(replay.routes)) {
      const group = new THREE.Group();
      const bodies = replay.bodies.map(() => {
        const b = new THREE.Group();
        group.add(b);
        return b;
      });
      const tint = new THREE.Color(colors[name] ?? "#888888");
      for (const g of replay.geoms.filter((g) => g.body >= 0)) {
        const geo = geomMesh(g, replay.meshes, cache);
        if (!geo) continue;
        const base = new THREE.Color(...g.rgba.slice(0, 3));
        const mat = new THREE.MeshStandardMaterial({ color: base, roughness: 0.42,
                                                     metalness: 0.08 });
        mat.userData = { base, tint };
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.fromArray(g.pos);
        mesh.quaternion.copy(mjQuat(g.quat));
        mesh.castShadow = true;
        bodies[g.body].add(mesh);
      }
      let flex = null;
      if (replay.flex) {
        const idx = replay.flex.bodies;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(idx.length * 3), 3));
        geo.setIndex(replay.flex.faces.flat());
        const base = new THREE.Color(...replay.flex.rgba.slice(0, 3));
        const mat = new THREE.MeshStandardMaterial({ color: base, roughness: 0.6,
                                                     side: THREE.DoubleSide });
        mat.userData = { base, tint };
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.frustumCulled = false;
        group.add(mesh);
        flex = { geo, idx };
      }
      const arrows = new THREE.Group();
      const skip = new Set(replay.flex?.bodies ?? []);
      const trail = this.makeTrails(replay.routes[name], tint, skip);
      this.root.add(group, arrows, trail);
      this.routes[name] = { name, group, bodies, arrows, trail, flex, data: replay.routes[name] };
    }
    this.seek(0);
    if (!replay.geoms.some((g) => g.type === "plane")) this.addFloor();
    this.frameCamera();
    this.setStudio(this.studio);
  }

  // Scenes without a ground plane get a floor below everything that moves in the replay;
  // bodies falling more than 0.3 m below the fixed geometry are treated as leaving the scene.
  addFloor() {
    const fixed = this.statics.children.length
      ? new THREE.Box3().setFromObject(this.statics).min.z : 0;
    const base = Math.min(0, fixed);
    let bottom = base;
    for (const r of Object.values(this.routes)) {
      const { frames, nb, com } = r.data;
      r.bodies.forEach((b, i) => {
        const below = b.children.length
          ? com[i * 3 + 2] - new THREE.Box3().setFromObject(b).min.z : 0;
        let low = Infinity;
        for (let f = 0; f < frames; f++) low = Math.min(low, com[(f * nb + i) * 3 + 2]);
        const z = low - below;
        if (z > base - 0.3) bottom = Math.min(bottom, z);
      });
    }
    const mat = new THREE.MeshStandardMaterial({ color: 0x1b2027, roughness: 0.75 });
    mat.userData = { map: this.checker.clone(), extent: [2, 2] };
    this.floors.push(mat);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    floor.position.z = bottom - 0.002;
    floor.receiveShadow = true;
    this.statics.add(floor);
  }

  makeTrails(data, color, skip) {
    const group = new THREE.Group();
    const { frames, nb, com } = data;
    for (let b = 0; b < nb; b++) {
      if (skip.has(b)) continue;
      const pts = new Float32Array(frames * 3);
      let moved = 0;
      for (let f = 0; f < frames; f++) {
        for (let i = 0; i < 3; i++) pts[f * 3 + i] = com[(f * nb + b) * 3 + i];
        if (f) moved = Math.max(moved, Math.hypot(pts[f * 3] - pts[0], pts[f * 3 + 1] - pts[1],
                                                  pts[f * 3 + 2] - pts[2]));
      }
      if (moved < 0.005) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
      group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true,
                                                                  opacity: 0.8 })));
    }
    return group;
  }

  frameCamera() {
    const box = new THREE.Box3();
    for (const r of Object.values(this.routes)) {
      r.flex?.geo.computeBoundingBox();
      box.expandByObject(r.group);
    }
    const center = box.getCenter(new THREE.Vector3());
    const size = Math.max(0.25, box.getSize(new THREE.Vector3()).length());
    this.pointSize = Math.min(0.006, Math.max(0.0015, size / 250));
    const square = size > 1 ? 0.25 : 0.05;
    this.floors.forEach((m) => m.userData.map.repeat.set(...m.userData.extent.map((e) =>
      e / (2 * square))));
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(0.9, -1.25, 0.65).multiplyScalar(size));
    const half = Math.max(0.3, size * 0.75);
    Object.assign(this.sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half,
                                            near: 0.01, far: size * 10 + 5 });
    this.sun.shadow.camera.updateProjectionMatrix();
    this.sunOffset = new THREE.Vector3(0.6, -0.5, 1).multiplyScalar(size * 2 + 1);
    this.sunCenter = center.clone().sub(Object.values(this.routes)[0]?.bodies[0].position
      ?? new THREE.Vector3());
    this.followSun();
    this.camera.near = size / 200;
    this.camera.far = size * 40;
    this.camera.updateProjectionMatrix();
  }

  // Keep the shadow region on the first body, so walking robots stay shadowed.
  followSun() {
    const lead = Object.values(this.routes ?? {})[0]?.bodies[0];
    if (!lead || !this.sunCenter) return;
    this.sun.target.position.copy(lead.position).add(this.sunCenter);
    this.sun.position.copy(this.sun.target.position).add(this.sunOffset);
  }

  setVisible(names) {
    this.visible = names;
    const stacked = this.mode === "overlay" ? names.length : 1;
    const each = this.opacity >= 1 ? 1 : 1 - (1 - this.opacity) ** (1 / stacked);
    names.forEach((name, i) => {
      const r = this.routes[name];
      if (!r) return;
      const ghost = this.mode === "overlay" && i > 0;
      const alpha = Math.min(ghost ? 0.45 : 1, each);
      r.group.traverse((o) => {
        if (!o.material?.userData?.base) return;
        const { base, tint } = o.material.userData;
        o.material.color.copy(base).lerp(tint, ghost ? 0.65 : 0.12);
        o.material.transparent = alpha < 1;
        o.material.opacity = alpha;
        o.material.depthWrite = alpha >= 1;
      });
    });
    this.statics?.traverse((o) => {
      const u = o.material?.userData;
      if (!u?.fixed) return;
      const alpha = Math.min(u.alpha, this.opacity < 1 ? Math.max(0.15, this.opacity) : 1);
      o.material.transparent = alpha < 1;
      o.material.opacity = alpha;
      o.material.depthWrite = alpha >= 1;
    });
    this.updateLabels();
  }

  setOpacity(value) {
    this.opacity = value;
    if (this.visible) this.setVisible(this.visible);
  }

  updateLabels() {
    this.labels.forEach((l) => l.remove());
    this.labels = [];
    if (this.mode !== "side") return;
    const names = this.visible || [];
    names.forEach((name, i) => {
      const l = document.createElement("div");
      l.className = "vp-label";
      l.style.left = `${(100 * i) / names.length + 0.6}%`;
      l.style.borderLeft = `3px solid ${this.colors[name]}`;
      l.textContent = this.routeLabel?.(name) ?? name;
      this.el.appendChild(l);
      this.labels.push(l);
    });
  }

  // Each route has its own sampling (the article's bundles are 10 ms), so seek by time.
  seek(t) {
    this.t = t;
    for (const r of Object.values(this.routes)) {
      const d = r.data;
      const k = frameAt(d.t, t);
      r.bodies.forEach((b, i) => {
        const o = (k * d.nb + i) * 3;
        const q = (k * d.nb + i) * 4;
        b.position.set(d.pos[o], d.pos[o + 1], d.pos[o + 2]);
        b.quaternion.set(d.quat[q + 1], d.quat[q + 2], d.quat[q + 3], d.quat[q]).normalize();
      });
      if (r.flex) {
        const { geo, idx } = r.flex;
        const p = geo.attributes.position.array;
        idx.forEach((b, j) => {
          const o = (k * d.nb + b) * 3;
          p[3 * j] = d.pos[o]; p[3 * j + 1] = d.pos[o + 1]; p[3 * j + 2] = d.pos[o + 2];
        });
        geo.attributes.position.needsUpdate = true;
        geo.computeVertexNormals();
        geo.computeBoundingSphere();
      }
      this.drawArrows(r, k);
    }
    this.followSun();
  }

  drawArrows(r, k) {
    r.arrows.clear();
    const c = r.data.contacts;
    if (!c || (!this.showForces && !this.showPoints)) return;
    const color = new THREE.Color(this.colors[r.name]);
    const onTop = (o) => {
      o.renderOrder = 10;
      o.traverse((m) => {
        if (m.material) Object.assign(m.material, { depthTest: false, transparent: true });
      });
      return o;
    };
    const n = c.start[k + 1] - c.start[k];
    if (this.showPoints && n) {
      // Yellow = nonzero force; blue (50% opacity) = inactive proximity (~0 N).
      const size = this.pointSize ?? 0.003;
      const active = [], idle = [];
      for (let i = 0; i < n; i++) {
        const j = c.start[k] + i;
        const f = Math.hypot(c.force[3 * j], c.force[3 * j + 1], c.force[3 * j + 2]);
        (f > 1e-4 ? active : idle).push(j);
      }
      const addDots = (idxs, color, opacity, scale, order) => {
        if (!idxs.length) return;
        const dots = new THREE.InstancedMesh(this.pointGeo, new THREE.MeshBasicMaterial({
          color, depthTest: false, transparent: true, opacity, depthWrite: false }), idxs.length);
        const m = new THREE.Matrix4();
        const s = size * scale;
        idxs.forEach((j, i) => {
          m.makeScale(s, s, s).setPosition(c.pos[3 * j], c.pos[3 * j + 1], c.pos[3 * j + 2]);
          dots.setMatrixAt(i, m);
        });
        dots.renderOrder = order;
        r.arrows.add(dots);
      };
      addDots(idle, 0x6ec8ff, 0.5, 1.0, 11);
      addDots(active, 0xffe066, 0.95, 1.15, 12);
    }
    if (!this.showForces) return;
    for (let i = c.start[k]; i < c.start[k + 1]; i++) {
      const dir = new THREE.Vector3(c.force[3 * i], c.force[3 * i + 1], c.force[3 * i + 2]);
      const len = dir.length() * this.forceScale;
      if (len < 1e-4) continue;
      const origin = new THREE.Vector3(c.pos[3 * i], c.pos[3 * i + 1], c.pos[3 * i + 2]);
      const head = Math.min(0.012, len * 0.3);
      r.arrows.add(onTop(new THREE.ArrowHelper(dir.normalize(), origin, len, color, head,
                                               head * 0.5)));
    }
  }

  render() {
    if (!this.routes) return;
    const names = this.visible || Object.keys(this.routes);
    const { clientWidth: w, clientHeight: h } = this.el;
    const all = Object.entries(this.routes);
    const show = (pred) => all.forEach(([n, r]) => {
      const on = pred(n);
      r.group.visible = on;
      r.arrows.visible = on;
      r.trail.visible = on && this.showTrails;
    });
    if (this.mode === "side" && names.length > 1) {
      this.renderer.setScissorTest(true);
      const vw = Math.floor(w / names.length);
      this.camera.aspect = vw / h;
      this.camera.updateProjectionMatrix();
      names.forEach((name, i) => {
        show((n) => n === name);
        this.renderer.setViewport(i * vw, 0, vw, h);
        this.renderer.setScissor(i * vw, 0, vw, h);
        this.renderer.render(this.scene, this.camera);
      });
      this.renderer.setScissorTest(false);
    } else {
      show((n) => names.includes(n));
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setViewport(0, 0, w, h);
      this.renderer.render(this.scene, this.camera);
    }
  }
}
