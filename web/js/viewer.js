// Three.js replay of exported traces: one body hierarchy per engine, overlay or side by side.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const mjQuat = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

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
    case "plane": geo = new THREE.PlaneGeometry(2 * (s[0] || 1), 2 * (s[1] || 1)); break;
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
    el.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.005, 50);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e11);
    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242a, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(1.5, -1.2, 2.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1 });
    this.scene.add(sun);
    this.mode = "overlay";
    this.showForces = true;
    this.showTrails = true;
    this.forceScale = 0.01;
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
    for (const g of replay.geoms.filter((g) => g.body < 0)) {
      const geo = geomMesh(g, replay.meshes, cache);
      if (!geo) continue;
      const plane = g.type === "plane";
      const mat = new THREE.MeshStandardMaterial({
        color: plane ? 0x1b2027 : new THREE.Color(...g.rgba.slice(0, 3)), roughness: 0.9,
        transparent: !plane && g.rgba[3] < 1, opacity: g.rgba[3] });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.fromArray(g.pos);
      mesh.quaternion.copy(mjQuat(g.quat));
      mesh.receiveShadow = true;
      statics.add(mesh);
    }
    if (!replay.geoms.some((g) => g.type === "plane")) {
      const grid = new THREE.GridHelper(2, 40, 0x2a3038, 0x1a1f26);
      grid.rotateX(Math.PI / 2);
      grid.position.z = -0.0005;
      statics.add(grid);
    }
    this.root.add(statics);

    this.routes = {};
    for (const name of Object.keys(replay.routes)) {
      const group = new THREE.Group();
      const bodies = replay.bodies.map(() => {
        const b = new THREE.Group();
        group.add(b);
        return b;
      });
      const tint = new THREE.Color(colors[name]);
      for (const g of replay.geoms.filter((g) => g.body >= 0)) {
        const geo = geomMesh(g, replay.meshes, cache);
        if (!geo) continue;
        const base = new THREE.Color(...g.rgba.slice(0, 3));
        const mat = new THREE.MeshStandardMaterial({ color: base, roughness: 0.55,
                                                     metalness: 0.05 });
        mat.userData = { base, tint };
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.fromArray(g.pos);
        mesh.quaternion.copy(mjQuat(g.quat));
        mesh.castShadow = true;
        bodies[g.body].add(mesh);
      }
      const arrows = new THREE.Group();
      const trail = this.makeTrails(replay.routes[name], tint);
      this.root.add(group, arrows, trail);
      this.routes[name] = { name, group, bodies, arrows, trail, data: replay.routes[name] };
    }
    this.frame(0);
    this.frameCamera();
  }

  makeTrails(data, color) {
    const group = new THREE.Group();
    const { frames, nb, com } = data;
    for (let b = 0; b < nb; b++) {
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
    for (const r of Object.values(this.routes)) box.expandByObject(r.group);
    const center = box.getCenter(new THREE.Vector3());
    const size = Math.max(0.25, box.getSize(new THREE.Vector3()).length());
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(0.9, -1.25, 0.65).multiplyScalar(size));
    this.camera.near = size / 200;
    this.camera.far = size * 40;
    this.camera.updateProjectionMatrix();
  }

  setVisible(names) {
    this.visible = names;
    names.forEach((name, i) => {
      const r = this.routes[name];
      const ghost = this.mode === "overlay" && i > 0;
      r.group.traverse((o) => {
        if (!o.material?.userData?.base) return;
        const { base, tint } = o.material.userData;
        o.material.color.copy(base).lerp(tint, ghost ? 0.65 : 0.12);
        o.material.transparent = ghost;
        o.material.opacity = ghost ? 0.45 : 1;
        o.material.depthWrite = !ghost;
      });
    });
    this.updateLabels();
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

  frame(f) {
    this.f = f;
    for (const r of Object.values(this.routes)) {
      const d = r.data;
      const k = Math.min(f, d.frames - 1);
      r.bodies.forEach((b, i) => {
        const o = (k * d.nb + i) * 3;
        const q = (k * d.nb + i) * 4;
        b.position.set(d.pos[o], d.pos[o + 1], d.pos[o + 2]);
        b.quaternion.set(d.quat[q + 1], d.quat[q + 2], d.quat[q + 3], d.quat[q]).normalize();
      });
      this.drawArrows(r, k);
    }
  }

  drawArrows(r, k) {
    r.arrows.clear();
    const c = r.data.contacts;
    if (!this.showForces || !c) return;
    const color = new THREE.Color(this.colors[r.name]);
    for (let i = c.start[k]; i < c.start[k + 1]; i++) {
      const dir = new THREE.Vector3(c.force[3 * i], c.force[3 * i + 1], c.force[3 * i + 2]);
      const len = dir.length() * this.forceScale;
      if (len < 1e-4) continue;
      const origin = new THREE.Vector3(c.pos[3 * i], c.pos[3 * i + 1], c.pos[3 * i + 2]);
      const head = Math.min(0.012, len * 0.3);
      r.arrows.add(new THREE.ArrowHelper(dir.normalize(), origin, len, color, head, head * 0.5));
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
