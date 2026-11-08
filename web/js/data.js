// Fetch and decode exported data (see packages/simbench/src/simbench/web.py).

const cache = new Map();

export async function json(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(path).then((r) => {
      if (!r.ok) throw new Error(`${path}: ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(path);
}

function bytes(b64) {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out.buffer;
}

export const f32 = (b64) => new Float32Array(bytes(b64));
export const u32 = (b64) => new Uint32Array(bytes(b64));

function scaled(b64, scale) {
  const raw = new Int16Array(bytes(b64));
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw[i] * scale;
  return out;
}

// One route's replay as typed arrays: pos/com/force (frames x bodies x 3), quat (x 4).
export function decodeRoute(r, scales, nb) {
  const out = {
    t: f32(r.t), frames: r.frames, nb,
    pos: scaled(r.pos, scales.pos), quat: scaled(r.quat, scales.quat),
    com: scaled(r.com, scales.pos), force: scaled(r.force, scales.force),
  };
  if (r.q) { out.q = f32(r.q); out.nq = r.nq; }
  if (r.contacts) {
    const count = r.contacts.count;
    const start = new Uint32Array(count.length + 1);
    for (let i = 0; i < count.length; i++) start[i + 1] = start[i] + count[i];
    out.contacts = { start, pos: scaled(r.contacts.pos, scales.pos),
                     force: scaled(r.contacts.force, scales.force) };
  }
  return out;
}

const meshCache = new Map();

function mesh(key, all) {
  if (!meshCache.has(key)) {
    const m = all[key];
    const buf = bytes(m.face);
    meshCache.set(key, { vert: scaled(m.vert, m.scale),
                         face: m.index_bytes === 2 ? new Uint16Array(buf) : new Uint32Array(buf) });
  }
  return meshCache.get(key);
}

export async function loadReplay(file) {
  const [raw, all] = await Promise.all([json(`data/replays/${file}`), json("data/meshes.json")]);
  const nb = raw.bodies.length;
  const routes = {};
  for (const [name, r] of Object.entries(raw.routes)) routes[name] = decodeRoute(r, raw.scales, nb);
  const meshes = {};
  for (const g of raw.geoms) if (g.mesh) meshes[g.mesh] = mesh(g.mesh, all);
  return { ...raw, routes, meshes };
}
