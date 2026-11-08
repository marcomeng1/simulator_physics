"""Prototype of the article's flex ball (raw MuJoCo, no simbench Scene): mesh, lumped
masses and plate compression, used to pick the flex attributes.
Usage: python flex_proto.py [radius]

Run from the repo root inside a route env: uv run --locked --project envs/<route> python scripts/probes/flex_proto.py ...
"""
import gzip, json, sys, time
import numpy as np
import mujoco

REF = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "reference/article/assets/deformable") + "/"
g = json.load(gzip.open(REF + "grasp-2N.json.gz"))["traces"][0]["scene"]["ball_mesh"]
V = np.array(g["vertices_m"]); T = np.array(g["tetrahedra"]); M = np.array(g["node_mass_kg"])
radius = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
dt = 5e-5

def fmt(a): return " ".join(f"{x:.12g}" for x in np.ravel(a))

bodies = "\n".join(
    f'<body name="ball_{i}" pos="{fmt(v)}"><inertial pos="0 0 0" mass="{float(M[i])!r}" diaginertia="1e-9 1e-9 1e-9"/>'
    f'<joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/></body>'
    for i, v in enumerate(V))
names = " ".join(f"ball_{i}" for i in range(len(V)))
xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="Euler" solver="Newton" iterations="100" tolerance="1e-10" cone="elliptic"/>
<worldbody>
<body name="plate_lo" pos="0 0 -0.0351"><joint name="lo" type="slide" axis="0 0 1"/><geom type="box" size="0.05 0.05 0.005" mass="1" contype="4" conaffinity="2" friction="0.5 0 0" solref="0.01 1" solimp="0.99 0.999 0.001 0.5 2"/></body>
<body name="plate_hi" pos="0 0 0.0351"><joint name="hi" type="slide" axis="0 0 1"/><geom type="box" size="0.05 0.05 0.005" mass="1" contype="4" conaffinity="2" friction="0.5 0 0" solref="0.01 1" solimp="0.99 0.999 0.001 0.5 2"/></body>
{bodies}
</worldbody>
<deformable><flex name="ball" dim="3" body="{names}" element="{fmt(T)}" radius="{radius}">
<contact contype="2" conaffinity="5" condim="3" friction="0.5 0 0" solref="0.01 1" solimp="0.99 0.999 0.001 0.5 2" selfcollide="none"/>
<elasticity young="20000" poisson="0.3" damping="2e-5"/></flex></deformable>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
print(mujoco.__version__, "nv", m.nv, "flex", m.nflex, "nflexelem", m.nflexelem, "mass", m.body_mass[3:].sum())

def smooth(t, t0, t1):
    s = np.clip((t - t0) / (t1 - t0), 0, 1); return 3 * s**2 - 2 * s**3, (6 * s - 6 * s**2) / (t1 - t0) * (0 < s < 1)

def zref(t):
    a, ad = smooth(t, 0.1, 0.6); b, bd = smooth(t, 0.8, 1.3)
    return 0.006 * (a - b), 0.006 * (ad - bd)

lo, hi = m.body("plate_lo").id, m.body("plate_hi").id
qa_lo, qa_hi = m.jnt_qposadr[m.joint("lo").id], m.jnt_qposadr[m.joint("hi").id]
f6 = np.zeros(6); out = []
start = time.time()
for k in range(int(1.5 / dt)):
    t = k * dt
    z, zd = zref(t)
    for qa, sgn in ((qa_lo, 1), (qa_hi, -1)):
        e = sgn * z - d.qpos[qa]; ed = sgn * zd - d.qvel[qa]
        d.qfrc_applied[qa] = 1e5 * e + 600 * ed
    mujoco.mj_step(m, d)
    if k % 20 == 0:
        F = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]
            for side, gid in ((-1, c.geom[0]), (1, c.geom[1])):
                if gid >= 0 and m.geom_bodyid[gid] == hi:
                    F += side * fw[2]
        x = d.flexvert_xpos.copy()
        out.append((t, d.xpos[hi][2], F, np.ptp(x[:, 2]), np.isfinite(x).all()))
print(f"{time.time()-start:.1f}s")
for t, z, F, ez, ok in out[::50]:
    print(f"t={t:.2f} plate {z*1e3:.3f} F {F:.3f} extent_z {ez*1e3:.2f} ok {ok}")
hold = [F for t, z, F, e, ok in out if 0.6 <= t <= 0.8]
ext = [e for t, z, F, e, ok in out if 0.6 <= t <= 0.8]
print("hold mean F", np.mean(hold), "extent", np.mean(ext) * 1e3)
