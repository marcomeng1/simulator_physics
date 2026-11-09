"""Spike: Genesis rigid Panda + FEM soft peg + rigid bore (vertex constraints to hand).

Attachment path that works in this Genesis build:
  FEMOptions(use_implicit_solver=True, enable_vertex_constraints=True)
  + LegacyCouplerOptions(rigid_fem=True)
  + hard set_vertex_constraints(..., link=hand)

Explicit FEM (default) goes NaN even for a free cylinder; soft/hard constraints on
the explicit solver also explode. If implicit+link attach fails, fallback is
kinematic mount drive: rewrite mount vertex positions from the hand pose each step
(weaker one-way coupling; no reaction wrench into the arm).

Usage (repo root):
  uv run --locked --project envs/genesis python scripts/probes/insertion_fem_spike.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import numpy as np


def main() -> int:
    import genesis as gs
    import torch

    from simbench.readback import quat_to_mat
    from simbench.scenes.insertion import PEG_LENGTH, PEG_MOUNT, PEG_R, peg_mount, insertion

    scene = insertion(dt=0.001, clearance_mm=0.10, case="tilted")
    case = scene.params["case"]
    path = scene.mjcf_path()
    tree = ET.parse(path)
    peg_geom = tree.find(".//geom[@name='peg_geom']")
    if peg_geom is not None:
        peg_geom.set("contype", "0")
        peg_geom.set("conaffinity", "0")
    path_nc = path.with_name(path.stem + "_nocolpeg.xml")
    tree.write(path_nc)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    gscene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scene.dt, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(dt=scene.dt),
        fem_options=gs.options.FEMOptions(
            dt=scene.dt,
            damping=20.0,
            use_implicit_solver=True,
            enable_vertex_constraints=True,
            n_newton_iterations=5,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=True),
        show_viewer=False,
    )
    robot = gscene.add_entity(gs.morphs.MJCF(file=str(path_nc)))
    peg = gscene.add_entity(
        morph=gs.morphs.Cylinder(
            height=PEG_LENGTH, radius=PEG_R, pos=(0.0, 0.0, 1.0),
            force_retet=True, visualization=True, collision=True),
        material=gs.materials.FEM.Elastic(
            E=1e6, nu=0.3, rho=1000.0, friction_mu=0.5, hydroelastic_modulus=1e7),
        name="fem_peg",
    )
    gscene.build()

    joints = {j.name: j for j in robot.joints}
    q = robot.get_qpos().detach().cpu().numpy().astype(float).copy()
    for name, value in scene.joint_init.items():
        q[joints[name].qs_idx_local[0]] = value
    robot.set_qpos(q)
    gscene.step()

    hand = next(link for link in robot.links if link.name == "hand")
    hand_pos = hand.get_pos().detach().cpu().numpy().astype(float)
    hand_quat = hand.get_quat().detach().cpu().numpy().astype(float)
    R_hand = quat_to_mat(hand_quat)
    R_mount = peg_mount(case)
    R_peg = R_hand @ R_mount
    peg_com = hand_pos + R_hand @ np.array([0.0, 0.0, PEG_MOUNT])
    tip_dir = R_peg @ np.array([0.0, 0.0, 1.0])

    # Place FEM verts: rest cylinder along +z about its COM, then R_peg + peg_com.
    rest = peg.init_positions.detach().cpu().numpy()
    if rest.ndim == 3:
        rest = rest[0]
    rest_com = rest.mean(axis=0)
    local = rest - rest_com
    placed = peg_com + (R_peg @ local.T).T
    peg.set_position(placed)
    peg.set_velocity(np.zeros(3))

    arr = peg.get_state().pos.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    print(f"hand={hand_pos.round(4)} peg_com={peg_com.round(4)} "
          f"peg_mean={arr.mean(0).round(4)}")
    proj = (arr - peg_com) @ tip_dir
    mount = np.argsort(proj)[:48].tolist()  # large mount patch (thin ring inverts at E=1e6)
    print(f"n_vertices={peg.n_vertices} mount={len(mount)} "
          f"proj={proj[mount].min():.4f}..{proj[mount].max():.4f}")
    peg.set_vertex_constraints(mount, link=hand, is_soft_constraint=False)

    dofs = [joints[n].dofs_idx_local[0] for n in scene.joints]
    qs = [joints[n].qs_idx_local[0] for n in scene.joints]
    ok = True
    for k in range(200):
        t = k * scene.dt
        qpos = robot.get_qpos().detach().cpu().numpy().astype(float)
        qd = robot.get_dofs_velocity().detach().cpu().numpy().astype(float)
        if not np.isfinite(qpos).all():
            print(f"nonfinite q at t={t:.3f}")
            ok = False
            break
        tau = scene.effort_at(t, qpos[qs], qd[dofs])
        force = torch.zeros(robot.n_dofs, dtype=gs.tc_float, device=gs.device)
        force[dofs] = torch.as_tensor(tau, dtype=gs.tc_float, device=gs.device)
        robot.control_dofs_force(force)
        try:
            gscene.step()
        except Exception as exc:
            print(f"FAIL at step {k}: {type(exc).__name__}: {exc}")
            ok = False
            break
        if k % 50 == 0:
            peg_z = peg.get_state().pos.detach().cpu().numpy()
            if peg_z.ndim == 3:
                peg_z = peg_z[0]
            mean = peg_z.mean(axis=0)
            hz = float(hand.get_pos().detach().cpu().numpy()[2])
            if not np.isfinite(mean).all() or np.linalg.norm(mean - peg_com) > 0.5:
                print(f"peg unstable at t={t:.3f}: mean={mean}")
                ok = False
                break
            print(f"  t={t:.3f}s hand_z={hz:.4f} peg_mean_z={mean[2]:.4f}")

    print("SPIKE_OK" if ok else "SPIKE_FAIL")
    print("ATTACHMENT=implicit_FEM+hard_vertex_constraints+LegacyCoupler.rigid_fem")
    print("FALLBACK=kinematic_mount_vertex_drive_from_hand_pose")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
