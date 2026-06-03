"""Minimal Kangaroo elbow rollout with fixed parameters.

This script is the first clean building block for the custom Kangaroo SysID
pipeline:

1. Load XML.
2. Find arm_left_4_joint indices.
3. Load converted SysID NPZ.
4. Build MJX model/data.
5. Set one fixed parameter triplet in the MJX model.
6. Replay real ctrl commands for a short rollout.
6. Compare q_sim against q_real.

Run from repo root:
    python scripts/Kangaroo/01_rollout_fixed_params.py \
        --npz assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    ROOT
    / "assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz"
)
DEFAULT_XML = ROOT / "assets/robots/kangaroo_grippers/kangaroo_grippers_mjx.xml"
JOINT_NAME = "arm_left_4_joint"


def _name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Could not find {name!r} in model.")
    return int(idx)


def _pin_base(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def _apply_joint_params(
    model: mjx.Model,
    dof_idx: int,
    armature: float | None,
    damping: float | None,
    frictionloss: float | None,
) -> mjx.Model:
    dof_armature = model.dof_armature
    dof_damping = model.dof_damping
    dof_frictionloss = model.dof_frictionloss

    if armature is not None:
        dof_armature = dof_armature.at[dof_idx].set(armature)
    if damping is not None:
        dof_damping = dof_damping.at[dof_idx].set(damping)
    if frictionloss is not None:
        dof_frictionloss = dof_frictionloss.at[dof_idx].set(frictionloss)

    return model.replace(
        dof_armature=dof_armature,
        dof_damping=dof_damping,
        dof_frictionloss=dof_frictionloss,
    )


def _rollout(
    model: mjx.Model,
    data_template: mjx.Data,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    pin_base: bool,
) -> tuple[np.ndarray, np.ndarray]:
    qpos0_j = jnp.asarray(qpos0)
    qvel0_j = jnp.asarray(qvel0)
    ctrl_j = jnp.asarray(ctrl)

    def rollout_jax(q0: jnp.ndarray, v0: jnp.ndarray, ctrl_seq: jnp.ndarray):
        data0 = data_template.replace(qpos=q0, qvel=v0)
        if pin_base:
            data0 = _pin_base(data0)

        def step(carry: mjx.Data, ctrl_t: jnp.ndarray):
            data = carry
            # Dataset convention: qpos[k] is the state before ctrl[k] is applied.
            y = (data.qpos, data.qvel)
            data = data.replace(ctrl=ctrl_t)
            data = mjx.step(model, data)
            if pin_base:
                data = _pin_base(data)
            return data, y

        _, (qpos_sim, qvel_sim) = jax.lax.scan(step, data0, ctrl_seq)
        return qpos_sim, qvel_sim

    qpos_sim, qvel_sim = jax.jit(rollout_jax)(qpos0_j, qvel0_j, ctrl_j)
    return np.asarray(qpos_sim), np.asarray(qvel_sim)


def _print_stats(label: str, err: np.ndarray) -> None:
    print(f"  {label} mean abs: {np.mean(np.abs(err)): .8f}")
    print(f"  {label} RMS:      {np.sqrt(np.mean(err**2)): .8f}")
    print(f"  {label} 95% abs:  {np.percentile(np.abs(err), 95): .8f}")
    print(f"  {label} max abs:  {np.max(np.abs(err)): .8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--armature", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--frictionloss", type=float, default=None)
    parser.add_argument("--no-gravity", action="store_true", help="Debug: disable XML gravity.")
    parser.add_argument("--disable-constraints", action="store_true", help="Debug: disable equality and contact.")
    parser.add_argument("--free-base", action="store_true", help="Do not pin floating base.")
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    xml_path = args.xml.resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    raw = np.load(npz_path, allow_pickle=True)

    time = raw["time"].astype(float)
    ctrl = raw["ctrl"].astype(float)
    qpos_real = raw["qpos"].astype(float)
    qvel_real = raw["qvel"].astype(float)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(time)))

    model.opt.timestep = dt
    if args.no_gravity:
        model.opt.gravity[:] = 0.0
    if args.disable_constraints:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, args.joint)
    qpos_idx = int(model.jnt_qposadr[joint_id])
    dof_idx = int(model.jnt_dofadr[joint_id])
    qvel_idx = dof_idx

    actuator_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.joint)
    act_idx = actuator_id

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(model)
    mjx_model = _apply_joint_params(
        model=mjx_model,
        dof_idx=dof_idx,
        armature=args.armature,
        damping=args.damping,
        frictionloss=args.frictionloss,
    )

    if args.start < 0:
        raise ValueError("--start must be >= 0")
    end = min(args.start + args.horizon, len(ctrl))
    if end <= args.start:
        raise ValueError("Empty rollout window. Check --start and --horizon.")

    ctrl_win = ctrl[args.start:end]
    qpos0 = qpos_real[args.start]
    qvel0 = qvel_real[args.start]
    qpos_target = qpos_real[args.start:end]
    qvel_target = qvel_real[args.start:end]

    qpos_sim, qvel_sim = _rollout(
        model=mjx_model,
        data_template=mjx_data,
        qpos0=qpos0,
        qvel0=qvel0,
        ctrl=ctrl_win,
        pin_base=not args.free_base,
    )

    q_real = qpos_target[:, qpos_idx]
    dq_real = qvel_target[:, qvel_idx]
    q_sim = qpos_sim[:, qpos_idx]
    dq_sim = qvel_sim[:, qvel_idx]
    cmd = ctrl_win[:, act_idx]

    q_err = q_sim - q_real
    dq_err = dq_sim - dq_real
    corr = float(np.corrcoef(q_sim - q_sim.mean(), q_real - q_real.mean())[0, 1])

    print("\nKangaroo fixed-parameter rollout")
    print(f"  XML:          {xml_path}")
    print(f"  NPZ:          {npz_path}")
    print(f"  joint:        {args.joint}")
    print(f"  joint_id:     {joint_id}")
    print(f"  act_idx:      {act_idx}")
    print(f"  qpos_idx:     {qpos_idx}")
    print(f"  qvel_idx:     {qvel_idx}")
    print(f"  dof_idx:      {dof_idx}")
    print("")
    print("Dataset")
    print(f"  samples:      {len(time)}")
    print(f"  dt:           {dt:.6f} s")
    print(f"  ctrl shape:   {ctrl.shape}")
    print(f"  qpos shape:   {qpos_real.shape}")
    print(f"  qvel shape:   {qvel_real.shape}")
    print("")
    print("Simulation")
    print(f"  start:        {args.start}")
    print(f"  horizon:      {len(ctrl_win)}")
    print(f"  gravity:      {'off' if args.no_gravity else 'on'}")
    print(f"  constraints:  {'off' if args.disable_constraints else 'on'}")
    print(f"  base:         {'free' if args.free_base else 'pinned'}")
    print("")
    print("Joint parameters used")
    print(f"  armature:     {float(mjx_model.dof_armature[dof_idx]):.8f}")
    print(f"  damping:      {float(mjx_model.dof_damping[dof_idx]):.8f}")
    print(f"  frictionloss: {float(mjx_model.dof_frictionloss[dof_idx]):.8f}")
    print("")
    print("Signal ranges")
    print(f"  ctrl:         {cmd.min(): .8f} .. {cmd.max(): .8f}")
    print(f"  q_real:       {q_real.min(): .8f} .. {q_real.max(): .8f}")
    print(f"  q_sim:        {q_sim.min(): .8f} .. {q_sim.max(): .8f}")
    print(f"  dq_real:      {dq_real.min(): .8f} .. {dq_real.max(): .8f}")
    print(f"  dq_sim:       {dq_sim.min(): .8f} .. {dq_sim.max(): .8f}")
    print("")
    print("Comparison")
    _print_stats("q error", q_err)
    _print_stats("dq error", dq_err)
    print(f"  corr(q_sim, q_real): {corr: .6f}")


if __name__ == "__main__":
    main()
