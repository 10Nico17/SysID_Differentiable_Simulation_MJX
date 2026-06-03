"""Compare MJX rollout against a converted Kangaroo real SysID NPZ.

Usage (from mjx_sysid-main/):
    python scripts/Kangaroo/plot_kangaroo_sim_vs_real.py
    python scripts/Kangaroo/plot_kangaroo_sim_vs_real.py assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz

The script replays ctrl from the NPZ in MJX and plots q_sim vs q_real.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx


KANGAROO_DIR = Path(__file__).resolve().parent
ROOT = KANGAROO_DIR.parents[1]
DATASET_DIR = KANGAROO_DIR / "datasets"
DEFAULT_NPZ = (
    DATASET_DIR / "left_elbow_chirp_20260530_081011_sysid.npz"
)
DEFAULT_XML = KANGAROO_DIR / "Robot/kangaroo_grippers_mjx.xml"

JOINT_NAME = "arm_left_4_joint"
ACT_IDX = 19
QPOS_IDX = 12
QVEL_IDX = 11


def _pin_base(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def _rollout(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    pin_base: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mjx_model = mjx.put_model(model)
    data = mjx.make_data(model).replace(
        qpos=jnp.asarray(qpos0),
        qvel=jnp.asarray(qvel0),
    )
    if pin_base:
        data = _pin_base(data)

    @jax.jit
    def step_once(data_in: mjx.Data, ctrl_t: jnp.ndarray) -> mjx.Data:
        data_out = data_in.replace(ctrl=ctrl_t)
        data_out = mjx.step(mjx_model, data_out)
        if pin_base:
            data_out = _pin_base(data_out)
        return data_out

    qpos_sim = np.zeros((len(ctrl), model.nq), dtype=np.float32)
    qvel_sim = np.zeros((len(ctrl), model.nv), dtype=np.float32)

    for k, ctrl_t in enumerate(ctrl):
        # Match dataset convention: state[k] is before applying ctrl[k].
        qpos_sim[k] = np.asarray(data.qpos)
        qvel_sim[k] = np.asarray(data.qvel)
        data = step_once(data, jnp.asarray(ctrl_t))

    return qpos_sim, qvel_sim


def _name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Could not find {name!r} in model.")
    return int(idx)


def _apply_joint_param_overrides(
    model: mujoco.MjModel,
    joint_name: str,
    armature: float | None,
    damping: float | None,
    frictionloss: float | None,
) -> int:
    joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    dof_idx = int(model.jnt_dofadr[joint_id])
    if armature is not None:
        model.dof_armature[dof_idx] = armature
    if damping is not None:
        model.dof_damping[dof_idx] = damping
    if frictionloss is not None:
        model.dof_frictionloss[dof_idx] = frictionloss
    return dof_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--act-idx", type=int, default=ACT_IDX)
    parser.add_argument("--qpos-idx", type=int, default=QPOS_IDX)
    parser.add_argument("--qvel-idx", type=int, default=QVEL_IDX)
    parser.add_argument("--armature", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--frictionloss", type=float, default=None)
    parser.add_argument("--no-gravity", action="store_true", help="Debug: disable XML gravity.")
    parser.add_argument("--disable-constraints", action="store_true", help="Debug: disable equality and contact.")
    parser.add_argument("--free-base", action="store_true", help="Do not pin the floating base.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    xml_path = args.xml.resolve()
    raw = np.load(npz_path, allow_pickle=True)

    time = raw["time"].astype(float)
    ctrl = raw["ctrl"].astype(float)
    qpos_real = raw["qpos"].astype(float)
    qvel_real = raw["qvel"].astype(float)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(time)))

    if args.max_steps is not None:
        n = min(args.max_steps, len(time))
        time = time[:n]
        ctrl = ctrl[:n]
        qpos_real = qpos_real[:n]
        qvel_real = qvel_real[:n]

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = dt
    if args.no_gravity:
        model.opt.gravity[:] = 0.0
    if args.disable_constraints:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    dof_idx = _apply_joint_param_overrides(
        model=model,
        joint_name=args.joint,
        armature=args.armature,
        damping=args.damping,
        frictionloss=args.frictionloss,
    )

    qpos_sim, qvel_sim = _rollout(
        model=model,
        qpos0=qpos_real[0],
        qvel0=qvel_real[0],
        ctrl=ctrl,
        pin_base=not args.free_base,
    )

    q_real = qpos_real[:, args.qpos_idx]
    dq_real = qvel_real[:, args.qvel_idx]
    q_sim = qpos_sim[:, args.qpos_idx]
    dq_sim = qvel_sim[:, args.qvel_idx]
    cmd = ctrl[:, args.act_idx]

    err = q_sim - q_real
    dq_err = dq_sim - dq_real
    err_mean_abs = float(np.mean(np.abs(err)))
    err_rms = float(np.sqrt(np.mean(err**2)))
    err_p95 = float(np.percentile(np.abs(err), 95))
    err_max = float(np.max(np.abs(err)))
    dq_err_mean_abs = float(np.mean(np.abs(dq_err)))
    dq_err_rms = float(np.sqrt(np.mean(dq_err**2)))
    dq_err_p95 = float(np.percentile(np.abs(dq_err), 95))
    dq_err_max = float(np.max(np.abs(dq_err)))
    corr = float(np.corrcoef(q_sim - np.mean(q_sim), q_real - np.mean(q_real))[0, 1])

    print("\nMJX sim vs real SysID NPZ")
    print(f"  file:          {npz_path}")
    print(f"  XML:           {xml_path}")
    print(f"  joint:         {args.joint}")
    print(f"  indices:       act={args.act_idx}, qpos={args.qpos_idx}, qvel={args.qvel_idx}")
    print(f"  samples:       {len(time)}")
    print(f"  dt:            {dt:.6f} s")
    print(f"  gravity:       {'off' if args.no_gravity else 'on'}")
    print(f"  constraints:   {'off' if args.disable_constraints else 'on'}")
    print(f"  base:          {'free' if args.free_base else 'pinned'}")
    print(f"  armature:      {model.dof_armature[dof_idx]: .8f}")
    print(f"  damping:       {model.dof_damping[dof_idx]: .8f}")
    print(f"  frictionloss:  {model.dof_frictionloss[dof_idx]: .8f}")
    print(f"  q_real range:  {q_real.min(): .6f} .. {q_real.max(): .6f} rad")
    print(f"  q_sim range:   {q_sim.min(): .6f} .. {q_sim.max(): .6f} rad")
    print(f"  error mean abs:{err_mean_abs: .6f} rad")
    print(f"  error RMS:     {err_rms: .6f} rad")
    print(f"  error 95% abs: {err_p95: .6f} rad")
    print(f"  error max abs: {err_max: .6f} rad")
    print(f"  dq err mean abs:{dq_err_mean_abs: .6f} rad/s")
    print(f"  dq err RMS:    {dq_err_rms: .6f} rad/s")
    print(f"  dq err 95% abs:{dq_err_p95: .6f} rad/s")
    print(f"  dq err max abs:{dq_err_max: .6f} rad/s")
    print(f"  corr(sim,real):{corr: .4f}")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Vor Optimierung - MJX sim vs real - {args.joint} "
        f"({'no gravity' if args.no_gravity else 'gravity'}, "
        f"{'no constraints' if args.disable_constraints else 'constraints'})"
    )

    axes[0].plot(time, cmd, color="gray", linewidth=0.8, label="ctrl real")
    axes[0].plot(time, q_real, color="blue", linewidth=0.9, label="q real")
    axes[0].plot(time, q_sim, color="orange", linewidth=0.9, label="q sim")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        err,
        color="red",
        linewidth=0.8,
        label=f"q_sim - q_real, rms={err_rms:.3f}, max={err_max:.3f}",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("error [rad]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, dq_real, color="green", linewidth=0.8, label="dq real")
    axes[2].plot(time, dq_sim, color="purple", linewidth=0.8, label="dq sim")
    axes[2].fill_between(time, dq_real, dq_sim, color="purple", alpha=0.15)
    axes[2].set_ylabel("velocity [rad/s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(
        time,
        np.abs(dq_err),
        color="purple",
        linewidth=0.8,
        label=f"|dq_sim - dq_real|, rms={dq_err_rms:.3f}, max={dq_err_max:.3f}",
    )
    axes[3].set_ylabel("|dq error| [rad/s]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = (
        args.out
        if args.out is not None
        else DATASET_DIR / f"{npz_path.stem}.sim_vs_real.png"
    )
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
