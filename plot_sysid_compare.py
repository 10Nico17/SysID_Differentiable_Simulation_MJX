"""Compare real robot, old XML params, and new optimized params in one plot.

Run from scripts/Kangaroo/:
    python plot_sysid_compare.py \
        --npz datasets/left_elbow_chirp_20260530_081011_sysid.npz \
        --armature 0.00438062 \
        --damping 0.0 \
        --frictionloss 0.0 \
        --mass 3.71568513 \
        --inertia 0.00220383 0.00164147 0.00098735 \
        --ipos -0.65889198 0.40903932 0.28787863
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx


KANGAROO_DIR = Path(__file__).resolve().parent
ROOT = KANGAROO_DIR.parents[1]
DATASET_DIR = KANGAROO_DIR / "datasets"
DEFAULT_NPZ = DATASET_DIR / "left_elbow_chirp_20260530_081011_sysid.npz"
DEFAULT_XML = KANGAROO_DIR / "Robot/kangaroo_grippers_mjx.xml"
JOINT_NAME  = "arm_left_4_joint"


###############################################################################
# Rollout
###############################################################################


def _pin_base(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def rollout(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    pin_base: bool,
) -> np.ndarray:
    mjx_model = mjx.put_model(model)
    data = mjx.make_data(model).replace(
        qpos=jnp.asarray(qpos0),
        qvel=jnp.asarray(qvel0),
    )
    if pin_base:
        data = _pin_base(data)

    @jax.jit
    def step_once(d: mjx.Data, ctrl_t: jnp.ndarray) -> mjx.Data:
        d = d.replace(ctrl=ctrl_t)
        d = mjx.step(mjx_model, d)
        if pin_base:
            d = _pin_base(d)
        return d

    qpos_out = np.zeros((len(ctrl), model.nq), dtype=np.float32)
    for k, ctrl_t in enumerate(ctrl):
        qpos_out[k] = np.asarray(data.qpos)
        data = step_once(data, jnp.asarray(ctrl_t))
    return qpos_out


###############################################################################
# Model helpers
###############################################################################


def name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"{name!r} not found in model.")
    return int(idx)


def apply_new_params(model: mujoco.MjModel, args: argparse.Namespace, joint_name: str) -> None:
    joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    dof_idx  = int(model.jnt_dofadr[joint_id])
    body_idx = int(model.jnt_bodyid[joint_id])

    if args.armature     is not None: model.dof_armature[dof_idx]    = args.armature
    if args.damping      is not None: model.dof_damping[dof_idx]     = args.damping
    if args.frictionloss is not None: model.dof_frictionloss[dof_idx]= args.frictionloss
    if args.mass         is not None: model.body_mass[body_idx]       = args.mass
    if args.inertia      is not None: model.body_inertia[body_idx]    = args.inertia
    if args.ipos         is not None: model.body_ipos[body_idx]       = args.ipos


###############################################################################
# Stats
###############################################################################


def stats(q_sim: np.ndarray, q_real: np.ndarray) -> dict[str, float]:
    err = q_sim - q_real
    return {
        "rms":  float(np.sqrt(np.mean(err**2))),
        "mae":  float(np.mean(np.abs(err))),
        "max":  float(np.max(np.abs(err))),
        "corr": float(np.corrcoef(q_sim - q_sim.mean(), q_real - q_real.mean())[0, 1]),
    }


###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz",  type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml",  type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--free-base", action="store_true")
    parser.add_argument("--no-gravity", action="store_true")
    parser.add_argument("--disable-constraints", action="store_true")
    parser.add_argument("--out", type=Path, default=None)

    # New optimized parameters (all optional — omit to keep XML value)
    parser.add_argument("--armature",     type=float, default=None)
    parser.add_argument("--damping",      type=float, default=None)
    parser.add_argument("--frictionloss", type=float, default=None)
    parser.add_argument("--mass",         type=float, default=None)
    parser.add_argument("--inertia",      type=float, nargs=3, default=None,
                        metavar=("Ixx", "Iyy", "Izz"))
    parser.add_argument("--ipos",         type=float, nargs=3, default=None,
                        metavar=("x", "y", "z"))
    return parser.parse_args()


###############################################################################
# Main
###############################################################################


def main() -> None:
    args = parse_args()

    raw = np.load(args.npz.resolve(), allow_pickle=True)
    time     = raw["time"].astype(np.float32)
    ctrl     = raw["ctrl"].astype(np.float32)
    qpos_real= raw["qpos"].astype(np.float32)
    qvel_real= raw["qvel"].astype(np.float32)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(time)))

    if args.max_steps is not None:
        n = min(args.max_steps, len(time))
        time, ctrl, qpos_real, qvel_real = time[:n], ctrl[:n], qpos_real[:n], qvel_real[:n]

    def make_model() -> mujoco.MjModel:
        m = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
        m.opt.timestep = dt
        if args.no_gravity:
            m.opt.gravity[:] = 0.0
        if args.disable_constraints:
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        return m

    joint_id = name_to_id(make_model(), mujoco.mjtObj.mjOBJ_JOINT, args.joint)
    qpos_idx = int(make_model().jnt_qposadr[joint_id])
    act_idx  = name_to_id(make_model(), mujoco.mjtObj.mjOBJ_ACTUATOR, args.joint)

    # --- old model (XML params) ---
    model_old = make_model()
    print("Rolling out old (XML) model...")
    qpos_old = rollout(model_old, qpos_real[0], qvel_real[0], ctrl, not args.free_base)

    # --- new model (optimized params) ---
    model_new = make_model()
    apply_new_params(model_new, args, args.joint)
    print("Rolling out new (optimized) model...")
    qpos_new = rollout(model_new, qpos_real[0], qvel_real[0], ctrl, not args.free_base)

    q_real = qpos_real[:, qpos_idx]
    q_old  = qpos_old[:, qpos_idx]
    q_new  = qpos_new[:, qpos_idx]
    cmd    = ctrl[:, act_idx]

    s_old = stats(q_old, q_real)
    s_new = stats(q_new, q_real)

    print(f"\n{'':20s}  {'RMS':>10}  {'MAE':>10}  {'MAX':>10}  {'corr':>8}")
    print(f"  {'old (XML)':20s}  {s_old['rms']:10.6f}  {s_old['mae']:10.6f}  {s_old['max']:10.6f}  {s_old['corr']:8.4f}")
    print(f"  {'new (optimized)':20s}  {s_new['rms']:10.6f}  {s_new['mae']:10.6f}  {s_new['max']:10.6f}  {s_new['corr']:8.4f}")
    print(f"\n  ΔRMS: {s_old['rms'] - s_new['rms']:+.6f} rad  ({'better' if s_new['rms'] < s_old['rms'] else 'worse'})")

    ###########################################################################
    # Plot
    ###########################################################################

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(f"SysID comparison — {args.joint}", fontsize=13)

    # --- q position ---
    axes[0].plot(time, cmd,    color="gray",   lw=0.7, alpha=0.6, label="ctrl")
    axes[0].plot(time, q_real, color="black",  lw=1.2, label="real")
    axes[0].plot(time, q_old,  color="tomato", lw=0.9, alpha=0.85,
                 label=f"sim old  (rms={s_old['rms']:.4f})")
    axes[0].plot(time, q_new,  color="steelblue", lw=0.9, alpha=0.85,
                 label=f"sim new  (rms={s_new['rms']:.4f})")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- error ---
    axes[1].plot(time, q_old - q_real, color="tomato",   lw=0.8,
                 label=f"old error  rms={s_old['rms']:.4f}  max={s_old['max']:.4f}")
    axes[1].plot(time, q_new - q_real, color="steelblue", lw=0.8,
                 label=f"new error  rms={s_new['rms']:.4f}  max={s_new['max']:.4f}")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("q_sim − q_real [rad]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # --- |error| improvement ---
    improvement = np.abs(q_old - q_real) - np.abs(q_new - q_real)
    axes[2].fill_between(time, improvement, 0,
                         where=improvement >= 0, color="steelblue", alpha=0.4,
                         label="new better")
    axes[2].fill_between(time, improvement, 0,
                         where=improvement < 0,  color="tomato",   alpha=0.4,
                         label="old better")
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set_ylabel("|err_old| − |err_new| [rad]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.out if args.out is not None else DATASET_DIR / f"{args.npz.stem}.compare.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
