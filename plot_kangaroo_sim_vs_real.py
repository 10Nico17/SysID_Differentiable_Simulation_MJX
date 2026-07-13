"""Compare MJX rollout against a converted Kangaroo real SysID NPZ.

Usage (from mjx_sysid-main/):
    python scripts/Kangaroo/plot_kangaroo_sim_vs_real.py
    python scripts/Kangaroo/plot_kangaroo_sim_vs_real.py scripts/Kangaroo/datasets/Arms_sysid.npz --all-arm-joints --split-joints

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
PLOT_DIR = DATASET_DIR / "plots_sim_vs_real"
DEFAULT_NPZ = (
    DATASET_DIR / "left_elbow_chirp_20260530_081011_sysid.npz"
)
DEFAULT_XML = KANGAROO_DIR / "Robot/kangaroo_grippers_mjx.xml"

JOINT_NAME = "arm_left_4_joint"
ACT_IDX = 19
QPOS_IDX = 12
QVEL_IDX = 11
LEFT_ARM_JOINTS = tuple(f"arm_left_{i}_joint" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"arm_right_{i}_joint" for i in range(1, 8))
LEFT_LEG_JOINTS = (
    "leg_left_1_joint",
    "leg_left_2_joint",
    "leg_left_3_joint",
    "leg_left_length_joint",
    "leg_left_4_joint",
    "leg_left_5_joint",
)
RIGHT_LEG_JOINTS = (
    "leg_right_1_joint",
    "leg_right_2_joint",
    "leg_right_3_joint",
    "leg_right_length_joint",
    "leg_right_4_joint",
    "leg_right_5_joint",
)


def _arm_joint_names(side: str) -> tuple[str, ...]:
    if side == "left":
        return LEFT_ARM_JOINTS
    if side == "right":
        return RIGHT_ARM_JOINTS
    return LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


def _leg_joint_names(side: str) -> tuple[str, ...]:
    if side == "left":
        return LEFT_LEG_JOINTS
    if side == "right":
        return RIGHT_LEG_JOINTS
    return LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS


def _drive_prefixes(drive: str) -> tuple[str, ...]:
    if drive == "right-leg":
        return ("leg_right_",)
    if drive == "left-leg":
        return ("leg_left_",)
    if drive == "legs":
        return ("leg_left_", "leg_right_")
    return ()


def _hold_prefixes_for_drive(drive: str) -> tuple[str, ...]:
    if drive == "right-leg":
        return ("leg_left_", "pelvis_", "arm_left_", "arm_right_")
    if drive == "left-leg":
        return ("leg_right_", "pelvis_", "arm_left_", "arm_right_")
    if drive == "legs":
        return ("pelvis_", "arm_left_", "arm_right_")
    return ()


def _joint_indices_for_prefixes(
    model: mujoco.MjModel,
    prefixes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    qpos_idx = []
    qvel_idx = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name or not joint_name.startswith(prefixes):
            continue
        joint_type = model.jnt_type[joint_id]
        qadr = int(model.jnt_qposadr[joint_id])
        vadr = int(model.jnt_dofadr[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_idx.extend(range(qadr, qadr + 7))
            qvel_idx.extend(range(vadr, vadr + 6))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            qpos_idx.extend(range(qadr, qadr + 4))
            qvel_idx.extend(range(vadr, vadr + 3))
        else:
            qpos_idx.append(qadr)
            qvel_idx.append(vadr)
    return np.asarray(qpos_idx, dtype=np.int32), np.asarray(qvel_idx, dtype=np.int32)


def _actuator_ids_for_prefixes(
    model: mujoco.MjModel,
    prefixes: tuple[str, ...],
) -> list[tuple[int, int]]:
    actuator_slots = []
    for act_id in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        if not act_name or not act_name.startswith(prefixes):
            continue
        joint_id = int(model.actuator_trnid[act_id, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        actuator_slots.append((act_id, qadr))
    return actuator_slots


def _pin_base(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def _track_base(data: mjx.Data, qpos_ref: jnp.ndarray, qvel_ref: jnp.ndarray) -> mjx.Data:
    qpos = data.qpos.at[0:7].set(qpos_ref[0:7])
    qvel = data.qvel.at[0:6].set(qvel_ref[0:6])
    return data.replace(qpos=qpos, qvel=qvel)


def _rollout(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    pin_base: bool,
    track_base: bool,
    qpos_track: np.ndarray | None = None,
    qvel_track: np.ndarray | None = None,
    hold_qpos_idx: np.ndarray | None = None,
    hold_qvel_idx: np.ndarray | None = None,
    qpos_hold: np.ndarray | None = None,
    qvel_hold: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mjx_model = mjx.put_model(model)
    data = mjx.make_data(model).replace(
        qpos=jnp.asarray(qpos0),
        qvel=jnp.asarray(qvel0),
    )
    if pin_base:
        data = _pin_base(data)
    if hold_qpos_idx is not None and hold_qvel_idx is not None:
        hold_qpos_idx_j = jnp.asarray(hold_qpos_idx)
        hold_qvel_idx_j = jnp.asarray(hold_qvel_idx)
        qpos_hold_j = jnp.asarray(qpos_hold)
        qvel_hold_j = jnp.asarray(qvel_hold)
        data = data.replace(
            qpos=data.qpos.at[hold_qpos_idx_j].set(qpos_hold_j[hold_qpos_idx_j]),
            qvel=data.qvel.at[hold_qvel_idx_j].set(qvel_hold_j[hold_qvel_idx_j]),
        )
    else:
        hold_qpos_idx_j = jnp.asarray([], dtype=jnp.int32)
        hold_qvel_idx_j = jnp.asarray([], dtype=jnp.int32)
        qpos_hold_j = jnp.asarray(qpos0)
        qvel_hold_j = jnp.asarray(qvel0)

    @jax.jit
    def step_once(
        data_in: mjx.Data,
        ctrl_t: jnp.ndarray,
        qpos_ref: jnp.ndarray,
        qvel_ref: jnp.ndarray,
    ) -> mjx.Data:
        data_out = data_in.replace(ctrl=ctrl_t)
        data_out = data_out.replace(
            qpos=data_out.qpos.at[hold_qpos_idx_j].set(qpos_hold_j[hold_qpos_idx_j]),
            qvel=data_out.qvel.at[hold_qvel_idx_j].set(qvel_hold_j[hold_qvel_idx_j]),
        )
        data_out = mjx.step(mjx_model, data_out)
        data_out = data_out.replace(
            qpos=data_out.qpos.at[hold_qpos_idx_j].set(qpos_hold_j[hold_qpos_idx_j]),
            qvel=data_out.qvel.at[hold_qvel_idx_j].set(qvel_hold_j[hold_qvel_idx_j]),
        )
        if pin_base:
            data_out = _pin_base(data_out)
        if track_base:
            data_out = _track_base(data_out, qpos_ref, qvel_ref)
        return data_out

    qpos_sim = np.zeros((len(ctrl), model.nq), dtype=np.float32)
    qvel_sim = np.zeros((len(ctrl), model.nv), dtype=np.float32)

    for k, ctrl_t in enumerate(ctrl):
        # Match dataset convention: state[k] is before applying ctrl[k].
        qpos_sim[k] = np.asarray(data.qpos)
        qvel_sim[k] = np.asarray(data.qvel)
        if track_base:
            data = step_once(
                data,
                jnp.asarray(ctrl_t),
                jnp.asarray(qpos_track[k]),
                jnp.asarray(qvel_track[k]),
            )
        else:
            data = step_once(data, jnp.asarray(ctrl_t), data.qpos, data.qvel)

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


def _joint_indices(model: mujoco.MjModel, joint_name: str) -> tuple[int, int, int, int]:
    joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    act_idx = _name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    qpos_idx = int(model.jnt_qposadr[joint_id])
    qvel_idx = int(model.jnt_dofadr[joint_id])
    dof_idx = int(model.jnt_dofadr[joint_id])
    return act_idx, qpos_idx, qvel_idx, dof_idx


def _compare_metrics(
    q_sim: np.ndarray,
    q_real: np.ndarray,
    dq_sim: np.ndarray,
    dq_real: np.ndarray,
) -> dict[str, float]:
    err = q_sim - q_real
    dq_err = dq_sim - dq_real
    q_sim_centered = q_sim - np.mean(q_sim)
    q_real_centered = q_real - np.mean(q_real)
    corr = (
        float(np.corrcoef(q_sim_centered, q_real_centered)[0, 1])
        if np.std(q_sim_centered) > 0 and np.std(q_real_centered) > 0
        else np.nan
    )
    return {
        "err_mean_abs": float(np.mean(np.abs(err))),
        "err_rms": float(np.sqrt(np.mean(err**2))),
        "err_p95": float(np.percentile(np.abs(err), 95)),
        "err_max": float(np.max(np.abs(err))),
        "dq_err_mean_abs": float(np.mean(np.abs(dq_err))),
        "dq_err_rms": float(np.sqrt(np.mean(dq_err**2))),
        "dq_err_p95": float(np.percentile(np.abs(dq_err), 95)),
        "dq_err_max": float(np.max(np.abs(dq_err))),
        "corr": corr,
    }


def _plot_one_compare(
    *,
    time: np.ndarray,
    cmd: np.ndarray,
    q_real: np.ndarray,
    q_sim: np.ndarray,
    dq_real: np.ndarray,
    dq_sim: np.ndarray,
    joint_name: str,
    out: Path,
    title_prefix: str,
) -> dict[str, float]:
    metrics = _compare_metrics(q_sim, q_real, dq_sim, dq_real)
    err = q_sim - q_real
    dq_err = dq_sim - dq_real

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{title_prefix} - MJX sim vs real - {joint_name}")

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
        label=f"q_sim - q_real, rms={metrics['err_rms']:.3f}, max={metrics['err_max']:.3f}",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("q error [rad]")
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
        dq_err,
        color="purple",
        linewidth=0.8,
        label=f"dq_sim - dq_real, rms={metrics['dq_err_rms']:.3f}, max={metrics['dq_err_max']:.3f}",
    )
    axes[3].axhline(0.0, color="black", linewidth=0.5)
    axes[3].set_ylabel("dq error [rad/s]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return metrics


def _base_mode(pin_base: bool, free_base: bool, track_base: bool) -> str:
    if pin_base:
        return "pinned"
    if free_base:
        return "free"
    if track_base:
        return "tracked from dataset"
    return "not pinned/not tracked"


def _plot_joint_set(
    *,
    model: mujoco.MjModel,
    npz_path: Path,
    xml_path: Path,
    time: np.ndarray,
    ctrl: np.ndarray,
    qpos_real: np.ndarray,
    qvel_real: np.ndarray,
    qpos_sim: np.ndarray,
    qvel_sim: np.ndarray,
    joint_names: tuple[str, ...],
    out_dir: Path,
    label: str,
    title_prefix: str,
    no_gravity: bool,
    disable_constraints: bool,
    pin_base: bool,
    free_base: bool,
    track_base: bool,
    drive: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMJX sim vs real SysID NPZ - {label}")
    print(f"  file:          {npz_path}")
    print(f"  XML:           {xml_path}")
    print(f"  samples:       {len(time)}")
    print(f"  gravity:       {'off' if no_gravity else 'on'}")
    print(f"  constraints:   {'off' if disable_constraints else 'on'}")
    print(f"  base:          {_base_mode(pin_base, free_base, track_base)}")
    print(f"  drive:         {drive}")
    print("  joint                       act  qpos  qvel   q_rms     q_max     dq_rms    corr")

    for joint_name in joint_names:
        act_idx, qpos_idx, qvel_idx, _ = _joint_indices(model, joint_name)
        out_path = out_dir / f"{npz_path.stem}.{joint_name}.sim_vs_real.png"
        metrics = _plot_one_compare(
            time=time,
            cmd=ctrl[:, act_idx],
            q_real=qpos_real[:, qpos_idx],
            q_sim=qpos_sim[:, qpos_idx],
            dq_real=qvel_real[:, qvel_idx],
            dq_sim=qvel_sim[:, qvel_idx],
            joint_name=joint_name,
            out=out_path,
            title_prefix=title_prefix,
        )
        print(
            f"  {joint_name:<27} {act_idx:>3d}  {qpos_idx:>4d}  {qvel_idx:>4d}   "
            f"{metrics['err_rms']: .6f}  {metrics['err_max']: .6f}  "
            f"{metrics['dq_err_rms']: .6f}  {metrics['corr']: .4f}"
        )
        print(f"Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--act-idx", type=int, default=ACT_IDX)
    parser.add_argument("--qpos-idx", type=int, default=QPOS_IDX)
    parser.add_argument("--qvel-idx", type=int, default=QVEL_IDX)
    parser.add_argument("--all-arm-joints", action="store_true")
    parser.add_argument("--all-leg-joints", action="store_true")
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--leg-side", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--drive",
        choices=("all", "right-leg", "left-leg", "legs"),
        default="all",
        help="Which ctrl targets to simulate. Non-driven joints are held at the first state.",
    )
    parser.add_argument(
        "--split-joints",
        action="store_true",
        help="With --all-arm-joints, save one PNG per arm joint.",
    )
    parser.add_argument("--armature", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--frictionloss", type=float, default=None)
    parser.add_argument("--no-gravity", action="store_true", help="Debug: disable XML gravity.")
    parser.add_argument("--disable-constraints", action="store_true", help="Debug: disable equality and contact.")
    parser.add_argument("--free-base", action="store_true", help="Let the floating base evolve freely; disables base tracking.")
    parser.add_argument("--pin-base", action="store_true", help="Debug: pin floating base to a fixed pose.")
    parser.add_argument(
        "--track-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each MJX step, overwrite floating-base qpos/qvel from the dataset.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stop-step", type=int, default=None)
    args = parser.parse_args()
    if args.pin_base and args.free_base:
        raise ValueError("Use either --pin-base or --free-base, not both.")

    npz_path = args.npz.resolve()
    xml_path = args.xml.resolve()
    raw = np.load(npz_path, allow_pickle=True)

    time = raw["time"].astype(float)
    ctrl = raw["ctrl"].astype(float)
    qpos_real = raw["qpos"].astype(float)
    qvel_real = raw["qvel"].astype(float)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(time)))

    start_step = max(0, min(args.start_step, len(time) - 1))
    stop_step = len(time) if args.stop_step is None else max(start_step + 1, min(args.stop_step, len(time)))
    time = time[start_step:stop_step]
    ctrl = ctrl[start_step:stop_step]
    qpos_real = qpos_real[start_step:stop_step]
    qvel_real = qvel_real[start_step:stop_step]
    time = time - time[0]

    if args.max_steps is not None:
        n = min(args.max_steps, len(time))
        time = time[:n]
        ctrl = ctrl[:n]
        qpos_real = qpos_real[:n]
        qvel_real = qvel_real[:n]
    plot_stem = npz_path.stem
    if start_step != 0 or stop_step != len(raw["time"]):
        plot_stem = f"{plot_stem}_steps{start_step}_{start_step + len(time)}"
    elif args.max_steps is not None:
        plot_stem = f"{plot_stem}_first{len(time)}"
    qpos_track = qpos_real[1 : len(ctrl) + 1]
    qvel_track = qvel_real[1 : len(ctrl) + 1]
    if len(qpos_track) < len(ctrl):
        qpos_track = np.concatenate([qpos_track, qpos_real[-1:]], axis=0)
        qvel_track = np.concatenate([qvel_track, qvel_real[-1:]], axis=0)
    track_base = args.track_base and not args.free_base and not args.pin_base

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

    hold_prefixes = _hold_prefixes_for_drive(args.drive)
    hold_qpos_idx, hold_qvel_idx = _joint_indices_for_prefixes(model, hold_prefixes)
    qpos_hold = qpos_real[0].copy()
    qvel_hold = qvel_real[0].copy()
    if hold_prefixes:
        ctrl = ctrl.copy()
        for act_id, qpos_idx in _actuator_ids_for_prefixes(model, hold_prefixes):
            ctrl[:, act_id] = qpos_hold[qpos_idx]

    qpos_sim, qvel_sim = _rollout(
        model=model,
        qpos0=qpos_real[0],
        qvel0=qvel_real[0],
        ctrl=ctrl,
        pin_base=args.pin_base,
        track_base=track_base,
        qpos_track=qpos_track,
        qvel_track=qvel_track,
        hold_qpos_idx=hold_qpos_idx if hold_prefixes else None,
        hold_qvel_idx=hold_qvel_idx if hold_prefixes else None,
        qpos_hold=qpos_hold,
        qvel_hold=qvel_hold,
    )

    title_prefix = "MJX sim vs real"
    if args.armature is None and args.damping is None and args.frictionloss is None:
        title_prefix = "Vor Optimierung"

    if args.all_arm_joints:
        _plot_joint_set(
            model=model,
            npz_path=npz_path,
            xml_path=xml_path,
            time=time,
            ctrl=ctrl,
            qpos_real=qpos_real,
            qvel_real=qvel_real,
            qpos_sim=qpos_sim,
            qvel_sim=qvel_sim,
            joint_names=_arm_joint_names(args.arm_side),
            out_dir=PLOT_DIR / plot_stem / "arms",
            label=f"arm joints ({args.arm_side})",
            title_prefix=title_prefix,
            no_gravity=args.no_gravity,
            disable_constraints=args.disable_constraints,
            pin_base=args.pin_base,
            free_base=args.free_base,
            track_base=track_base,
            drive=args.drive,
        )
        return

    if args.all_leg_joints:
        _plot_joint_set(
            model=model,
            npz_path=npz_path,
            xml_path=xml_path,
            time=time,
            ctrl=ctrl,
            qpos_real=qpos_real,
            qvel_real=qvel_real,
            qpos_sim=qpos_sim,
            qvel_sim=qvel_sim,
            joint_names=_leg_joint_names(args.leg_side),
            out_dir=PLOT_DIR / plot_stem / "legs",
            label=f"leg joints ({args.leg_side})",
            title_prefix=title_prefix,
            no_gravity=args.no_gravity,
            disable_constraints=args.disable_constraints,
            pin_base=args.pin_base,
            free_base=args.free_base,
            track_base=track_base,
            drive=args.drive,
        )
        return

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
    print(f"  base:          {_base_mode(args.pin_base, args.free_base, track_base)}")
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
    out = PLOT_DIR / f"{plot_stem}.sim_vs_real.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
