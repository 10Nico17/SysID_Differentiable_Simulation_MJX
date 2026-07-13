"""Plot a converted Kangaroo SysID NPZ.

Usage (from mjx_sysid-main/):
    python scripts/Kangaroo/plot_kangaroo_sysid_npz.py
    python scripts/Kangaroo/plot_kangaroo_sysid_npz.py scripts/Kangaroo/datasets/Arms_sysid.npz --all-arm-joints --split-joints
    python scripts/Kangaroo/plot_kangaroo_sysid_npz.py scripts/Kangaroo/datasets/Arms_sysid.npz --all-joints

Expected NPZ keys:
    time, ctrl, qpos, qvel, actuator_names, dt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np


KANGAROO_DIR = Path(__file__).resolve().parent
ROOT = KANGAROO_DIR.parents[1]
DATASET_DIR = KANGAROO_DIR / "datasets"
PLOT_DIR = DATASET_DIR / "plots_sysid"
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


def _get_scalar_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    return str(value.item() if value.shape == () else value)


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


def _joint_group(joint_name: str) -> str:
    if joint_name.startswith("arm_left_"):
        return "arms/left"
    if joint_name.startswith("arm_right_"):
        return "arms/right"
    if joint_name.startswith("leg_left_"):
        return "legs/left"
    if joint_name.startswith("leg_right_"):
        return "legs/right"
    if joint_name.startswith("pelvis_"):
        return "pelvis"
    if joint_name.startswith("gripper_"):
        return "grippers"
    return "other"


def _actuated_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = []
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name:
            names.append(joint_name)
    return tuple(names)


def _name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Could not find {name!r} in model.")
    return int(idx)


def _joint_indices(model: mujoco.MjModel, joint_name: str) -> tuple[int, int, int]:
    joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    act_idx = _name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    qpos_idx = int(model.jnt_qposadr[joint_id])
    qvel_idx = int(model.jnt_dofadr[joint_id])
    return act_idx, qpos_idx, qvel_idx


def _metrics(ctrl: np.ndarray, q: np.ndarray, dt: float) -> dict[str, float]:
    err = ctrl - q
    ctrl_centered = ctrl - np.mean(ctrl)
    q_centered = q - np.mean(q)
    ctrl_rms = float(np.sqrt(np.mean(ctrl_centered**2)))
    q_rms = float(np.sqrt(np.mean(q_centered**2)))
    corr = (
        float(np.corrcoef(ctrl_centered, q_centered)[0, 1])
        if ctrl_rms > 0 and q_rms > 0
        else np.nan
    )
    stride = max(1, int(np.ceil(len(ctrl_centered) / 5000)))
    ctrl_lag = ctrl_centered[::stride]
    q_lag = q_centered[::stride]
    xcorr = np.correlate(q_lag, ctrl_lag, mode="full")
    lag_samples = int(np.argmax(xcorr) - (len(ctrl_lag) - 1)) * stride
    return {
        "err_mean_abs": float(np.mean(np.abs(err))),
        "err_rms": float(np.sqrt(np.mean(err**2))),
        "err_p95": float(np.percentile(np.abs(err), 95)),
        "err_max": float(np.max(np.abs(err))),
        "amp_ratio": q_rms / ctrl_rms if ctrl_rms > 0 else np.nan,
        "corr": corr,
        "lag_samples": lag_samples,
        "lag_seconds": lag_samples * dt,
    }


def _plot_one(
    *,
    time: np.ndarray,
    ctrl: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    joint_name: str,
    act_idx: int,
    qpos_idx: int,
    qvel_idx: int,
    dt: float,
    out: Path,
) -> dict[str, float]:
    m = _metrics(ctrl, q, dt)
    err = ctrl - q

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Converted SysID NPZ - {joint_name}")

    axes[0].plot(time, ctrl, color="gray", linewidth=0.9, label=f"ctrl[{act_idx}]")
    axes[0].plot(time, q, color="blue", linewidth=0.9, label=f"qpos[{qpos_idx}]")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        err,
        color="red",
        linewidth=0.8,
        label=f"ctrl - qpos, rms={m['err_rms']:.3f}, max={m['err_max']:.3f}",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("error [rad]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, dq, color="green", linewidth=0.8, label=f"qvel[{qvel_idx}]")
    axes[2].set_ylabel("velocity [rad/s]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return m


def _plot_joint_set(
    *,
    data: np.lib.npyio.NpzFile,
    model: mujoco.MjModel,
    npz_path: Path,
    xml_path: Path,
    time: np.ndarray,
    dt: float,
    joint_names: tuple[str, ...],
    out_root: Path,
    label: str,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nConverted SysID NPZ {label} check")
    print(f"  file:          {npz_path}")
    print(f"  XML:           {xml_path.resolve()}")
    print(f"  source:        {_get_scalar_str(data, 'source_npz', '-')}")
    print(f"  samples:       {len(time)}")
    print(f"  dt:            {dt:.6f} s")
    print(f"  out_dir:       {out_root}")
    print("  joint                       act  qpos  qvel   rms_err   max_err   corr")

    for joint_name in joint_names:
        act_idx, qpos_idx, qvel_idx = _joint_indices(model, joint_name)
        ctrl_j = data["ctrl"][:, act_idx].astype(float)
        q_j = data["qpos"][:, qpos_idx].astype(float)
        dq_j = data["qvel"][:, qvel_idx].astype(float)
        out_path = out_root / _joint_group(joint_name) / f"{joint_name}.sysid.png"
        m = _plot_one(
            time=time,
            ctrl=ctrl_j,
            q=q_j,
            dq=dq_j,
            joint_name=joint_name,
            act_idx=act_idx,
            qpos_idx=qpos_idx,
            qvel_idx=qvel_idx,
            dt=dt,
            out=out_path,
        )
        print(
            f"  {joint_name:<27} {act_idx:>3d}  {qpos_idx:>4d}  {qvel_idx:>4d}   "
            f"{m['err_rms']: .6f}  {m['err_max']: .6f}  {m['corr']: .4f}"
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
    parser.add_argument(
        "--all-joints",
        action="store_true",
        help="Plot every actuated model joint into grouped subfolders.",
    )
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--leg-side", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--split-joints",
        action="store_true",
        help="With --all-arm-joints, save one PNG per arm joint.",
    )
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    data = np.load(npz_path, allow_pickle=True)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))

    time = data["time"].astype(float)
    dt = float(data["dt"]) if "dt" in data.files else float(np.median(np.diff(time)))

    if args.all_joints:
        _plot_joint_set(
            data=data,
            model=model,
            npz_path=npz_path,
            xml_path=args.xml,
            time=time,
            dt=dt,
            joint_names=_actuated_joint_names(model),
            out_root=PLOT_DIR / npz_path.stem,
            label="all-joint",
        )
        return

    if args.all_arm_joints:
        _plot_joint_set(
            data=data,
            model=model,
            npz_path=npz_path,
            xml_path=args.xml,
            time=time,
            dt=dt,
            joint_names=_arm_joint_names(args.arm_side),
            out_root=PLOT_DIR / npz_path.stem,
            label=f"arm {args.arm_side}",
        )
        return

    if args.all_leg_joints:
        _plot_joint_set(
            data=data,
            model=model,
            npz_path=npz_path,
            xml_path=args.xml,
            time=time,
            dt=dt,
            joint_names=_leg_joint_names(args.leg_side),
            out_root=PLOT_DIR / npz_path.stem,
            label=f"leg {args.leg_side}",
        )
        return

    ctrl = data["ctrl"][:, args.act_idx].astype(float)
    q = data["qpos"][:, args.qpos_idx].astype(float)
    dq = data["qvel"][:, args.qvel_idx].astype(float)

    err = ctrl - q
    err_mean_abs = float(np.mean(np.abs(err)))
    err_rms = float(np.sqrt(np.mean(err**2)))
    err_p95 = float(np.percentile(np.abs(err), 95))
    err_max = float(np.max(np.abs(err)))

    ctrl_centered = ctrl - np.mean(ctrl)
    q_centered = q - np.mean(q)
    ctrl_rms = float(np.sqrt(np.mean(ctrl_centered**2)))
    q_rms = float(np.sqrt(np.mean(q_centered**2)))
    amp_ratio = q_rms / ctrl_rms if ctrl_rms > 0 else np.nan
    corr = (
        float(np.corrcoef(ctrl_centered, q_centered)[0, 1])
        if ctrl_rms > 0 and q_rms > 0
        else np.nan
    )

    xcorr = np.correlate(q_centered, ctrl_centered, mode="full")
    lag_samples = int(np.argmax(xcorr) - (len(ctrl_centered) - 1))
    lag_seconds = lag_samples * dt

    print("\nConverted SysID NPZ tracking check")
    print(f"  file:          {npz_path}")
    print(f"  source:        {_get_scalar_str(data, 'source_npz', '-')}")
    print(f"  measured:      {_get_scalar_str(data, 'measured_topic', '-')}")
    print(f"  command:       {_get_scalar_str(data, 'command_topic', '-')}")
    print(f"  joint:         {args.joint}")
    print(f"  indices:       act={args.act_idx}, qpos={args.qpos_idx}, qvel={args.qvel_idx}")
    print(f"  samples:       {len(time)}")
    print(f"  dt:            {dt:.6f} s")
    print(f"  duration:      {time[-1] - time[0]:.3f} s")
    print(f"  ctrl range:    {ctrl.min(): .6f} .. {ctrl.max(): .6f} rad")
    print(f"  q range:       {q.min(): .6f} .. {q.max(): .6f} rad")
    print(f"  dq range:      {dq.min(): .6f} .. {dq.max(): .6f} rad/s")
    print(f"  error mean abs:{err_mean_abs: .6f} rad")
    print(f"  error RMS:     {err_rms: .6f} rad")
    print(f"  error 95% abs: {err_p95: .6f} rad")
    print(f"  error max abs: {err_max: .6f} rad")
    print(f"  q/ctrl RMS amp:{amp_ratio: .3f}")
    print(f"  corr(ctrl, q): {corr: .4f}")
    print(f"  best lag:      {lag_samples:+d} samples ({lag_seconds:+.4f} s)")

    out = PLOT_DIR / f"{npz_path.stem}.plot.png"
    _plot_one(
        time=time,
        ctrl=ctrl,
        q=q,
        dq=dq,
        joint_name=args.joint,
        act_idx=args.act_idx,
        qpos_idx=args.qpos_idx,
        qvel_idx=args.qvel_idx,
        dt=dt,
        out=out,
    )
    print(f"Saved -> {out}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
