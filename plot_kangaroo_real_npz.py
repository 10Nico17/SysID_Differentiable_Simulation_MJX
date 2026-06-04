"""Plot relevant ROS topics from a recorded Kangaroo real-robot NPZ.

Usage (from mjx_sysid-main/):
    python scripts/Kangaroo/plot_kangaroo_real_npz.py
    python scripts/Kangaroo/plot_kangaroo_real_npz.py scripts/Kangaroo/datasets/Arms.npz --all-arm-joints --split-joints

The script uses absolute ROS header times to synchronize topics. Do not compare
the per-topic *_header_time_rel arrays directly across topics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


KANGAROO_DIR = Path(__file__).resolve().parent
ROOT = KANGAROO_DIR.parents[1]
DATASET_DIR = KANGAROO_DIR / "datasets"
PLOT_DIR = DATASET_DIR / "plots_raw"
DEFAULT_NPZ = (
    DATASET_DIR / "left_elbow_chirp_20260530_081011.npz"
)

MEASURED_TOPIC = "subscriber_controller_actual_js_state"
COMMAND_TOPIC = "subscriber_controller_desired_state"
JOINT_NAME = "arm_left_4_joint"
LEFT_ARM_JOINTS = tuple(f"arm_left_{i}_joint" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"arm_right_{i}_joint" for i in range(1, 8))


def _topic_key(topic: str, suffix: str) -> str:
    return f"{topic}__{suffix}"


def _messages(data: np.lib.npyio.NpzFile, topic: str) -> np.ndarray:
    return data[_topic_key(topic, "messages")]


def _topic_name(data: np.lib.npyio.NpzFile, topic: str) -> str:
    key = _topic_key(topic, "topic")
    return str(data[key][0]) if key in data.files else f"/{topic}"


def _extract_joint(
    data: np.lib.npyio.NpzFile,
    topic: str,
    joint_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    msgs = _messages(data, topic)
    first = msgs[0]
    names = list(first["name"])
    if joint_name not in names:
        raise ValueError(f"{joint_name!r} not found in {topic}; available: {names}")

    joint_idx = names.index(joint_name)
    time_abs = data[_topic_key(topic, "header_time")].astype(float)
    pos = np.fromiter(
        (msg["position"][joint_idx] for msg in msgs),
        dtype=float,
        count=len(msgs),
    )
    vel = np.fromiter(
        (
            msg["velocity"][joint_idx]
            if "velocity" in msg and len(msg["velocity"]) > joint_idx
            else np.nan
            for msg in msgs
        ),
        dtype=float,
        count=len(msgs),
    )
    return time_abs, pos, vel, joint_idx, names


def _print_time_stats(label: str, time_abs: np.ndarray) -> None:
    dt = np.diff(time_abs)
    print(f"  {label:<10} samples: {len(time_abs)}")
    print(f"  {label:<10} abs:     {time_abs[0]:.6f} .. {time_abs[-1]:.6f} s")
    print(
        f"  {label:<10} dt:      mean={dt.mean():.6f} "
        f"median={np.median(dt):.6f} min={dt.min():.6f} max={dt.max():.6f} s"
    )


def _arm_joint_names(side: str) -> tuple[str, ...]:
    if side == "left":
        return LEFT_ARM_JOINTS
    if side == "right":
        return RIGHT_ARM_JOINTS
    return LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


def _plot_all_arm_joints(
    *,
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    measured_topic: str,
    command_topic: str,
    side: str,
    split_joints: bool,
) -> None:
    joint_names = _arm_joint_names(side)
    first_measured = _messages(data, measured_topic)[0]
    first_command = _messages(data, command_topic)[0]
    measured_names = list(first_measured["name"])
    command_names = list(first_command["name"])

    missing = [
        name
        for name in joint_names
        if name not in measured_names or name not in command_names
    ]
    if missing:
        raise ValueError(f"Missing arm joints in topics: {missing}")

    q_time_abs = data[_topic_key(measured_topic, "header_time")].astype(float)
    ctrl_time_abs = data[_topic_key(command_topic, "header_time")].astype(float)
    t0 = max(float(q_time_abs[0]), float(ctrl_time_abs[0]))
    t1 = min(float(q_time_abs[-1]), float(ctrl_time_abs[-1]))
    if t1 <= t0:
        raise ValueError(
            "Measured and command topics do not overlap in absolute header time."
        )

    q_mask = (q_time_abs >= t0) & (q_time_abs <= t1)
    ctrl_mask = (ctrl_time_abs >= t0) & (ctrl_time_abs <= t1)
    q_time = q_time_abs[q_mask] - t0
    ctrl_time = ctrl_time_abs[ctrl_mask] - t0

    measured_msgs = _messages(data, measured_topic)[q_mask]
    command_msgs = _messages(data, command_topic)[ctrl_mask]
    n = len(joint_names)

    print("\nReal Kangaroo arm topic tracking check")
    print(f"  file:          {npz_path}")
    print(f"  measured:      {_topic_name(data, measured_topic)}")
    print(f"  command:       {_topic_name(data, command_topic)}")
    print(f"  side:          {side}")
    print(f"  joints:        {n}")
    print(f"  overlap:       {t0:.6f} .. {t1:.6f} s ({t1 - t0:.3f} s)")
    print("  joint                       q_rms_err    q_max_err    corr")

    if not split_joints:
        fig, axes = plt.subplots(n, 3, figsize=(18, max(10, 1.8 * n)), sharex="col")
        fig.suptitle(f"Real Kangaroo arm topics - {side}")
    else:
        axes = None
        out_dir = PLOT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    for row, joint_name in enumerate(joint_names):
        q_idx = measured_names.index(joint_name)
        ctrl_idx = command_names.index(joint_name)

        q = np.fromiter(
            (msg["position"][q_idx] for msg in measured_msgs),
            dtype=float,
            count=len(measured_msgs),
        )
        dq = np.fromiter(
            (
                msg["velocity"][q_idx]
                if "velocity" in msg and len(msg["velocity"]) > q_idx
                else np.nan
                for msg in measured_msgs
            ),
            dtype=float,
            count=len(measured_msgs),
        )
        ctrl = np.fromiter(
            (msg["position"][ctrl_idx] for msg in command_msgs),
            dtype=float,
            count=len(command_msgs),
        )
        ctrl_on_q = np.interp(q_time, ctrl_time, ctrl)
        err = ctrl_on_q - q

        q_centered = q - np.mean(q)
        ctrl_centered = ctrl_on_q - np.mean(ctrl_on_q)
        corr = (
            float(np.corrcoef(ctrl_centered, q_centered)[0, 1])
            if np.std(q_centered) > 0 and np.std(ctrl_centered) > 0
            else np.nan
        )
        err_rms = float(np.sqrt(np.mean(err**2)))
        err_max = float(np.max(np.abs(err)))
        print(f"  {joint_name:<27} {err_rms: .6f}   {err_max: .6f}   {corr: .4f}")

        if split_joints:
            fig_one, axes_one = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            fig_one.suptitle(f"Real Kangaroo ROS topics - {joint_name}")

            axes_one[0].plot(ctrl_time, ctrl, color="gray", linewidth=0.9, label="ctrl desired")
            axes_one[0].plot(q_time, q, color="blue", linewidth=0.9, label="q actual")
            axes_one[0].set_ylabel("position [rad]")
            axes_one[0].legend(loc="upper right")
            axes_one[0].grid(True, alpha=0.3)

            axes_one[1].plot(
                q_time,
                err,
                color="red",
                linewidth=0.8,
                label=f"ctrl_on_q - q, rms={err_rms:.3f}, max={err_max:.3f}",
            )
            axes_one[1].axhline(0.0, color="black", linewidth=0.5)
            axes_one[1].set_ylabel("error [rad]")
            axes_one[1].legend(loc="upper right")
            axes_one[1].grid(True, alpha=0.3)

            axes_one[2].plot(q_time, dq, color="green", linewidth=0.8, label="dq actual")
            axes_one[2].set_ylabel("velocity [rad/s]")
            axes_one[2].set_xlabel("time since overlap start [s]")
            axes_one[2].legend(loc="upper right")
            axes_one[2].grid(True, alpha=0.3)

            fig_one.tight_layout()
            out_path = out_dir / f"{npz_path.stem}.{joint_name}.topics.png"
            fig_one.savefig(out_path, dpi=150)
            plt.close(fig_one)
            print(f"Saved -> {out_path}")
            continue

        axes[row, 0].plot(ctrl_time, ctrl, color="gray", linewidth=0.7)
        axes[row, 0].plot(q_time, q, color="blue", linewidth=0.7)
        axes[row, 0].set_ylabel(joint_name, fontsize=8)
        axes[row, 0].grid(True, alpha=0.25)

        axes[row, 1].plot(q_time, err, color="red", linewidth=0.7)
        axes[row, 1].axhline(0.0, color="black", linewidth=0.4)
        axes[row, 1].grid(True, alpha=0.25)

        axes[row, 2].plot(q_time, dq, color="green", linewidth=0.7)
        axes[row, 2].grid(True, alpha=0.25)

    if not split_joints:
        axes[0, 0].set_title("ctrl desired / q actual")
        axes[0, 1].set_title("ctrl_on_q - q")
        axes[0, 2].set_title("dq actual")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        axes[-1, 2].set_xlabel("time [s]")

        plt.tight_layout()
        out_path = PLOT_DIR / f"{npz_path.stem}.arm_{side}.topics.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved -> {out_path}")

        if "agg" not in matplotlib.get_backend().lower():
            plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--all-arm-joints", action="store_true")
    parser.add_argument("--arm-side", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--split-joints",
        action="store_true",
        help="With --all-arm-joints, save one PNG per arm joint.",
    )
    parser.add_argument("--measured-topic", default=MEASURED_TOPIC)
    parser.add_argument("--command-topic", default=COMMAND_TOPIC)
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    data = np.load(npz_path, allow_pickle=True)

    if args.all_arm_joints:
        _plot_all_arm_joints(
            data=data,
            npz_path=npz_path,
            measured_topic=args.measured_topic,
            command_topic=args.command_topic,
            side=args.arm_side,
            split_joints=args.split_joints,
        )
        return

    q_time_abs, q, dq, q_idx, _ = _extract_joint(
        data, args.measured_topic, args.joint
    )
    ctrl_time_abs, ctrl, ctrl_vel, ctrl_idx, _ = _extract_joint(
        data, args.command_topic, args.joint
    )

    t0 = max(float(q_time_abs[0]), float(ctrl_time_abs[0]))
    t1 = min(float(q_time_abs[-1]), float(ctrl_time_abs[-1]))
    if t1 <= t0:
        raise ValueError(
            "Measured and command topics do not overlap in absolute header time."
        )

    q_mask = (q_time_abs >= t0) & (q_time_abs <= t1)
    ctrl_mask = (ctrl_time_abs >= t0) & (ctrl_time_abs <= t1)

    q_time = q_time_abs[q_mask] - t0
    ctrl_time = ctrl_time_abs[ctrl_mask] - t0
    q = q[q_mask]
    dq = dq[q_mask]
    ctrl = ctrl[ctrl_mask]
    ctrl_vel = ctrl_vel[ctrl_mask]

    ctrl_on_q = np.interp(q_time, ctrl_time, ctrl)
    err = ctrl_on_q - q

    err_mean_abs = float(np.mean(np.abs(err)))
    err_rms = float(np.sqrt(np.mean(err**2)))
    err_p95 = float(np.percentile(np.abs(err), 95))
    err_max = float(np.max(np.abs(err)))

    q_centered = q - np.mean(q)
    ctrl_centered = ctrl_on_q - np.mean(ctrl_on_q)
    q_rms = float(np.sqrt(np.mean(q_centered**2)))
    ctrl_rms = float(np.sqrt(np.mean(ctrl_centered**2)))
    amp_ratio = q_rms / ctrl_rms if ctrl_rms > 0 else np.nan
    corr = (
        float(np.corrcoef(ctrl_centered, q_centered)[0, 1])
        if ctrl_rms > 0 and q_rms > 0
        else np.nan
    )

    xcorr = np.correlate(q_centered, ctrl_centered, mode="full")
    lag_samples = int(np.argmax(xcorr) - (len(ctrl_centered) - 1))
    lag_seconds = lag_samples * float(np.median(np.diff(q_time)))

    print("\nReal Kangaroo NPZ tracking check")
    print(f"  file:          {npz_path}")
    print(f"  measured:      {_topic_name(data, args.measured_topic)}")
    print(f"  command:       {_topic_name(data, args.command_topic)}")
    print(f"  joint:         {args.joint}")
    print(f"  indices:       measured={q_idx}, command={ctrl_idx}")
    print(f"  overlap:       {t0:.6f} .. {t1:.6f} s ({t1 - t0:.3f} s)")
    _print_time_stats("measured", q_time_abs)
    _print_time_stats("command", ctrl_time_abs)
    print(f"  q range:       {q.min(): .6f} .. {q.max(): .6f} rad")
    print(f"  ctrl range:    {ctrl.min(): .6f} .. {ctrl.max(): .6f} rad")
    print(f"  dq range:      {np.nanmin(dq): .6f} .. {np.nanmax(dq): .6f} rad/s")
    print(f"  error mean abs:{err_mean_abs: .6f} rad")
    print(f"  error RMS:     {err_rms: .6f} rad")
    print(f"  error 95% abs: {err_p95: .6f} rad")
    print(f"  error max abs: {err_max: .6f} rad")
    print(f"  q/ctrl RMS amp:{amp_ratio: .3f}")
    print(f"  corr(ctrl, q): {corr: .4f}")
    print(f"  best lag:      {lag_samples:+d} measured samples ({lag_seconds:+.4f} s)")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Real Kangaroo {args.joint} - ROS topic tracking")

    axes[0].plot(ctrl_time, ctrl, color="gray", linewidth=0.9, label="ctrl desired")
    axes[0].plot(q_time, q, color="blue", linewidth=0.9, label="q actual")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        q_time,
        err,
        color="red",
        linewidth=0.8,
        label=f"ctrl_on_q - q, rms={err_rms:.3f}, max={err_max:.3f}",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("error [rad]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(q_time, dq, color="green", linewidth=0.8, label="dq actual")
    axes[2].set_ylabel("velocity [rad/s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(q_time[1:], np.diff(q_time), color="purple", linewidth=0.8, label="actual dt")
    axes[3].plot(
        ctrl_time[1:],
        np.diff(ctrl_time),
        color="orange",
        linewidth=0.8,
        label="command dt",
    )
    axes[3].set_ylabel("dt [s]")
    axes[3].set_xlabel("time since overlap start [s]")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / f"{npz_path.stem}.topics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
