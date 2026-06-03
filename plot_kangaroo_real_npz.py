"""Plot relevant ROS topics from a recorded Kangaroo real-robot NPZ.

Usage (from mjx_sysid-main/):
    python scripts/plot_kangaroo_real_npz.py
    python scripts/plot_kangaroo_real_npz.py assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011.npz

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
DEFAULT_NPZ = (
    DATASET_DIR / "left_elbow_chirp_20260530_081011.npz"
)

MEASURED_TOPIC = "subscriber_controller_actual_js_state"
COMMAND_TOPIC = "subscriber_controller_desired_state"
JOINT_NAME = "arm_left_4_joint"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--measured-topic", default=MEASURED_TOPIC)
    parser.add_argument("--command-topic", default=COMMAND_TOPIC)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    data = np.load(npz_path, allow_pickle=True)

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
    out = (
        args.out
        if args.out is not None
        else DATASET_DIR / f"{npz_path.stem}.topics.png"
    )
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
