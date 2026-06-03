"""Plot a converted Kangaroo SysID NPZ.

Usage (from mjx_sysid-main/):
    python scripts/plot_kangaroo_sysid_npz.py
    python scripts/plot_kangaroo_sysid_npz.py assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz

Expected NPZ keys:
    time, ctrl, qpos, qvel, actuator_names, dt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    ROOT
    / "assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz"
)

JOINT_NAME = "arm_left_4_joint"
ACT_IDX = 19
QPOS_IDX = 12
QVEL_IDX = 11


def _get_scalar_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    return str(value.item() if value.shape == () else value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--act-idx", type=int, default=ACT_IDX)
    parser.add_argument("--qpos-idx", type=int, default=QPOS_IDX)
    parser.add_argument("--qvel-idx", type=int, default=QVEL_IDX)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    data = np.load(npz_path, allow_pickle=True)

    time = data["time"].astype(float)
    ctrl = data["ctrl"][:, args.act_idx].astype(float)
    q = data["qpos"][:, args.qpos_idx].astype(float)
    dq = data["qvel"][:, args.qvel_idx].astype(float)
    dt = float(data["dt"]) if "dt" in data.files else float(np.median(np.diff(time)))

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

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Converted SysID NPZ - {args.joint}")

    axes[0].plot(time, ctrl, color="gray", linewidth=0.9, label="ctrl")
    axes[0].plot(time, q, color="blue", linewidth=0.9, label="qpos")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        err,
        color="red",
        linewidth=0.8,
        label=f"ctrl - qpos, rms={err_rms:.3f}, max={err_max:.3f}",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("error [rad]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, dq, color="green", linewidth=0.8, label="qvel")
    axes[2].set_ylabel("velocity [rad/s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time[1:], np.diff(time), color="purple", linewidth=0.8, label="dt")
    axes[3].set_ylabel("dt [s]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.out if args.out is not None else npz_path.with_suffix(".plot.png")
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
