"""Replay a converted Kangaroo SysID NPZ in the MuJoCo GUI.

Example:
    conda run -n mjx_sysid python replay_dataset_viewer.py \
      datasets/Kangaroo_real/leg_right_sequence_chirp_2026-07-02_09-36-08_sysid.npz \
      --pin-base
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_XML = ROOT / "Robot/kangaroo_grippers_mjx.xml"
DEFAULT_NPZ = (
    ROOT
    / "datasets/Kangaroo_real/leg_right_sequence_chirp_2026-07-02_09-36-08_sysid.npz"
)


def pin_base(data: mujoco.MjData, height: float) -> None:
    data.qpos[0:3] = [0.0, 0.0, height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[0:6] = 0.0


def joint_slots_for_prefixes(
    model: mujoco.MjModel,
    prefixes: tuple[str, ...],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    qpos_slots = []
    qvel_slots = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name or not joint_name.startswith(prefixes):
            continue
        joint_type = model.jnt_type[joint_id]
        qadr = int(model.jnt_qposadr[joint_id])
        vadr = int(model.jnt_dofadr[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_slots.append((qadr, qadr + 7))
            qvel_slots.append((vadr, vadr + 6))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            qpos_slots.append((qadr, qadr + 4))
            qvel_slots.append((vadr, vadr + 3))
        else:
            qpos_slots.append((qadr, qadr + 1))
            qvel_slots.append((vadr, vadr + 1))
    return qpos_slots, qvel_slots


def actuator_ids_for_prefixes(
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


def drive_prefixes(drive: str) -> tuple[str, ...]:
    if drive == "right-leg":
        return ("leg_right_",)
    if drive == "left-leg":
        return ("leg_left_",)
    if drive == "legs":
        return ("leg_left_", "leg_right_")
    return ()


def hold_slots(
    data: mujoco.MjData,
    qpos_hold: np.ndarray,
    qvel_hold: np.ndarray,
    qpos_slots: list[tuple[int, int]],
    qvel_slots: list[tuple[int, int]],
) -> None:
    for start, stop in qpos_slots:
        data.qpos[start:stop] = qpos_hold[start:stop]
    for start, stop in qvel_slots:
        data.qvel[start:stop] = qvel_hold[start:stop]


def hold_actuator_targets(
    data: mujoco.MjData,
    qpos_hold: np.ndarray,
    actuator_slots: list[tuple[int, int]],
) -> None:
    for act_id, qpos_idx in actuator_slots:
        data.ctrl[act_id] = qpos_hold[qpos_idx]


def set_command_pose(
    data: mujoco.MjData,
    ctrl_row: np.ndarray,
    drive_actuator_slots: list[tuple[int, int]],
) -> None:
    for act_id, qpos_idx in drive_actuator_slots:
        data.qpos[qpos_idx] = ctrl_row[act_id]
    data.qvel[:] = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=0.9)
    parser.add_argument("--pin-base", action="store_true")
    parser.add_argument("--hide-ground", action="store_true", help="Make the ground plane transparent in the viewer.")
    parser.add_argument(
        "--drive",
        choices=("all", "right-leg", "left-leg", "legs"),
        default="all",
        help="Which dataset ctrl targets to apply. Non-driven joints are held at the start pose.",
    )
    parser.add_argument("--hold-arms", action="store_true", help="Keep arm joints fixed at the start pose.")
    parser.add_argument(
        "--background",
        choices=("start", "real"),
        default="start",
        help="For non-driven joints: use the start pose or track qpos/qvel from the dataset.",
    )
    parser.add_argument("--real-state", action="store_true", help="Show recorded qpos/qvel instead of simulating ctrl.")
    parser.add_argument(
        "--command-state",
        action="store_true",
        help="Show selected ctrl targets directly as qpos instead of simulating dynamics.",
    )
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-gravity", action="store_true")
    return parser.parse_args()


def hold_prefixes_for_drive(drive: str, hold_arms: bool) -> tuple[str, ...]:
    prefixes = []
    if drive == "right-leg":
        prefixes.append("leg_left_")
    if drive == "left-leg":
        prefixes.append("leg_right_")
    if drive not in ("all", "legs"):
        prefixes.extend(["pelvis_", "arm_left_", "arm_right_"])
    elif hold_arms:
        prefixes.extend(["arm_left_", "arm_right_"])
    return tuple(prefixes)


def background_qpos(background: str, qpos_ref: np.ndarray, qpos_hold: np.ndarray) -> np.ndarray:
    return qpos_ref if background == "real" else qpos_hold


def background_qvel(background: str, qvel_ref: np.ndarray, qvel_hold: np.ndarray) -> np.ndarray:
    return qvel_ref if background == "real" else qvel_hold


def main() -> None:
    args = parse_args()
    raw = np.load(args.npz.resolve(), allow_pickle=True)
    qpos = raw["qpos"].astype(float)
    qvel = raw["qvel"].astype(float)
    ctrl = raw["ctrl"].astype(float)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(raw["time"])))

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.opt.timestep = dt
    if args.no_gravity:
        model.opt.gravity[:] = 0.0
    if args.hide_ground:
        ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        if ground_id >= 0:
            model.geom_rgba[ground_id, 3] = 0.0

    data = mujoco.MjData(model)
    start = max(0, min(args.start, len(ctrl) - 1))
    stop = len(ctrl) if args.stop is None else max(start + 1, min(args.stop, len(ctrl)))
    qpos_hold = qpos[start].copy()
    qvel_hold = qvel[start].copy()
    hold_prefixes = hold_prefixes_for_drive(args.drive, args.hold_arms)
    hold_qpos_slots, hold_qvel_slots = joint_slots_for_prefixes(model, hold_prefixes)
    hold_actuator_slots = actuator_ids_for_prefixes(model, hold_prefixes)
    drive_actuator_slots = actuator_ids_for_prefixes(model, drive_prefixes(args.drive))

    def reset_to_start() -> int:
        data.qpos[:] = qpos[start]
        data.qvel[:] = qvel[start]
        data.ctrl[:] = ctrl[start]
        hold_actuator_targets(
            data,
            background_qpos(args.background, qpos[start], qpos_hold),
            hold_actuator_slots,
        )
        hold_slots(
            data,
            background_qpos(args.background, qpos[start], qpos_hold),
            background_qvel(args.background, qvel[start], qvel_hold),
            hold_qpos_slots,
            hold_qvel_slots,
        )
        if args.pin_base:
            pin_base(data, args.height)
        mujoco.mj_forward(model, data)
        return start

    k = reset_to_start()
    sleep_dt = dt / max(args.speed, 1e-6)

    print(f"NPZ:        {args.npz.resolve()}")
    print(f"XML:        {args.xml.resolve()}")
    print(f"steps:      {start} .. {stop - 1}")
    print(f"dt:         {dt:.6f} s")
    if args.command_state:
        mode = "direct ctrl targets as qpos"
    elif args.real_state:
        mode = "recorded real qpos/qvel"
    else:
        mode = "simulate ctrl"
    print(f"mode:       {mode}")
    print(f"base:       {'pinned in air' if args.pin_base else 'from data/free'}")
    print(f"drive:      {args.drive}")
    print(f"background: {args.background}")
    print(f"held:       {', '.join(hold_prefixes) if hold_prefixes else '-'}")
    print("Close the MuJoCo window to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_t0 = time.time()

            if args.command_state:
                data.qpos[:] = background_qpos(args.background, qpos[k], qpos_hold)
                data.qvel[:] = background_qvel(args.background, qvel[k], qvel_hold)
                set_command_pose(data, ctrl[k], drive_actuator_slots)
                hold_slots(
                    data,
                    background_qpos(args.background, qpos[k], qpos_hold),
                    background_qvel(args.background, qvel[k], qvel_hold),
                    hold_qpos_slots,
                    hold_qvel_slots,
                )
                if args.pin_base:
                    pin_base(data, args.height)
                mujoco.mj_forward(model, data)
            elif args.real_state:
                data.qpos[:] = qpos[k]
                data.qvel[:] = qvel[k]
                hold_slots(
                    data,
                    background_qpos(args.background, qpos[k], qpos_hold),
                    background_qvel(args.background, qvel[k], qvel_hold),
                    hold_qpos_slots,
                    hold_qvel_slots,
                )
                if args.pin_base:
                    pin_base(data, args.height)
                mujoco.mj_forward(model, data)
            else:
                data.ctrl[:] = ctrl[k]
                hold_actuator_targets(
                    data,
                    background_qpos(args.background, qpos[k], qpos_hold),
                    hold_actuator_slots,
                )
                hold_slots(
                    data,
                    background_qpos(args.background, qpos[k], qpos_hold),
                    background_qvel(args.background, qvel[k], qvel_hold),
                    hold_qpos_slots,
                    hold_qvel_slots,
                )
                if args.pin_base:
                    pin_base(data, args.height)
                mujoco.mj_step(model, data)
                hold_slots(
                    data,
                    background_qpos(args.background, qpos[k], qpos_hold),
                    background_qvel(args.background, qvel[k], qvel_hold),
                    hold_qpos_slots,
                    hold_qvel_slots,
                )
                if args.pin_base:
                    pin_base(data, args.height)
                    mujoco.mj_forward(model, data)

            viewer.sync()
            k += 1
            if k >= stop:
                if not args.loop:
                    break
                k = reset_to_start()

            elapsed = time.time() - frame_t0
            time.sleep(max(0.0, sleep_dt - elapsed))


if __name__ == "__main__":
    main()
