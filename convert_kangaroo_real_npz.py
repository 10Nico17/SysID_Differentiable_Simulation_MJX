"""Convert a real Kangaroo ROS-topic NPZ into the SysID NPZ format.

Usage (from mjx_sysid-main/):
    python scripts/convert_kangaroo_real_npz.py \
        assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011.npz

Output:
    assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz

The converter uses absolute ROS header times to synchronize topics. It resamples
actual joint state and desired commands onto a uniform time grid because the
MJX rollout uses one fixed timestep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT / "assets/robots/kangaroo_grippers/kangaroo_grippers_mjx.xml"
OUT_DIR = ROOT / "assets/datasets/kangaroo_grippers"

MEASURED_TOPIC = "subscriber_controller_actual_js_state"
COMMAND_TOPIC = "subscriber_controller_desired_state"


def _topic_key(topic: str, suffix: str) -> str:
    return f"{topic}__{suffix}"


def _messages(data: np.lib.npyio.NpzFile, topic: str) -> np.ndarray:
    key = _topic_key(topic, "messages")
    if key not in data.files:
        raise KeyError(f"Missing NPZ key: {key}")
    return data[key]


def _time_abs(data: np.lib.npyio.NpzFile, topic: str) -> np.ndarray:
    key = _topic_key(topic, "header_time")
    if key not in data.files:
        raise KeyError(f"Missing NPZ key: {key}")
    return data[key].astype(float)


def _topic_name(data: np.lib.npyio.NpzFile, topic: str) -> str:
    key = _topic_key(topic, "topic")
    return str(data[key][0]) if key in data.files else f"/{topic}"


def _extract_matrix(
    data: np.lib.npyio.NpzFile,
    topic: str,
    field: str,
) -> tuple[np.ndarray, list[str]]:
    msgs = _messages(data, topic)
    names = list(msgs[0]["name"])
    mat = np.asarray([msg[field] for msg in msgs], dtype=float)
    if mat.shape[1] != len(names):
        raise ValueError(
            f"{topic}.{field} has width {mat.shape[1]}, but name list has {len(names)}"
        )
    return mat, names


def _actuator_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]


def _initial_state(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    ctrl0 = np.zeros(model.nu)
    for act_id, act_name in enumerate(_actuator_names(model)):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, act_name)
        if jnt_id >= 0:
            ctrl0[act_id] = qpos0[model.jnt_qposadr[jnt_id]]
    return qpos0, qvel0, ctrl0


def _interp_columns(src_time: np.ndarray, src_values: np.ndarray, dst_time: np.ndarray) -> np.ndarray:
    out = np.empty((len(dst_time), src_values.shape[1]), dtype=float)
    for col in range(src_values.shape[1]):
        out[:, col] = np.interp(dst_time, src_time, src_values[:, col])
    return out


def convert(
    input_npz: Path,
    output_npz: Path,
    xml_path: Path,
    measured_topic: str,
    command_topic: str,
    dt: float | None,
) -> None:
    raw = np.load(input_npz, allow_pickle=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos0, qvel0, ctrl0 = _initial_state(model)
    act_names = _actuator_names(model)

    measured_time_abs = _time_abs(raw, measured_topic)
    command_time_abs = _time_abs(raw, command_topic)
    measured_pos, measured_names = _extract_matrix(raw, measured_topic, "position")
    measured_vel, _ = _extract_matrix(raw, measured_topic, "velocity")
    command_pos, command_names = _extract_matrix(raw, command_topic, "position")

    t0 = max(float(measured_time_abs[0]), float(command_time_abs[0]))
    t1 = min(float(measured_time_abs[-1]), float(command_time_abs[-1]))
    if t1 <= t0:
        raise ValueError("Measured and command topics do not overlap in absolute time.")

    measured_mask = (measured_time_abs >= t0) & (measured_time_abs <= t1)
    command_mask = (command_time_abs >= t0) & (command_time_abs <= t1)

    measured_time = measured_time_abs[measured_mask] - t0
    command_time = command_time_abs[command_mask] - t0
    measured_pos = measured_pos[measured_mask]
    measured_vel = measured_vel[measured_mask]
    command_pos = command_pos[command_mask]

    if dt is None:
        dt = float(np.median(np.diff(measured_time)))
    n_steps = int(np.floor((min(measured_time[-1], command_time[-1])) / dt)) + 1
    time = np.arange(n_steps, dtype=float) * dt

    measured_pos_u = _interp_columns(measured_time, measured_pos, time)
    measured_vel_u = _interp_columns(measured_time, measured_vel, time)
    command_pos_u = _interp_columns(command_time, command_pos, time)

    qpos = np.tile(qpos0, (n_steps, 1))
    qvel = np.tile(qvel0, (n_steps, 1))
    ctrl = np.tile(ctrl0, (n_steps, 1))

    qpos_mapped = 0
    qvel_mapped = 0
    for ros_idx, joint_name in enumerate(measured_names):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id < 0:
            continue
        qpos_adr = model.jnt_qposadr[jnt_id]
        dof_adr = model.jnt_dofadr[jnt_id]
        if model.jnt_type[jnt_id] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qpos[:, qpos_adr] = measured_pos_u[:, ros_idx]
            qvel[:, dof_adr] = measured_vel_u[:, ros_idx]
            qpos_mapped += 1
            qvel_mapped += 1

    ctrl_mapped = 0
    for ros_idx, joint_name in enumerate(command_names):
        if joint_name not in act_names:
            continue
        act_id = act_names.index(joint_name)
        ctrl[:, act_id] = command_pos_u[:, ros_idx]
        ctrl_mapped += 1

    np.savez(
        output_npz,
        time=time,
        ctrl=ctrl,
        qpos=qpos,
        qvel=qvel,
        actuator_names=np.asarray(act_names),
        dt=dt,
        xml_path=str(xml_path),
        source_npz=str(input_npz),
        measured_topic=_topic_name(raw, measured_topic),
        command_topic=_topic_name(raw, command_topic),
        measured_joint_names=np.asarray(measured_names),
        command_joint_names=np.asarray(command_names),
        sync_t0_abs=t0,
        sync_t1_abs=t1,
    )

    elbow = "arm_left_4_joint"
    act_idx = act_names.index(elbow)
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, elbow)
    qpos_idx = model.jnt_qposadr[jnt_id]
    qvel_idx = model.jnt_dofadr[jnt_id]
    err = ctrl[:, act_idx] - qpos[:, qpos_idx]

    print(f"Saved -> {output_npz}")
    print(f"  time: {time.shape}, dt={dt:.6f}, duration={time[-1]:.3f}s")
    print(f"  ctrl: {ctrl.shape}, qpos: {qpos.shape}, qvel: {qvel.shape}")
    print(f"  mapped measured joints: {qpos_mapped} qpos / {qvel_mapped} qvel")
    print(f"  mapped command joints:  {ctrl_mapped}")
    print(f"  measured topic: {_topic_name(raw, measured_topic)}")
    print(f"  command topic:  {_topic_name(raw, command_topic)}")
    print(f"  overlap abs:    {t0:.6f} .. {t1:.6f}")
    print("  arm_left_4_joint:")
    print(f"    ctrl idx={act_idx}, qpos idx={qpos_idx}, qvel idx={qvel_idx}")
    print(f"    ctrl range: {ctrl[:, act_idx].min(): .6f} .. {ctrl[:, act_idx].max(): .6f}")
    print(f"    qpos range: {qpos[:, qpos_idx].min(): .6f} .. {qpos[:, qpos_idx].max(): .6f}")
    print(f"    qvel range: {qvel[:, qvel_idx].min(): .6f} .. {qvel[:, qvel_idx].max(): .6f}")
    print(
        "    ctrl-q mean_abs/rms/max: "
        f"{np.mean(np.abs(err)):.6f} / {np.sqrt(np.mean(err**2)):.6f} / {np.max(np.abs(err)):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--measured-topic", default=MEASURED_TOPIC)
    parser.add_argument("--command-topic", default=COMMAND_TOPIC)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args()

    input_npz = args.input_npz.resolve()
    output_npz = (
        args.out.resolve()
        if args.out is not None
        else OUT_DIR / f"{input_npz.stem}_sysid.npz"
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    convert(
        input_npz=input_npz,
        output_npz=output_npz,
        xml_path=args.xml.resolve(),
        measured_topic=args.measured_topic,
        command_topic=args.command_topic,
        dt=args.dt,
    )


if __name__ == "__main__":
    main()
