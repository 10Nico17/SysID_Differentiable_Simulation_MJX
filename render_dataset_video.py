"""Render a side-by-side video: real dataset pose vs MJX ctrl rollout.

Left:  qpos/qvel from the converted SysID NPZ.
Right: MJX simulation driven by the same real ctrl commands.

The MJX rollout is computed step-by-step with a jitted MJX step, then rendered
with MuJoCo's offscreen renderer and encoded through ffmpeg.
        
python scripts/Kangaroo/04_render_mjx_dataset_video.py \
  --npz scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011_sysid.npz \
  --mode sim \
  --start 0 \
  --stop 12560 \
  --stride 2 \
  --fps 100 \
  --out scripts/Kangaroo/datasets/left_elbow_chirp_mjx_sim.mp4        
            
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


KANGAROO_DIR = Path(__file__).resolve().parent
ROOT = KANGAROO_DIR.parents[1]
DATASET_DIR = KANGAROO_DIR / "datasets"
DEFAULT_NPZ = (
    DATASET_DIR / "left_elbow_chirp_20260530_081011_sysid.npz"
)
DEFAULT_XML = ROOT / "assets/robots/kangaroo_grippers/kangaroo_grippers_mjx.xml"


###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--mode",
        choices=("side-by-side", "real", "sim"),
        default="side-by-side",
        help="Render comparison or only one motion.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Named camera from XML. If omitted, a front free camera is used.",
    )
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=180.0)
    parser.add_argument("--camera-elevation", type=float, default=-10.0)
    parser.add_argument("--no-gravity", action="store_true", help="Debug: disable XML gravity.")
    parser.add_argument("--disable-constraints", action="store_true", help="Debug: disable equality and contact.")
    parser.add_argument("--free-base", action="store_true", help="Do not pin floating base.")
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


###############################################################################
# Model / Data
###############################################################################


def load_sysid_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    raw = np.load(path.resolve(), allow_pickle=True)
    time = raw["time"].astype(np.float32)
    ctrl = raw["ctrl"].astype(np.float32)
    qpos = raw["qpos"].astype(np.float32)
    qvel = raw["qvel"].astype(np.float32)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(time)))
    return time, ctrl, qpos, qvel, dt


def load_model(args: argparse.Namespace, dt: float) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.opt.timestep = dt
    model.opt.iterations = int(args.iterations)

    if args.no_gravity:
        model.opt.gravity[:] = 0.0
    if args.disable_constraints:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    return model


def pin_base_mjx(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def pin_base_mj(data: mujoco.MjData) -> None:
    data.qpos[0:3] = [0.0, 0.0, 0.9]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[0:6] = 0.0


###############################################################################
# MJX Rollout
###############################################################################


def rollout_mjx(
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
        data = pin_base_mjx(data)

    @jax.jit
    def step_once(data_in: mjx.Data, ctrl_t: jnp.ndarray) -> mjx.Data:
        data_out = data_in.replace(ctrl=ctrl_t)
        data_out = mjx.step(mjx_model, data_out)
        if pin_base:
            data_out = pin_base_mjx(data_out)
        return data_out

    qpos_hist = np.zeros((len(ctrl), model.nq), dtype=np.float32)
    qvel_hist = np.zeros((len(ctrl), model.nv), dtype=np.float32)

    for k, ctrl_t in enumerate(ctrl):
        qpos_hist[k] = np.asarray(data.qpos)
        qvel_hist[k] = np.asarray(data.qvel)
        data = step_once(data, jnp.asarray(ctrl_t))

    return qpos_hist, qvel_hist


###############################################################################
# Rendering / Video
###############################################################################


def make_free_camera(args: argparse.Namespace) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    return camera


def render_state(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray,
    camera: str | mujoco.MjvCamera,
    pin_base: bool,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    if pin_base:
        pin_base_mj(data)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def open_ffmpeg_writer(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_video(
    *,
    model: mujoco.MjModel,
    qpos_real: np.ndarray,
    qvel_real: np.ndarray,
    qpos_sim: np.ndarray,
    qvel_sim: np.ndarray,
    frame_indices: np.ndarray,
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    camera: str | mujoco.MjvCamera,
    mode: str,
    pin_base: bool,
) -> None:
    left_renderer = mujoco.Renderer(model, height=height, width=width)
    right_renderer = mujoco.Renderer(model, height=height, width=width)
    real_data = mujoco.MjData(model)
    sim_data = mujoco.MjData(model)

    video_width = width * 2 if mode == "side-by-side" else width
    video_height = height
    proc = open_ffmpeg_writer(out_path, video_width, video_height, fps)
    if proc.stdin is None:
        raise RuntimeError("Could not open ffmpeg stdin.")

    try:
        for i, k in enumerate(frame_indices):
            if mode == "real":
                frame = render_state(
                    left_renderer,
                    model,
                    real_data,
                    qpos_real[k],
                    qvel_real[k],
                    camera,
                    pin_base,
                )
            elif mode == "sim":
                frame = render_state(
                    right_renderer,
                    model,
                    sim_data,
                    qpos_sim[k],
                    qvel_sim[k],
                    camera,
                    pin_base,
                )
            else:
                left = render_state(
                    left_renderer,
                    model,
                    real_data,
                    qpos_real[k],
                    qvel_real[k],
                    camera,
                    pin_base,
                )
                right = render_state(
                    right_renderer,
                    model,
                    sim_data,
                    qpos_sim[k],
                    qvel_sim[k],
                    camera,
                    pin_base,
                )
                frame = np.concatenate([left, right], axis=1)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if i % 50 == 0:
                print(f"  rendered frame {i + 1}/{len(frame_indices)}")
    finally:
        proc.stdin.close()
        ret = proc.wait()
        left_renderer.close()
        right_renderer.close()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}")


###############################################################################
# Main
###############################################################################


def main() -> None:
    args = parse_args()
    npz_path = args.npz.resolve()
    xml_path = args.xml.resolve()
    out_path = (
        args.out.resolve()
        if args.out is not None
        else DATASET_DIR / f"{npz_path.stem}.mjx_real_vs_sim.mp4"
    )

    time, ctrl, qpos_real, qvel_real, dt = load_sysid_npz(npz_path)
    model = load_model(args, dt)

    start = max(0, args.start)
    stop = min(len(ctrl), args.stop)
    if stop <= start:
        raise ValueError(f"Invalid range: start={start}, stop={stop}")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    ctrl_win = ctrl[start:stop]
    qpos0 = qpos_real[start]
    qvel0 = qvel_real[start]

    print("\nMJX dataset video")
    print(f"  XML:          {xml_path}")
    print(f"  NPZ:          {npz_path}")
    print(f"  out:          {out_path}")
    print(f"  samples:      {start} .. {stop - 1} ({stop - start})")
    print(f"  frames:       {len(np.arange(0, stop - start, args.stride))}")
    print(f"  dt:           {dt:.6f} s")
    print(f"  fps:          {args.fps:.2f}")
    print(f"  mode:         {args.mode}")
    print(
        "  camera:       "
        + (
            f"named '{args.camera}'"
            if args.camera is not None
            else (
                "front free "
                f"(azimuth={args.camera_azimuth:.1f}, "
                f"elevation={args.camera_elevation:.1f}, "
                f"distance={args.camera_distance:.1f})"
            )
        )
    )
    print(f"  gravity:      {'off' if args.no_gravity else 'on'}")
    print(f"  constraints:  {'off' if args.disable_constraints else 'on'}")
    print(f"  base:         {'free' if args.free_base else 'pinned'}")
    if args.mode == "side-by-side":
        print("  left:         real qpos from dataset")
        print("  right:        MJX rollout from real ctrl")
    elif args.mode == "real":
        print("  rendered:     real qpos from dataset")
    else:
        print("  rendered:     MJX rollout from real ctrl")

    print("\nRolling out MJX...")
    qpos_sim_win, qvel_sim_win = rollout_mjx(
        model=model,
        qpos0=qpos0,
        qvel0=qvel0,
        ctrl=ctrl_win,
        pin_base=not args.free_base,
    )

    frame_indices = np.arange(0, stop - start, args.stride)
    camera = args.camera if args.camera is not None else make_free_camera(args)
    print("\nRendering video...")
    write_video(
        model=model,
        qpos_real=qpos_real[start:stop],
        qvel_real=qvel_real[start:stop],
        qpos_sim=qpos_sim_win,
        qvel_sim=qvel_sim_win,
        frame_indices=frame_indices,
        out_path=out_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera=camera,
        mode=args.mode,
        pin_base=not args.free_base,
    )
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
