"""Optimize Kangaroo elbow parameters with a q-only MJX loss.

Standalone script. It does not use the repo task registry.

Run from repo root:
    python scripts/Kangaroo/03_optimize_q.py \
        --npz assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz \
        --horizon 100 \
        --max-iter 100 \
        --lr 0.002 \
        --optimize armature
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from mujoco import mjx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    ROOT
    / "assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz"
)
DEFAULT_XML = ROOT / "assets/robots/kangaroo_grippers/kangaroo_grippers_mjx.xml"

JOINT_NAME = "arm_left_4_joint"
PARAM_NAMES = ("armature", "damping", "frictionloss")

ARMATURE_DEFAULT = 0.00663065
DAMPING_DEFAULT = 1e-4
FRICTIONLOSS_DEFAULT = 1e-4


###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)

    parser.add_argument(
        "--armature",
        type=float,
        default=None,
        help="Override start armature. Default: value from XML.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help="Override start damping. Default: value from XML.",
    )
    parser.add_argument(
        "--frictionloss",
        type=float,
        default=None,
        help="Override start frictionloss. Default: value from XML.",
    )
    parser.add_argument(
        "--optimize",
        nargs="+",
        choices=PARAM_NAMES,
        default=["armature"],
        help="Selected parameters to update.",
    )

    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--no-gravity", action="store_true", help="Debug: disable XML gravity.")
    parser.add_argument("--disable-constraints", action="store_true", help="Debug: disable equality and contact.")
    parser.add_argument("--free-base", action="store_true", help="Do not pin floating base.")
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


###############################################################################
# Model
###############################################################################


def name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Could not find {name!r} in model.")
    return int(idx)


def load_mujoco_model(args: argparse.Namespace, dt: float) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.opt.timestep = dt
    model.opt.iterations = int(args.iterations)

    if args.no_gravity:
        model.opt.gravity[:] = 0.0
    if args.disable_constraints:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    return model


def get_joint_indices(model: mujoco.MjModel, joint_name: str) -> dict[str, int]:
    joint_id = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    act_idx = name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    dof_idx = int(model.jnt_dofadr[joint_id])
    return {
        "joint_id": joint_id,
        "act_idx": act_idx,
        "qpos_idx": int(model.jnt_qposadr[joint_id]),
        "qvel_idx": dof_idx,
        "dof_idx": dof_idx,
    }


def pin_base(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def print_model_info(
    args: argparse.Namespace,
    indices: dict[str, int],
    init_values: np.ndarray,
) -> None:
    print("\nModel")
    print(f"  XML:          {args.xml.resolve()}")
    print(f"  joint:        {args.joint}")
    print(f"  joint_id:     {indices['joint_id']}")
    print(f"  act_idx:      {indices['act_idx']}")
    print(f"  qpos_idx:     {indices['qpos_idx']}")
    print(f"  qvel_idx:     {indices['qvel_idx']}")
    print(f"  dof_idx:      {indices['dof_idx']}")
    print(f"  gravity:      {'off' if args.no_gravity else 'on'}")
    print(f"  constraints:  {'off' if args.disable_constraints else 'on'}")
    print(f"  base:         {'free' if args.free_base else 'pinned'}")
    print(f"  optimize:     {', '.join(args.optimize)}")
    print(
        "  start params: "
        f"armature={init_values[0]:.8f} "
        f"damping={init_values[1]:.8f} "
        f"frictionloss={init_values[2]:.8f}"
    )


###############################################################################
# Data
###############################################################################


def load_sysid_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    raw = np.load(path.resolve(), allow_pickle=True)
    time = raw["time"].astype(np.float32)
    ctrl = raw["ctrl"].astype(np.float32)
    qpos = raw["qpos"].astype(np.float32)
    qvel = raw["qvel"].astype(np.float32)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(time)))
    return time, ctrl, qpos, qvel, dt


def make_single_window(
    args: argparse.Namespace,
    ctrl: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qpos_idx: int,
) -> dict[str, jnp.ndarray]:
    if args.start < 0:
        raise ValueError("--start must be >= 0")

    end = min(args.start + args.horizon, len(ctrl) - 1)
    if end <= args.start:
        raise ValueError("Empty rollout window. Check --start and --horizon.")

    return {
        "ctrl": jnp.asarray(ctrl[args.start:end]),
        "qpos0": jnp.asarray(qpos[args.start]),
        "qvel0": jnp.asarray(qvel[args.start]),
        # ctrl[k] advances q[k] -> q[k+1], so target is shifted by one step.
        "q_target": jnp.asarray(qpos[args.start + 1 : end + 1, qpos_idx]),
    }


def print_data_info(args: argparse.Namespace, time: np.ndarray, ctrl: np.ndarray, qpos: np.ndarray, qvel: np.ndarray, dt: float, window: dict[str, jnp.ndarray]) -> None:
    print("\nData")
    print(f"  NPZ:          {args.npz.resolve()}")
    print(f"  samples:      {len(time)}")
    print(f"  dt:           {dt:.6f} s")
    print(f"  ctrl shape:   {ctrl.shape}")
    print(f"  qpos shape:   {qpos.shape}")
    print(f"  qvel shape:   {qvel.shape}")
    print(f"  start:        {args.start}")
    print(f"  horizon:      {len(window['ctrl'])}")


###############################################################################
# Parameters
###############################################################################


def initial_param_values(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    dof_idx: int,
) -> np.ndarray:
    armature = float(model.dof_armature[dof_idx]) if args.armature is None else args.armature
    damping = float(model.dof_damping[dof_idx]) if args.damping is None else args.damping
    frictionloss = (
        float(model.dof_frictionloss[dof_idx])
        if args.frictionloss is None
        else args.frictionloss
    )

    return np.asarray(
        [
            max(armature, 1e-12),
            max(damping, 1e-12),
            max(frictionloss, 1e-12),
        ],
        dtype=np.float32,
    )


def params_from_log(log_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    params = jnp.exp(log_params)
    return params[0], params[1], params[2]


def optimize_mask(names: list[str]) -> jnp.ndarray:
    return jnp.asarray([name in names for name in PARAM_NAMES], dtype=jnp.float32)


###############################################################################
# Loss
###############################################################################


def make_q_loss(
    *,
    mjx_model_base: mjx.Model,
    mjx_data: mjx.Data,
    window: dict[str, jnp.ndarray],
    dof_idx: int,
    qpos_idx: int,
    pin_floating_base: bool,
):
    ctrl_win = window["ctrl"]
    qpos0 = window["qpos0"]
    qvel0 = window["qvel0"]
    q_target = window["q_target"]

    def loss_q(log_params: jnp.ndarray) -> jnp.ndarray:
        armature, damping, frictionloss = params_from_log(log_params)

        sim_model = mjx_model_base.replace(
            dof_armature=mjx_model_base.dof_armature.at[dof_idx].set(armature),
            dof_damping=mjx_model_base.dof_damping.at[dof_idx].set(damping),
            dof_frictionloss=mjx_model_base.dof_frictionloss.at[dof_idx].set(
                frictionloss
            ),
        )

        data0 = mjx_data.replace(qpos=qpos0, qvel=qvel0)
        if pin_floating_base:
            data0 = pin_base(data0)

        def step(data: mjx.Data, ctrl_t: jnp.ndarray):
            data = data.replace(ctrl=ctrl_t)
            data = mjx.step(sim_model, data)
            if pin_floating_base:
                data = pin_base(data)
            return data, data.qpos[qpos_idx]

        _, q_sim = jax.lax.scan(step, data0, ctrl_win)
        return jnp.mean(jnp.square(q_sim - q_target))

    return jax.jit(loss_q), jax.jit(jax.jacfwd(loss_q))


###############################################################################
# Optimizer
###############################################################################


def run_optimizer(
    *,
    args: argparse.Namespace,
    loss_fn,
    grad_fn,
    init_values: np.ndarray,
) -> tuple[float, np.ndarray]:
    log_params = jnp.asarray(np.log(init_values), dtype=jnp.float32)
    grad_mask = optimize_mask(args.optimize)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(log_params)

    best_loss = float("inf")
    best_params = init_values.copy()

    print("\nOptimizer")
    print(f"  algorithm:    Adam")
    print(f"  lr:           {args.lr}")
    print(f"  max_iter:     {args.max_iter}")
    print(f"  q loss:       mean((q_sim_after_step - q_real_next)^2)")

    for step in range(args.max_iter):
        loss_val = loss_fn(log_params)
        grad_val = grad_fn(log_params) * grad_mask

        updates, opt_state = optimizer.update(grad_val, opt_state, log_params)
        log_params = optax.apply_updates(log_params, updates)

        params = np.exp(np.asarray(log_params))
        loss_float = float(loss_val)
        if loss_float < best_loss:
            best_loss = loss_float
            best_params = params.copy()

        if step % args.log_every == 0 or step == args.max_iter - 1:
            grad_np = np.asarray(grad_val)
            print(
                f"[{step:5d}] loss={loss_float:.8e} "
                f"rms={np.sqrt(loss_float):.8e} "
                f"grad=({grad_np[0]: .3e}, {grad_np[1]: .3e}, {grad_np[2]: .3e}) "
                f"armature={params[0]:.8f} "
                f"damping={params[1]:.8f} "
                f"frictionloss={params[2]:.8f}"
            )

    return best_loss, best_params


def print_best_result(best_loss: float, best_params: np.ndarray) -> None:
    print("\nBest result on this window")
    print(f"  loss_q MSE:   {best_loss:.8e}")
    print(f"  loss_q RMS:   {np.sqrt(best_loss):.8e} rad")
    print(f"  armature:     {best_params[0]:.8f}")
    print(f"  damping:      {best_params[1]:.8f}")
    print(f"  frictionloss: {best_params[2]:.8f}")


###############################################################################
# Main
###############################################################################


def main() -> None:
    args = parse_args()

    time, ctrl, qpos, qvel, dt = load_sysid_npz(args.npz)
    model = load_mujoco_model(args, dt)
    indices = get_joint_indices(model, args.joint)
    init_values = initial_param_values(args, model, indices["dof_idx"])

    mjx_model_base = mjx.put_model(model)
    mjx_data = mjx.make_data(model)
    window = make_single_window(args, ctrl, qpos, qvel, indices["qpos_idx"])

    print("\nKangaroo q-only optimization")
    print_model_info(args, indices, init_values)
    print_data_info(args, time, ctrl, qpos, qvel, dt, window)

    loss_fn, grad_fn = make_q_loss(
        mjx_model_base=mjx_model_base,
        mjx_data=mjx_data,
        window=window,
        dof_idx=indices["dof_idx"],
        qpos_idx=indices["qpos_idx"],
        pin_floating_base=not args.free_base,
    )

    best_loss, best_params = run_optimizer(
        args=args,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        init_values=init_values,
    )
    print_best_result(best_loss, best_params)


if __name__ == "__main__":
    main()
