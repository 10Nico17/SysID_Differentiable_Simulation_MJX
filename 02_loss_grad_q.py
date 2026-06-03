"""MJX q-position loss and forward-mode gradients for Kangaroo elbow.

This is the second clean building block:

1. Load XML and converted SysID NPZ.
2. Find arm_left_4_joint indices.
3. Build an MJX rollout.
4. Compute q-only loss:

       mean((q_sim_after_step - q_real_next) ** 2)

5. Compute forward-mode gradients with respect to
   armature, damping, and frictionloss.

Run from repo root:
    python scripts/Kangaroo/02_loss_grad_q.py \
        --npz assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz \
        --horizon 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    ROOT
    / "assets/datasets/kangaroo_grippers/left_elbow_chirp_20260530_081011_sysid.npz"
)
DEFAULT_XML = ROOT / "assets/robots/kangaroo_grippers/kangaroo_grippers_mjx.xml"
JOINT_NAME = "arm_left_4_joint"

ARMATURE_DEFAULT = 0.00663065
DAMPING_DEFAULT = 1e-4
FRICTIONLOSS_DEFAULT = 1e-4


def _name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Could not find {name!r} in model.")
    return int(idx)


def _pin_base(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def _params_from_log(log_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    params = jnp.exp(log_params)
    return params[0], params[1], params[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint", default=JOINT_NAME)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--armature", type=float, default=ARMATURE_DEFAULT)
    parser.add_argument("--damping", type=float, default=DAMPING_DEFAULT)
    parser.add_argument("--frictionloss", type=float, default=FRICTIONLOSS_DEFAULT)
    parser.add_argument("--no-gravity", action="store_true", help="Debug: disable XML gravity.")
    parser.add_argument("--disable-constraints", action="store_true", help="Debug: disable equality and contact.")
    parser.add_argument("--free-base", action="store_true", help="Do not pin floating base.")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    xml_path = args.xml.resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    raw = np.load(npz_path, allow_pickle=True)

    time = raw["time"].astype(np.float32)
    ctrl = raw["ctrl"].astype(np.float32)
    qpos_real = raw["qpos"].astype(np.float32)
    qvel_real = raw["qvel"].astype(np.float32)
    dt = float(raw["dt"]) if "dt" in raw.files else float(np.median(np.diff(time)))

    model.opt.timestep = dt
    model.opt.iterations = int(args.iterations)
    if args.no_gravity:
        model.opt.gravity[:] = 0.0
    if args.disable_constraints:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EQUALITY
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, args.joint)
    qpos_idx = int(model.jnt_qposadr[joint_id])
    dof_idx = int(model.jnt_dofadr[joint_id])
    qvel_idx = dof_idx
    act_idx = _name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.joint)

    if args.start < 0:
        raise ValueError("--start must be >= 0")
    end = min(args.start + args.horizon, len(ctrl) - 1)
    if end <= args.start:
        raise ValueError("Empty rollout window. Check --start and --horizon.")

    ctrl_win = jnp.asarray(ctrl[args.start:end])
    qpos0 = jnp.asarray(qpos_real[args.start])
    qvel0 = jnp.asarray(qvel_real[args.start])
    q_target = jnp.asarray(qpos_real[args.start + 1 : end + 1, qpos_idx])

    mjx_model_base = mjx.put_model(model)
    mjx_data = mjx.make_data(model)

    def loss_q_fn(log_params: jnp.ndarray) -> jnp.ndarray:
        armature, damping, frictionloss = _params_from_log(log_params)
        sim_model = mjx_model_base.replace(
            dof_armature=mjx_model_base.dof_armature.at[dof_idx].set(armature),
            dof_damping=mjx_model_base.dof_damping.at[dof_idx].set(damping),
            dof_frictionloss=mjx_model_base.dof_frictionloss.at[dof_idx].set(
                frictionloss
            ),
        )

        data0 = mjx_data.replace(qpos=qpos0, qvel=qvel0)
        if not args.free_base:
            data0 = _pin_base(data0)

        def step(data: mjx.Data, ctrl_t: jnp.ndarray):
            data = data.replace(ctrl=ctrl_t)
            data = mjx.step(sim_model, data)
            if not args.free_base:
                data = _pin_base(data)
            return data, data.qpos[qpos_idx]

        _, q_sim = jax.lax.scan(step, data0, ctrl_win)
        return jnp.mean(jnp.square(q_sim - q_target))

    loss_jit = jax.jit(loss_q_fn)
    grad_jit = jax.jit(jax.jacfwd(loss_q_fn))

    init_params = np.asarray(
        [
            max(args.armature, 1e-12),
            max(args.damping, 1e-12),
            max(args.frictionloss, 1e-12),
        ],
        dtype=np.float32,
    )
    log_params = jnp.asarray(np.log(init_params), dtype=jnp.float32)

    loss_q = float(loss_jit(log_params))
    grad_log = np.asarray(grad_jit(log_params))
    grad_param = grad_log / init_params

    print("\nKangaroo q-loss and forward-mode gradient")
    print(f"  XML:          {xml_path}")
    print(f"  NPZ:          {npz_path}")
    print(f"  joint:        {args.joint}")
    print(f"  act_idx:      {act_idx}")
    print(f"  qpos_idx:     {qpos_idx}")
    print(f"  qvel_idx:     {qvel_idx}")
    print(f"  dof_idx:      {dof_idx}")
    print("")
    print("Window")
    print(f"  start:        {args.start}")
    print(f"  horizon:      {len(ctrl_win)}")
    print(f"  dt:           {dt:.6f} s")
    print(f"  gravity:      {'off' if args.no_gravity else 'on'}")
    print(f"  constraints:  {'off' if args.disable_constraints else 'on'}")
    print(f"  base:         {'free' if args.free_base else 'pinned'}")
    print("")
    print("q-only loss")
    print(f"  loss_q MSE:   {loss_q:.8e}")
    print(f"  loss_q RMS:   {np.sqrt(loss_q):.8e} rad")
    print("")
    print("Parameters and gradients")
    print("  name           value          dL/dlog(param)   dL/dparam")
    print(
        f"  armature       {init_params[0]: .8e}  "
        f"{grad_log[0]: .8e}  {grad_param[0]: .8e}"
    )
    print(
        f"  damping        {init_params[1]: .8e}  "
        f"{grad_log[1]: .8e}  {grad_param[1]: .8e}"
    )
    print(
        f"  frictionloss   {init_params[2]: .8e}  "
        f"{grad_log[2]: .8e}  {grad_param[2]: .8e}"
    )


if __name__ == "__main__":
    main()
