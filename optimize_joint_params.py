"""Optimize Kangaroo joint and/or body parameters with a q-only MJX loss.

Standalone script. Does not use the repo task registry.

Configure which parameters to optimize by editing ACTUATOR_PARAM_NAMES and
BODY_PARAM_NAMES at the top of this file. Initial values are read from the XML.

Run from mjx_sysid-main/:
    python scripts/Kangaroo/optimize_joint_params.py \
        --npz scripts/Kangaroo/datasets/Arms_sysid.npz \
        --horizon 100 \
        --batch-size 32 \
        --max-iter 300 \
        --lr 0.002
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from mujoco import mjx

from mjx_solver_scan import enable_solver_scan


KANGAROO_DIR = Path(__file__).resolve().parent
ROOT = KANGAROO_DIR.parents[1]
DATASET_DIR = KANGAROO_DIR / "datasets"
DEFAULT_NPZ = DATASET_DIR / "Arms_sysid.npz"
DEFAULT_XML = KANGAROO_DIR / "Robot/kangaroo_grippers_mjx.xml"

LEFT_ARM_JOINTS = tuple(f"arm_left_{i}_joint" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"arm_right_{i}_joint" for i in range(1, 8))
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
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
LEG_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS
PARAM_CHOICES = ("armature", "damping", "frictionloss", "mass", "inertia", "ipos")

# Joint/actuator parameters — indexed via dof_idx.
ACTUATOR_PARAM_NAMES: tuple[str, ...] = ("armature", "damping", "frictionloss")

# Body (link) parameters — indexed via body_idx (body the joint is attached to).
BODY_PARAM_NAMES: tuple[str, ...] = ("mass", "inertia", "ipos")

# Log-space parameters must start positive. If XML value is zero, use this.
POSITIVE_INIT_FLOOR: dict[str, float] = {
    "damping": 1e-4,
    "frictionloss": 1e-4,
}



###############################################################################
# Parameter registry
###############################################################################
# Each entry: (mjx model field name, number of scalars, use log-space)
# Log-space is used for strictly positive parameters.
# ipos (CoM offset) can be negative → direct space.

_DOF_FIELDS: dict[str, tuple[str, int, bool]] = {
    "armature":     ("dof_armature",     1, True),
    "damping":      ("dof_damping",      1, True),
    "frictionloss": ("dof_frictionloss", 1, True),
}

_BODY_FIELDS: dict[str, tuple[str, int, bool]] = {
    "mass":    ("body_mass",    1, True),
    "inertia": ("body_inertia", 3, True),
    "ipos":    ("body_ipos",    3, False),
}


def build_param_meta_for_joints(
    joint_infos: list[dict],
    actuator_params: tuple[str, ...],
    body_params: tuple[str, ...],
) -> list[dict]:
    """Build metadata that maps a flat raw-param vector to MJX model fields."""
    meta: list[dict] = []
    offset = 0
    for info in joint_infos:
        joint_name = info["joint_name"]
        for param_name in actuator_params:
            field, size, log = _DOF_FIELDS[param_name]
            meta.append(
                dict(
                    name=f"{joint_name}/{param_name}",
                    param_name=param_name,
                    joint_name=joint_name,
                    field=field,
                    idx=info["dof_idx"],
                    size=size,
                    log=log,
                    slice=(offset, offset + size),
                )
            )
            offset += size
        for param_name in body_params:
            field, size, log = _BODY_FIELDS[param_name]
            meta.append(
                dict(
                    name=f"{joint_name}/{param_name}",
                    param_name=param_name,
                    joint_name=joint_name,
                    body_name=info["body_name"],
                    field=field,
                    idx=info["body_idx"],
                    size=size,
                    log=log,
                    slice=(offset, offset + size),
                )
            )
            offset += size
    return meta


def extract_raw_params(model: mujoco.MjModel, meta: list[dict]) -> np.ndarray:
    """Read physical values from model and convert to raw (log or direct) space."""
    parts = []
    for p in meta:
        val = np.asarray(getattr(model, p["field"])[p["idx"]], dtype=np.float64).reshape(-1)
        if p["log"]:
            floor = POSITIVE_INIT_FLOOR.get(p["param_name"], 1e-12)
            val = np.log(np.maximum(val, floor))
        parts.append(val.astype(np.float32))
    return np.concatenate(parts)


def apply_params_to_model(
    model_base: mjx.Model,
    raw_params: jnp.ndarray,
    meta: list[dict],
) -> mjx.Model:
    """Apply flat raw-param vector to MJX model. JIT-safe (meta is static)."""
    field_arrays: dict[str, jnp.ndarray] = {}
    for p in meta:
        raw = raw_params[p["slice"][0] : p["slice"][1]]
        phys = jnp.exp(raw) if p["log"] else raw
        if p["size"] == 1:
            phys = phys[0]
        base = field_arrays.get(p["field"], getattr(model_base, p["field"]))
        field_arrays[p["field"]] = base.at[p["idx"]].set(phys)
    return model_base.replace(**field_arrays)


def raw_to_physical(raw_params: np.ndarray, meta: list[dict]) -> dict[str, np.ndarray]:
    result = {}
    for p in meta:
        raw = raw_params[p["slice"][0] : p["slice"][1]]
        result[p["name"]] = np.exp(raw) if p["log"] else raw.copy()
    return result


###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--out-xml",
        type=Path,
        default=None,
        help="Where to write the optimized XML. Defaults to <xml stem>_sysid.xml.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Debug: use one fixed fragment start instead of random batches.",
    )
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument(
        "--arm-side",
        choices=("left", "right", "both"),
        default="both",
        help="Which arm joints to include in the loss/optimization. Ignored when --joint is set.",
    )
    parser.add_argument(
        "--joint-group",
        choices=("arm", "leg"),
        default="arm",
        help="Which joint family to include in the loss/optimization. Ignored when --joint is set.",
    )
    parser.add_argument(
        "--leg-side",
        choices=("left", "right", "both"),
        default="both",
        help="Which leg joints to include in the loss/optimization. Ignored unless --joint-group leg.",
    )
    parser.add_argument(
        "--joint",
        default=None,
        help="Optimize only one joint, for example arm_left_4_joint.",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        choices=PARAM_CHOICES,
        default=list(ACTUATOR_PARAM_NAMES) + list(BODY_PARAM_NAMES),
        help="Parameter names to optimize. Example: --params armature",
    )
    parser.add_argument("--reg", type=float, default=0.0,
                        help="L2 regularization weight (keeps params near XML values).")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-gravity", action="store_true")
    parser.add_argument("--disable-constraints", action="store_true")
    parser.add_argument("--free-base", action="store_true", help="Let the floating base evolve freely; disables base tracking.")
    parser.add_argument("--pin-base", action="store_true", help="Debug: pin floating base to a fixed pose.")
    parser.add_argument(
        "--track-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each MJX step, overwrite floating-base qpos/qvel from measured data.",
    )
    parser.add_argument(
        "--track-nondriven",
        choices=("none", "real"),
        default="none",
        help="Track non-optimized joints from measured qpos/qvel after each MJX step.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--grad-mode",
        choices=("reverse", "forward"),
        default="reverse",
        help="Use reverse-mode value_and_grad or forward-mode jacfwd.",
    )
    parser.add_argument(
        "--solver-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch MJX solver while_loop to scan for reverse-mode autodiff.",
    )
    return parser.parse_args()


def arm_joint_names(side: str) -> tuple[str, ...]:
    if side == "left":
        return LEFT_ARM_JOINTS
    if side == "right":
        return RIGHT_ARM_JOINTS
    return ARM_JOINTS


def leg_joint_names(side: str) -> tuple[str, ...]:
    if side == "left":
        return LEFT_LEG_JOINTS
    if side == "right":
        return RIGHT_LEG_JOINTS
    return LEG_JOINTS


def selected_joint_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.joint is not None:
        return (args.joint,)
    if args.joint_group == "leg":
        return leg_joint_names(args.leg_side)
    return arm_joint_names(args.arm_side)


def split_param_names(params: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    actuator_params = tuple(p for p in params if p in _DOF_FIELDS)
    body_params = tuple(p for p in params if p in _BODY_FIELDS)
    return actuator_params, body_params


###############################################################################
# Model
###############################################################################


def name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Could not find {name!r} in model.")
    return int(idx)


def load_mujoco_model(args: argparse.Namespace, dt: float) -> mujoco.MjModel:
    if args.solver_scan:
        enable_solver_scan()

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
    body_idx = int(model.jnt_bodyid[joint_id])
    return {
        "joint_name": joint_name,
        "joint_id": joint_id,
        "act_idx":  act_idx,
        "qpos_idx": int(model.jnt_qposadr[joint_id]),
        "qvel_idx": dof_idx,
        "dof_idx":  dof_idx,
        "body_idx": body_idx,
        "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_idx) or "",
    }


def get_joint_group_indices(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[dict]:
    return [get_joint_indices(model, joint_name) for joint_name in joint_names]


def tracked_nondriven_indices(
    model: mujoco.MjModel,
    driven_joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    driven = set(driven_joint_names)
    qpos_idx: list[int] = []
    qvel_idx: list[int] = []
    tracked_names: list[str] = []

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name or joint_name == "origin" or joint_name in driven:
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
        tracked_names.append(joint_name)

    return (
        np.asarray(qpos_idx, dtype=np.int32),
        np.asarray(qvel_idx, dtype=np.int32),
        tracked_names,
    )


def pin_base(data: mjx.Data) -> mjx.Data:
    qpos = data.qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.9]))
    qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = data.qvel.at[0:6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)


def track_base(data: mjx.Data, qpos_ref: jnp.ndarray, qvel_ref: jnp.ndarray) -> mjx.Data:
    qpos = data.qpos.at[0:7].set(qpos_ref[0:7])
    qvel = data.qvel.at[0:6].set(qvel_ref[0:6])
    return data.replace(qpos=qpos, qvel=qvel)


def track_indices(
    data: mjx.Data,
    qpos_ref: jnp.ndarray,
    qvel_ref: jnp.ndarray,
    qpos_idx: jnp.ndarray,
    qvel_idx: jnp.ndarray,
) -> mjx.Data:
    qpos = data.qpos.at[qpos_idx].set(qpos_ref[qpos_idx])
    qvel = data.qvel.at[qvel_idx].set(qvel_ref[qvel_idx])
    return data.replace(qpos=qpos, qvel=qvel)


def print_model_info(
    args: argparse.Namespace,
    joint_infos: list[dict],
    phys: dict[str, np.ndarray],
    tracked_names: list[str],
) -> None:
    print("\nModel")
    print(f"  XML:          {args.xml.resolve()}")
    if args.joint is not None:
        joint_scope = args.joint
    elif args.joint_group == "leg":
        joint_scope = f"{args.leg_side} leg{'s' if args.leg_side == 'both' else ''}"
    else:
        joint_scope = f"{args.arm_side} arm{'s' if args.arm_side == 'both' else ''}"
    print(f"  joint_scope:  {joint_scope}")
    print(f"  joints:       {len(joint_infos)}")
    print("  joint                       act  qpos  qvel  dof   body")
    for info in joint_infos:
        print(
            f"  {info['joint_name']:<27} {info['act_idx']:>3d}  "
            f"{info['qpos_idx']:>4d}  {info['qvel_idx']:>4d}  "
            f"{info['dof_idx']:>3d}   {info['body_name']}"
        )
    print(f"  gravity:      {'off' if args.no_gravity else 'on'}")
    print(f"  constraints:  {'off' if args.disable_constraints else 'on'}")
    if args.pin_base:
        base_mode = "pinned"
    elif args.free_base:
        base_mode = "free"
    elif args.track_base:
        base_mode = "tracked from dataset"
    else:
        base_mode = "not pinned/not tracked"
    print(f"  base:         {base_mode}")
    print(f"  nondriven:    {args.track_nondriven}")
    if tracked_names:
        print(f"  tracked dofs: {len(tracked_names)} joints")
    print(f"  grad_mode:    {args.grad_mode}")
    print(f"  solver_scan:  {'on' if args.solver_scan else 'off'}")
    print(f"  optimize:     {', '.join(args.params)}")
    print("  start params:")
    _print_phys(phys)


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


def max_start_index(ctrl: np.ndarray, horizon: int) -> int:
    max_start = len(ctrl) - horizon - 1
    if max_start < 0:
        raise ValueError(f"Dataset too short for horizon={horizon}: len(ctrl)={len(ctrl)}")
    return max_start


def make_dataset_arrays(
    ctrl: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> dict[str, jnp.ndarray]:
    return {
        "ctrl": jnp.asarray(ctrl),
        "qpos": jnp.asarray(qpos),
        "qvel": jnp.asarray(qvel),
    }


def print_data_info(
    args: argparse.Namespace,
    time: np.ndarray,
    ctrl: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    dt: float,
) -> None:
    print("\nData")
    print(f"  NPZ:          {args.npz.resolve()}")
    print(f"  samples:      {len(time)}")
    print(f"  dt:           {dt:.6f} s")
    print(f"  ctrl shape:   {ctrl.shape}")
    print(f"  qpos shape:   {qpos.shape}")
    print(f"  qvel shape:   {qvel.shape}")
    if args.start is None:
        print("  sampling:     random trajectory fragments")
    else:
        print(f"  sampling:     fixed start={args.start}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  eval_batch:   {args.eval_batch_size}")
    print(f"  horizon:      {args.horizon}")


###############################################################################
# Loss
###############################################################################


def make_q_loss(
    *,
    mjx_model_base: mjx.Model,
    mjx_data: mjx.Data,
    dataset: dict[str, jnp.ndarray],
    meta: list[dict],
    horizon: int,
    qpos_indices: np.ndarray,
    pin_floating_base: bool,
    track_floating_base: bool,
    track_nondriven: bool,
    nondriven_qpos_indices: np.ndarray,
    nondriven_qvel_indices: np.ndarray,
    raw_params_init: jnp.ndarray,
    reg: float,
    grad_mode: str,
):
    ctrl_all = dataset["ctrl"]
    qpos_all = dataset["qpos"]
    qvel_all = dataset["qvel"]
    offsets = jnp.arange(horizon, dtype=jnp.int32)
    qpos_idx_j = jnp.asarray(qpos_indices, dtype=jnp.int32)
    nondriven_qpos_idx_j = jnp.asarray(nondriven_qpos_indices, dtype=jnp.int32)
    nondriven_qvel_idx_j = jnp.asarray(nondriven_qvel_indices, dtype=jnp.int32)

    raw_init_j = jnp.asarray(raw_params_init)

    def loss_q(raw_params: jnp.ndarray, starts: jnp.ndarray) -> jnp.ndarray:
        sim_model = apply_params_to_model(mjx_model_base, raw_params, meta)

        def fragment_loss(start: jnp.ndarray) -> jnp.ndarray:
            idx = start + offsets
            ctrl_win = ctrl_all[idx]
            q_target = qpos_all[idx + 1][:, qpos_idx_j]
            qpos_base_target = qpos_all[idx + 1]
            qvel_base_target = qvel_all[idx + 1]
            data0 = mjx_data.replace(qpos=qpos_all[start], qvel=qvel_all[start])
            if pin_floating_base:
                data0 = pin_base(data0)
            if track_nondriven:
                data0 = track_indices(
                    data0,
                    qpos_all[start],
                    qvel_all[start],
                    nondriven_qpos_idx_j,
                    nondriven_qvel_idx_j,
                )

            def step(data: mjx.Data, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
                ctrl_t, qpos_ref, qvel_ref = inputs
                data = data.replace(ctrl=ctrl_t)
                if track_nondriven:
                    data = track_indices(
                        data,
                        qpos_ref,
                        qvel_ref,
                        nondriven_qpos_idx_j,
                        nondriven_qvel_idx_j,
                    )
                data = mjx.step(sim_model, data)
                if pin_floating_base:
                    data = pin_base(data)
                if track_floating_base:
                    data = track_base(data, qpos_ref, qvel_ref)
                if track_nondriven:
                    data = track_indices(
                        data,
                        qpos_ref,
                        qvel_ref,
                        nondriven_qpos_idx_j,
                        nondriven_qvel_idx_j,
                    )
                return data, data.qpos[qpos_idx_j]

            # Rematerialize MJX steps during reverse-mode to avoid storing the
            # full solver state for every timestep.
            _, q_sim = jax.lax.scan(
                jax.checkpoint(step),
                data0,
                (ctrl_win, qpos_base_target, qvel_base_target),
            )
            return jnp.mean(jnp.square(q_sim - q_target))

        # Evaluate batch fragments sequentially. This uses less peak memory than
        # vmap for reverse-mode through constrained MJX rollouts.
        def batch_step(loss_sum: jnp.ndarray, start: jnp.ndarray):
            return loss_sum + fragment_loss(start), None

        loss_sum, _ = jax.lax.scan(
            batch_step,
            jnp.asarray(0.0, dtype=raw_params.dtype),
            starts,
        )
        q_loss = loss_sum / starts.shape[0]
        reg_loss = jnp.mean(jnp.square(raw_params - raw_init_j))
        return q_loss + reg * reg_loss

    loss_fn = jax.jit(loss_q)
    if grad_mode == "reverse":
        value_and_grad = jax.jit(jax.value_and_grad(loss_q))

        def grad_q(raw_params: jnp.ndarray, starts: jnp.ndarray) -> jnp.ndarray:
            _, grad = value_and_grad(raw_params, starts)
            return grad
    else:
        grad_q = jax.jacfwd(loss_q)

    return loss_fn, jax.jit(grad_q)


###############################################################################
# Optimizer
###############################################################################


def _format_value(val: np.ndarray) -> str:
    val = np.asarray(val).reshape(-1)
    if val.size == 1:
        return f"{val[0]:.6g}"
    return "[" + ", ".join(f"{v:.4g}" for v in val) + "]"


def _print_phys(phys: dict[str, np.ndarray]) -> None:
    by_joint: dict[str, dict[str, np.ndarray]] = {}
    param_order = [p for p in PARAM_CHOICES if any(name.endswith(f"/{p}") for name in phys)]

    for name, val in phys.items():
        joint_name, param_name = name.rsplit("/", 1)
        by_joint.setdefault(joint_name, {})[param_name] = val

    joint_width = max([len("joint"), *(len(joint) for joint in by_joint)])
    col_widths = {
        param: max(
            len(param),
            *(
                len(_format_value(params[param]))
                for params in by_joint.values()
                if param in params
            ),
        )
        for param in param_order
    }

    header = f"    {'joint':<{joint_width}}  " + "  ".join(
        f"{param:>{col_widths[param]}}" for param in param_order
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for joint_name, params in by_joint.items():
        row = f"    {joint_name:<{joint_width}}  " + "  ".join(
            f"{_format_value(params[param]):>{col_widths[param]}}"
            if param in params
            else " " * col_widths[param]
            for param in param_order
        )
        print(row)


def run_optimizer(
    *,
    args: argparse.Namespace,
    loss_fn,
    grad_fn,
    raw_params_init: np.ndarray,
    meta: list[dict],
    max_start: int,
) -> tuple[float, dict[str, np.ndarray]]:
    raw_params = jnp.asarray(raw_params_init, dtype=jnp.float32)
    rng = jax.random.PRNGKey(args.seed)
    eval_starts = jax.random.randint(
        jax.random.PRNGKey(args.seed + 10_000),
        shape=(args.eval_batch_size,),
        minval=0,
        maxval=max_start + 1,
        dtype=jnp.int32,
    )
    fixed_starts = (
        jnp.full((args.batch_size,), args.start, dtype=jnp.int32)
        if args.start is not None
        else None
    )

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(raw_params)

    best_loss = float("inf")
    best_raw = raw_params_init.copy()

    print("\nOptimizer")
    print(f"  algorithm:    Adam")
    print(f"  lr:           {args.lr}")
    print(f"  max_iter:     {args.max_iter}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  eval_batch:   {args.eval_batch_size}")
    print(f"  horizon:      {args.horizon}")
    print(f"  seed:         {args.seed}")
    print(f"  q loss:       mean((q_sim_after_step - q_real_next)^2)")

    for step in range(args.max_iter):
        if fixed_starts is None:
            rng, batch_key = jax.random.split(rng)
            starts = jax.random.randint(
                batch_key,
                shape=(args.batch_size,),
                minval=0,
                maxval=max_start + 1,
                dtype=jnp.int32,
            )
        else:
            starts = fixed_starts

        loss_val = loss_fn(raw_params, starts)
        grad_val = grad_fn(raw_params, starts)

        updates, opt_state = optimizer.update(grad_val, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)

        if step % args.log_every == 0 or step == args.max_iter - 1:
            eval_loss = float(loss_fn(raw_params, eval_starts))
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_raw = np.asarray(raw_params).copy()
            print(
                f"iter {step + 1:>{len(str(args.max_iter))}d}/{args.max_iter}  "
                f"train_loss={float(loss_val):.6e}  "
                f"eval_loss={eval_loss:.6e}  "
                f"eval_rmse={np.sqrt(eval_loss):.6e} rad"
            )

    return best_loss, raw_to_physical(best_raw, meta)


###############################################################################
# XML export
###############################################################################


def save_optimized_xml(
    xml_path: Path,
    meta: list[dict],
    best_phys: dict[str, np.ndarray],
    out_path: Path | None = None,
) -> Path:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    body_by_name = {
        body.get("name"): body
        for body in root.iter("body")
        if body.get("name") is not None
    }

    for p in meta:
        key = p["name"]
        if key not in best_phys:
            continue
        param_name = p["param_name"]
        value = np.asarray(best_phys[key]).reshape(-1)

        if param_name in ("armature", "damping", "frictionloss"):
            joint_el = root.find(f".//*joint[@name='{p['joint_name']}']")
            if joint_el is None:
                raise ValueError(f"Joint {p['joint_name']!r} not found in XML.")
            joint_el.set(param_name, f"{float(value[0]):.8f}")
            continue

        body = body_by_name.get(p.get("body_name", ""))
        if body is None:
            continue
        inertial = body.find("inertial")
        if inertial is None:
            continue
        if param_name == "mass":
            inertial.set("mass", f"{float(value[0]):.8f}")
        elif param_name == "inertia":
            inertial.set("diaginertia", f"{value[0]:.8f} {value[1]:.8f} {value[2]:.8f}")
        elif param_name == "ipos":
            inertial.set("pos", f"{value[0]:.8f} {value[1]:.8f} {value[2]:.8f}")

    if out_path is None:
        out_path = xml_path.parent / (xml_path.stem + "_sysid.xml")
    ET.indent(tree, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="unicode", xml_declaration=False)
    return out_path


###############################################################################
# Main
###############################################################################


def main() -> None:
    args = parse_args()
    if args.pin_base and args.free_base:
        raise ValueError("Use either --pin-base or --free-base, not both.")
    actuator_params, body_params = split_param_names(args.params)
    if not actuator_params and not body_params:
        raise ValueError("No optimizable parameters selected.")

    time, ctrl, qpos, qvel, dt = load_sysid_npz(args.npz)
    model = load_mujoco_model(args, dt)
    joint_names = selected_joint_names(args)
    joint_infos = get_joint_group_indices(model, joint_names)
    qpos_indices = np.asarray([info["qpos_idx"] for info in joint_infos], dtype=np.int32)
    nondriven_qpos_indices, nondriven_qvel_indices, tracked_names = tracked_nondriven_indices(
        model,
        joint_names,
    )

    meta = build_param_meta_for_joints(
        joint_infos=joint_infos,
        actuator_params=actuator_params,
        body_params=body_params,
    )
    raw_params_init = extract_raw_params(model, meta)
    phys_init = raw_to_physical(raw_params_init, meta)

    mjx_model_base = mjx.put_model(model)
    mjx_data = mjx.make_data(model)
    dataset = make_dataset_arrays(ctrl, qpos, qvel)
    max_start = max_start_index(ctrl, args.horizon)

    print("\nKangaroo q-only optimization")
    print_model_info(args, joint_infos, phys_init, tracked_names if args.track_nondriven == "real" else [])
    print_data_info(args, time, ctrl, qpos, qvel, dt)

    loss_fn, grad_fn = make_q_loss(
        mjx_model_base=mjx_model_base,
        mjx_data=mjx_data,
        dataset=dataset,
        meta=meta,
        horizon=args.horizon,
        qpos_indices=qpos_indices,
        pin_floating_base=args.pin_base,
        track_floating_base=(args.track_base and not args.free_base and not args.pin_base),
        track_nondriven=(args.track_nondriven == "real"),
        nondriven_qpos_indices=nondriven_qpos_indices,
        nondriven_qvel_indices=nondriven_qvel_indices,
        raw_params_init=raw_params_init,
        reg=args.reg,
        grad_mode=args.grad_mode,
    )

    best_loss, best_phys = run_optimizer(
        args=args,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        raw_params_init=raw_params_init,
        meta=meta,
        max_start=max_start,
    )

    print("\nBest result")
    print(f"  loss MSE:     {best_loss:.8e}")
    print(f"  loss RMS:     {np.sqrt(best_loss):.8e} rad")
    _print_phys(best_phys)

    out_xml = save_optimized_xml(args.xml.resolve(), meta, best_phys, args.out_xml)
    print(f"\nSaved optimized XML → {out_xml}")


if __name__ == "__main__":
    main()
