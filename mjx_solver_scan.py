"""Runtime patch for reverse-mode autodiff through the MJX constraint solver.

MuJoCo 3.9.0 still calls ``jax.lax.while_loop`` inside
``mujoco.mjx._src.solver.solve`` when solver iterations are greater than one.
Reverse-mode autodiff cannot backpropagate through that dynamic loop. The
upstream workaround is to run the solver loop through a scan-based equivalent.

This module patches only the current Python process. It does not modify the
installed MuJoCo package in site-packages.
"""

from __future__ import annotations


def enable_solver_scan() -> None:
    """Patch ``mujoco.mjx`` so solver iterations use scan instead of while_loop."""
    import jax
    import mujoco
    from mujoco.mjx._src import math
    from mujoco.mjx._src import solver
    from mujoco.mjx._src.types import DisableBit, SolverType

    if getattr(solver.solve, "_sysid_solver_scan_patch", False):
        return

    jp = solver.jp
    Context = solver.Context
    OptionJAX = solver.OptionJAX

    def solve(m, d):
        """Finds forces that satisfy constraints using a scan-based loop."""
        if not isinstance(m.opt._impl, OptionJAX):
            raise ValueError("solve requires JAX backend implementation.")

        def cond(ctx):
            improvement = solver._rescale(m, ctx.prev_cost - ctx.cost)
            gradient = solver._rescale(m, math.norm(ctx.grad))

            done = ctx.solver_niter >= m.opt.iterations
            done |= improvement < m.opt.tolerance
            done |= gradient < m.opt.tolerance
            return ~done

        def body(ctx):
            ctx = solver._linesearch(m, d, ctx)
            prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad
            ctx = solver._update_constraint(m, d, ctx)
            ctx = solver._update_gradient(m, d, ctx)

            if m.opt.solver == SolverType.NEWTON:
                search = -ctx.Mgrad
            else:
                beta = jp.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
                beta = beta / jp.maximum(mujoco.mjMINVAL, jp.dot(prev_grad, prev_Mgrad))
                beta = jp.maximum(0, beta)
                search = -ctx.Mgrad + beta * ctx.search
            return ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

        qacc = d.qacc_smooth
        if not m.opt.disableflags & DisableBit.WARMSTART:
            warm = Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
            smth = Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
            qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
        d = d.replace(qacc=qacc)

        ctx = Context.create(m, d)
        if m.opt.iterations == 1:
            ctx = body(ctx)
        else:
            ctx = solver._while_loop_scan(cond, body, ctx, m.opt.iterations)

        return d.tree_replace(
            {
                "qfrc_constraint": ctx.qfrc_constraint,
                "qacc": ctx.qacc,
                "_impl.efc_force": ctx.efc_force,
            }
        )

    solve._sysid_solver_scan_patch = True
    solver.solve = solve
