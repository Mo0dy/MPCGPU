#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    knot_points = [2, 8, 16, 32, 64, 128, 256, 512]
    with expr("baseline_with_precond_sim-time=2000"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            adaptive_max_iters=False,
            const_update_freq=True
        )

    with expr("without_precond_sim-time=2000"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            adaptive_max_iters=False,
            const_update_freq=True,
            enable_preconditioning=False
        )


    with expr("pcg_iters=200"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            pcg_max_iters=200,
            adaptive_max_iters=False,
            const_update_freq=False
        )

    with expr("sim-time=2000_adaptive-max-iters"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            adaptive_max_iters=True,
            const_update_freq=True
        )

    with expr("sim-time=2000_fine-grained-timing"):
        run_expr(
            knot_points=knot_points,
            time_linsys=True,
            fine_grained_timing=True,
            const_update_freq=True,
            adaptive_max_iters=True,
        )
    print_experiment_header("DONE")
