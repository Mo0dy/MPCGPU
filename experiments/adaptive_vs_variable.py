#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    with expr("sim-time=2000_max-iters=10000"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            adaptive_max_iters=False,
            const_update_freq=True
        )

    # NOTE: way too many max iters
    with expr("sim-time=2000_adaptive-max-iters"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            adaptive_max_iters=False,
            max_iters=10000,
            const_update_freq=False
        )

    with expr("sim-time=2000_variable-freq_adaptive-max-iters"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            adaptive_max_iters=True,
            const_update_freq=False
        )

    print_experiment_header("DONE")
