#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    knot_points = [2, 4, 8, 16]
    with expr("with_precond"):
        run_expr(
            knot_points=knot_points,
            time_linsys=True,
            fine_grained_timing=True,
            adaptive_max_iters=False,
            const_update_freq=True
        )

    with expr("without_precond"):
        run_expr(
            knot_points=knot_points,
            time_linsys=True,
            fine_grained_timing=True,
            adaptive_max_iters=False,
            const_update_freq=True,
            enable_preconditioning=False
        )
