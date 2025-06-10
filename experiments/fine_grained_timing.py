#!/usr/bin/env python3


# Runs the experiment with fine-grained and linsys timing.
# Also runs a baseline with minimal instrumentation to estimate other overhead (memory copies, etc.)

from runner import *

if __name__ == "__main__":
    knot_points = [2, 4, 8, 16]
    with expr("baseline"):
        run_expr(
            knot_points=knot_points,
            time_linsys=False,
            adaptive_max_iters=True,
            fine_grained_timing=False,
            const_update_freq=True,
            run_qdldl=False
        )

    with expr("fine_grained"):
        run_expr(
            knot_points=knot_points,
            time_linsys=True,
            adaptive_max_iters=True,
            fine_grained_timing=True,
            const_update_freq=True,
            run_qdldl=False
        )
