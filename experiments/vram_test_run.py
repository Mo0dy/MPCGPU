#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    with expr("vram-test-run"):
        run_expr(
            knot_points=512,
            time_linsys=False,
            adaptive_max_iters=False,
            const_update_freq=True
        )
    print_experiment_header("DONE")
