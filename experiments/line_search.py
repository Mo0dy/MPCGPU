#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    init_runner()

    knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    # knot_points = [2, 4, 8]

    # Baseline. The settings the paper authors used for the experiments

    for ls_version in LineSearchMode:
        run_expr(
            knot_points,
            Settings(
                # FINE_GRAINED is also interesting.
                timing_mode=TimingMode.MINIMAL,
                # ADAPTIVE == early termination in pcg
                # same as used in the MPCGPU paper experiments
                pcg_max_iters=ADAPTIVE,
                # leave as is
                sqp_sim_period=2000,
                line_search_version=ls_version
            ),
            name_prefix=str(ls_version)
        )
    print_experiment_header("DONE")
