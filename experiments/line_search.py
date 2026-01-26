#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    init_runner()

    #knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    # knot_points = [2, 4, 8]
    #Felix wanted 32 for reasonable results:
    knot_points = [16]
    # Baseline. The settings the paper authors used for the experiments

    for ls_version in LineSearchMode:

        # this is set, that we only get the new line search methods in our experiment. (numbers: 12,13,14,15)
        # if (ls_version < 12):
        #    continue
        #till here

        run_expr(
            knot_points,
            Settings(
                # FINE_GRAINED is also interesting.
                timing_mode=TimingMode.FINE_GRAINED,
                # ADAPTIVE == early termination in pcg
                # same as used in the MPCGPU paper experiments
                pcg_max_iters=ADAPTIVE,
                # leave as is
                sqp_sim_period=2000,
                sqp_max_time_us=2000,
                line_search_version=ls_version
            ),
            name_prefix=str(ls_version),
            run_qdldl=False
        )
    print_experiment_header("DONE")
