#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    init_runner()

    #knot_points = [2, 4, 8, 16, 32, 64, 128, 256]
    knot_points = [16, 32, 128]
    #knot_points = [128]
    # knot_points = [2, 4, 8]

    # Baseline. The settings the paper authors used for the experiments

    for ls_version in LineSearchMode:
        run_expr(
            knot_points,
            Settings(
                # FINE_GRAINED is also interesting.
                timing_mode=TimingMode.FINE_GRAINED,
                # ADAPTIVE == early termination in pcg
                # same as used in the MPCGPU paper experiments
                pcg_max_iters=ADAPTIVE,
                # solve until convergence --> bad solves lead to bad results.
                # Note that sim period is different to sqp max time. This is unphysical but needed for the experiments. It guarantees that the sqp problems are not degraded by bad tracking.
                sqp_sim_period=2000, # 2ms simulation period
                sqp_max_time_us=100000, # 100ms. --> no early stopping
                line_search_version=ls_version
            ),
            name_prefix=str(ls_version),
            run_qdldl=False
        )
    print_experiment_header("DONE")
