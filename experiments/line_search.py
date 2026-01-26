#!/usr/bin/env python3

from runner import *
from itertools import product

if __name__ == "__main__":
    init_runner()

    #knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    # knot_points = [2, 4, 8]
    #Felix wanted 32 for reasonable results:
    knot_points = [16]
    # Baseline. The settings the paper authors used for the experiments

    for ls_version, num_alphas in product(LineSearchMode, [8, 16]):
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
                line_search_version=ls_version,
                num_alphas=num_alphas
            ),
            name_prefix=f"{str(ls_version)}_{num_alphas}",
            run_qdldl=False
        )
    print_experiment_header("DONE")




















