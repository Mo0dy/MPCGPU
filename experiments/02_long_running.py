#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    init_runner()

    knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Baseline. The settings the paper authors used for the experiments
    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.LINSYS,
            pcg_max_iters=ADAPTIVE,
            sqp_sim_period=2000
        ),
        name_prefix="baseline_time_linsys"
    )

    # NOTE: this is "kinda" cheating since the sqp_sim_period is not adaptive ...
    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.LINSYS,
            pcg_max_iters=1000,
            sqp_sim_period=2000
        ),
        name_prefix="run_through_pcg_iters_linsys"
    )

    print_experiment_header("DONE")
