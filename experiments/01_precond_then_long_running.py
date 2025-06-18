#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    init_runner()

    knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Baseline. The settings the paper authors used for the experiments
    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.MINIMAL,
            pcg_max_iters=ADAPTIVE,
            sqp_sim_period=2000
        ),
        name_prefix="baseline"
    )

    # Fine grained timing
    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.FINE_GRAINED,
            pcg_max_iters=ADAPTIVE,
            sqp_sim_period=2000
        ),
        name_prefix="fine_grained"
    )

    # To check effect of preconditioning we should disable adaptive max iters and either increase or disable the sqp_sim_period ...
    # Baseline:
    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.FINE_GRAINED,
            pcg_max_iters=200,
            sqp_sim_period=ADAPTIVE,
            enable_preconditioning=True,
        ),
        name_prefix="precond"
    )

    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.FINE_GRAINED,
            pcg_max_iters=200,
            sqp_sim_period=ADAPTIVE,
            enable_preconditioning=False,
        ),
        name_prefix="no_precond"
    )

    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.MINIMAL,
            pcg_max_iters=200,
            sqp_sim_period=ADAPTIVE,
        ),
        name_prefix="run_through_pcg_iters"
    )
    print_experiment_header("DONE")
