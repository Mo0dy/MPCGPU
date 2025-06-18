#!/usr/bin/env python3

from runner import *

if __name__ == "__main__":
    init_runner()

    knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.LINSYS,
            pcg_max_iters=200,
            sqp_sim_period=ADAPTIVE,
        ),
        name_prefix="baseline"
    )

    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.LINSYS,
            pcg_max_iters=200,
            sqp_sim_period=ADAPTIVE,
            pcg_result_reuse = PCGResultReuse.NO_RESULT_REUSE,
        ),
        name_prefix="no_reuse"
    )

    run_expr(
        knot_points,
        Settings(
            timing_mode=TimingMode.LINSYS,
            pcg_max_iters=200,
            sqp_sim_period=ADAPTIVE,
            pcg_result_reuse = PCGResultReuse.MEASURE_CLOSENESS_INITIAL_GUESS
        ),
        name_prefix="measure_closeness_initial_guess"
    )
