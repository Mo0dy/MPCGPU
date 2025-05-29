#!/usr/bin/env python3

import os
from pathlib import Path
import re
import numpy as np

def compile():
    os.system("make clean && make examples")

def run():
    compile()
    os.system("./run_examples.sh")

project_root = Path(__file__).parent
settings_file = project_root / "include/common/settings.cuh"
results_tmp_dir = project_root / "tmp/results"
results_dir = project_root / "results"

results_dir.mkdir(parents=True, exist_ok=True)

settings_f_str = """#pragma once


#ifndef KNOT_POINTS
#define KNOT_POINTS {knot_points}
#endif

// default value is for iiwa arm
#ifndef STATE_SIZE
#define STATE_SIZE  14
#endif


/*******************************************************************************
 *                           Print Settings                               *
 *******************************************************************************/


#ifndef LIVE_PRINT_PATH
#define LIVE_PRINT_PATH 0
#endif

#ifndef LIVE_PRINT_STATS
#define LIVE_PRINT_STATS 0
#endif

/*******************************************************************************
 *                           Test Settings                               *
 *******************************************************************************/


#ifndef TEST_ITERS
#define TEST_ITERS 1
#endif

#ifndef SAVE_DATA
#define SAVE_DATA   1
#endif

#ifndef USE_DOUBLES
#define USE_DOUBLES 0
#endif

#if USE_DOUBLES
typedef double linsys_t;
#else
typedef float linsys_t;
#endif

/*******************************************************************************
 *                           MPC Settings                               *
 *******************************************************************************/


#define CONST_UPDATE_FREQ {const_update_freq}

// runs sqp a bunch of times before starting to track
#ifndef REMOVE_JITTERS
#define REMOVE_JITTERS  1
#endif

// this constant controls when xu and goal will be shifted, should be a fraction of a timestep
#ifndef SHIFT_THRESHOLD
#define SHIFT_THRESHOLD (1 * timestep)
#endif

#ifndef SIMULATION_PERIOD
#define SIMULATION_PERIOD 2000
#endif

#ifndef MERIT_THREADS
#define MERIT_THREADS       128
#endif

// when enabled ABSOLUTE_QD_PENALTY penalizes qd like controls, rather than penalizing relative distance to precomputed traj
#ifndef ABSOLUTE_QD_PENALTY
#define ABSOLUTE_QD_PENALTY 0
#endif


#ifndef R_COST
	#if KNOT_POINTS == 64
#define R_COST .001
	#else
#define R_COST .0001
	#endif
#endif

#ifndef QD_COST
#define QD_COST .0001
#endif



/*******************************************************************************
 *                           Linsys Settings                               *
 *******************************************************************************/


/* time_linsys = 1 to record linear system solve times.
time_linsys = 0 to record number of sqp iterations.
In both cases, the tracking error will also be recorded. */

#define TIME_LINSYS {time_linsys}

#ifndef PCG_NUM_THREADS
#define PCG_NUM_THREADS	128
#endif


/* LINSYS_SOLVE = 1 uses pcg as the underlying linear system solver
LINSYS_SOLVE = 0 uses qdldl as the underlying linear system solver */

#ifndef LINSYS_SOLVE
#define LINSYS_SOLVE 1
#endif

{adaptive_max_iters}

// Values found using experiments
#ifndef PCG_MAX_ITER
	#if LINSYS_SOLVE
		#if KNOT_POINTS == 32
#define PCG_MAX_ITER 173
		#elif KNOT_POINTS == 64
#define PCG_MAX_ITER 167
		#elif KNOT_POINTS == 128
#define PCG_MAX_ITER 167
		#elif KNOT_POINTS == 256
#define PCG_MAX_ITER 118
		#elif KNOT_POINTS == 512
#define PCG_MAX_ITER 67
		#else
#define PCG_MAX_ITER 200
		#endif
	#else
#define PCG_MAX_ITER -1
#define PCG_EXIT_TOL -1
	#endif

#endif


/*******************************************************************************
 *                           SQP Settings                               *
 *******************************************************************************/


#if TIME_LINSYS == 1
    #define SQP_MAX_ITER    20
    typedef double toplevel_return_type;
#else
    #define SQP_MAX_ITER    40
    typedef uint32_t toplevel_return_type;
#endif


#ifndef SQP_MAX_TIME_US
#define SQP_MAX_TIME_US 2000
#endif

#ifndef SCHUR_THREADS
#define SCHUR_THREADS       128
#endif

#ifndef DZ_THREADS
#define DZ_THREADS          128
#endif

#ifndef KKT_THREADS
#define KKT_THREADS         128
#endif



/*******************************************************************************
 *                           Rho Settings                               *
 *******************************************************************************/



#ifndef RHO_MIN
#define RHO_MIN 1e-3
#endif

//TODO: get rid of rho in defines
#ifndef RHO_FACTOR
#define RHO_FACTOR 1.2
#endif

#ifndef RHO_MAX
#define RHO_MAX 10
#endif
"""

def write_settings(
        knot_points: int,
        time_linsys: bool,
        adaptive_max_iters: bool,
        max_iters: int = 10000,
        const_update_freq: bool = False,
) -> None:
    settings_str = settings_f_str.format(
        knot_points=knot_points,
        time_linsys=int(time_linsys),
        const_update_freq=bool(const_update_freq),
        adaptive_max_iters=(
            "" if adaptive_max_iters else f"#define PCG_MAX_ITER {max_iters}"
        )
    )

    with open(settings_file, 'w') as f:
        f.write(settings_str)

def set_knot_points(n: int):
    with open(settings_file, 'r') as f:
        content = f.read()

    content = re.sub(r'#define\s+KNOT_POINTS\s+\d+', f'#define KNOT_POINTS {n}', content)

    with open(settings_file, 'w') as f:
        f.write(content)

def run_expr(ns: list[int]):
    for n in ns:
        set_knot_points(n)
        compile()
        run()


def store_results(name):
    """Copies the results tmp dir into the actual results dir with the specified name."""
    os.rename(results_tmp_dir, results_dir / name)
    results_tmp_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    for n in knot_points:
        write_settings(
            knot_points=n,
            time_linsys=False,
            adaptive_max_iters=True,
            const_update_freq=False
        )
        run()
    store_results("sqp_time_adaptive")

    for n in knot_points:
        write_settings(
            knot_points=n,
            time_linsys=False,
            adaptive_max_iters=False,
            max_iters=10000,
            const_update_freq=False
        )
        run()
    store_results("sqp_time_max_iters=10000")
