#!/usr/bin/env python3

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from typing import List, Union

def compile():
    os.system("make clean && make examples")

def run(run_qdldl: bool = True):
    compile()
    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_path = f"{current_path}:{os.getcwd()}/qdldl/build/out"
    os.environ["LD_LIBRARY_PATH"] = new_path

    # Run the PCG executable and redirect output to the Python script's output
    try:
        print("Running pcg.exe...")
        subprocess.run(["./examples/pcg.exe"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running pcg.exe: {e}")

    if not run_qdldl:
        return
    # Run the QDLDL executable and redirect output to the Python script's output
    try:
        print("Running qdldl.exe...")
        subprocess.run(["./examples/qdldl.exe"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running qdldl.exe: {e}")

project_root = Path(__file__).parent.parent
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
#define SIMULATION_PERIOD {simulation_period}
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

#define ENABLE_PRECONDITIONING {enable_preconditioning}

/* time_linsys = 1 to record linear system solve times.
time_linsys = 0 to record number of sqp iterations.
In both cases, the tracking error will also be recorded. */

#define TIME_LINSYS {time_linsys}
#define FINE_GRAINED_TIMING {fine_grained_timing}

#if FINE_GRAINED_TIMING && !TIME_LINSYS
#error "Fine grained timing requires time linsys to be enabled"
#endif

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
        fine_grained_timing: bool,
        pcg_max_iters: int,
        const_update_freq: bool,
        simulation_period: int,
        enable_preconditioning: bool
) -> None:
    if fine_grained_timing and not time_linsys:
        raise ValueError("Fine grained timing requires time linsys to be enabled")

    print(f"""Writing settings.cuh with the following parameters:
    knot_points: {knot_points}
    time_linsys: {time_linsys}
    adaptive_max_iters: {adaptive_max_iters}
    pcg_max_iters: {pcg_max_iters}
    const_update_freq: {const_update_freq}
    simulation_period: {simulation_period}
    enable_preconditioning: {enable_preconditioning}
""")

    settings_str = settings_f_str.format(
        knot_points=knot_points,
        time_linsys=int(time_linsys),
        fine_grained_timing=int(fine_grained_timing),
        const_update_freq=int(const_update_freq),
        adaptive_max_iters=(
            "" if adaptive_max_iters else f"#define PCG_MAX_ITER {pcg_max_iters}"
        ),
        simulation_period=simulation_period,
        enable_preconditioning=int(enable_preconditioning)
    )

    with open(settings_file, 'w') as f:
        f.write(settings_str)

def store_results(name):
    """Copies the results tmp dir into the actual results dir with the specified name."""
    print(f"Storing results in {name}...")
    os.rename(results_tmp_dir, results_dir / name)
    results_tmp_dir.mkdir(parents=True, exist_ok=True)

    settings_file_dest = results_dir / name / "settings.cuh"
    with open(settings_file, 'r') as src:
        with open(settings_file_dest, 'w') as f:
            f.write(src.read())

def print_experiment_header(experiment: str):
    print("=========================================================================")
    print(f"Running experiment: {experiment}")
    print("=========================================================================")


@contextmanager
def expr(name: str):
    print_experiment_header(name)
    yield
    store_results(name)
    print("Finished experiment:", name)

def run_expr(
    knot_points: Union[int, List[int]],
    time_linsys: bool,
    adaptive_max_iters: bool,
    fine_grained_timing: bool = False,
    pcg_max_iters: int = 200,
    const_update_freq: bool = True,
    simulation_period: int = 2000,
    enable_preconditioning: bool = True,
    run_qdldl: bool = True
):
    if isinstance(knot_points, int):
        knot_points = [knot_points]
    for n in knot_points:
        write_settings(
            knot_points=n,
            time_linsys=time_linsys,
            adaptive_max_iters=adaptive_max_iters,
            fine_grained_timing=fine_grained_timing,
            pcg_max_iters=pcg_max_iters,
            const_update_freq=const_update_freq,
            simulation_period=simulation_period,
            enable_preconditioning=enable_preconditioning
        )
        compile()
        run(run_qdldl=run_qdldl)
