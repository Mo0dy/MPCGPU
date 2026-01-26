#!/usr/bin/env python3

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from typing import List, Union, TypeAlias, Literal
from datetime import datetime
from dataclasses import dataclass

from argparse import ArgumentParser

from enum import Enum

# Compatibility with python 3.8
from typing import Union, List

has_been_initialized = False
dry_run = False

class TimingMode(Enum):
    MINIMAL = 0
    LINSYS = 1
    FINE_GRAINED = 2

    def __str__(self):
        # without TimingMode.
        return self.name.lower()


# NOTE: should match the defines in settings.cuh
class LineSearchMode(Enum):
    # preselecting only some
    LINE_SEARCH_EXP_GRID = 1
    LINE_SEARCH_EXP_ACADOS_BACKTRACKING = 2
    LINE_SEARCH_FULLSTEP_01 = 3
    LINE_SEARCH_QUADR_MINIMUM_50P = 4
    LINE_SEARCH_QUADR_WEIGHTED_BELL_A_09_S_1 = 5
    LINE_SEARCH_QUADR_WEIGHTED_BELL_A_08_S_1 = 6
    LINE_SEARCH_QUADR_WEIGHTED_BELL_A_0999_S_1 = 7
    LINE_SEARCH_QUADR_MINIMUM = 8
    LINE_SEARCH_EXP_GRID_Mod8_No_Bell = 12
    LINE_SEARCH_QUADRATIC_GRID_No_Bell = 13
    LINE_SEARCH_QUADRATIC_GRID_Bell = 14
    LINE_SEARCH_EXP_GRID_Mod8_Bell = 15

    def __str__(self):
        return self.name.lower()

ADAPTIVE = "adaptive"
Adaptive: TypeAlias = Literal["adaptive"]
PCGMaxIters: TypeAlias = Union[int, Adaptive]


SimPeriod: TypeAlias = Union[int, Literal["adaptive"]]


@dataclass
class Settings:
    timing_mode: TimingMode
    pcg_max_iters: PCGMaxIters
    sqp_sim_period: SimPeriod = 2000  # the time the robot is simulated for in us
    sqp_max_time_us: int | None = None  # the max time sqp is allowed to run for in us if sqp_sim_period is not ADAPTIVE. If None this is set to sqp_sim_period
    enable_preconditioning: bool = True
    line_search_version: LineSearchMode = LineSearchMode.LINE_SEARCH_EXP_GRID_Mod8_No_Bell
    num_alphas: int = 8  # number of alphas used in line search

    @classmethod
    def default(cls):
        return cls(
            timing_mode=TimingMode.MINIMAL,
            pcg_max_iters=ADAPTIVE,
            sqp_sim_period=2000,
            enable_preconditioning=True,
        )

    def __str__(self):
        return f"timing_mode={self.timing_mode}\npcg_max_iters={self.pcg_max_iters}\nsqp_sim_period={self.sqp_sim_period}\nsqp_max_time_us={self.sqp_max_time_us}\nenable_preconditioning={self.enable_preconditioning}\nline_search_version={self.line_search_version}\nnum_alphas={self.num_alphas}\n"

    def make_title(self) -> str:
        return f"TM={self.timing_mode}_PCG={self.pcg_max_iters}_SP={self.sqp_sim_period}_Pre={int(self.enable_preconditioning)}"


def compile():
    os.system("make clean && make examples -j $(nproc)")

def run(run_qdldl: bool = True):
    # NOTE: not perfect but better than forgetting
    if not has_been_initialized:
        print("Runner has not been initialized. Please call init_runner() first.")
        exit(1)

    compile()
    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_path = f"{current_path}:{os.getcwd()}/qdldl/build/out"
    os.environ["LD_LIBRARY_PATH"] = new_path

    if dry_run:
        print("Dry run mode: only compiling, not running the executables.")
        return

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
tmp_dir = project_root / "tmp"
results_tmp_dir = tmp_dir / "results"
results_dir = project_root / "results"

results_dir.mkdir(parents=True, exist_ok=True)

settings_f_str = """#pragma once

// ===============================================
// Line search settings
// ===============================================

//MPCGPU default
//for these settings use 8 sampling points (parallel streams)
#define LINE_SEARCH_EXP_GRID 1

//Acados default backtracking selection criteria
#define LINE_SEARCH_EXP_ACADOS_BACKTRACKING 2
//functions as baseline -> always use QP solution (since alpha = 1 is often selected as best)
#define LINE_SEARCH_FULLSTEP_01 3

//for these settings use 50 sampling points (parallel streams)
#define LINE_SEARCH_QUADR_MINIMUM 4
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_09_S_1 5
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_08_S_1 6
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_0999_S_1 7
//for these settings use 100 sampling points (parallel streams)
#define LINE_SEARCH_QUADR_MINIMUM 8

#define LINE_SEARCH_EXP_GRID_Mod8_No_Bell 12
#define LINE_SEARCH_QUADRATIC_GRID_No_Bell 13
#define LINE_SEARCH_QUADRATIC_GRID_Bell 14
#define LINE_SEARCH_EXP_GRID_Mod8_Bell 15

#define LINE_SEARCH_VERSION {line_search_version}

#define USE_TOL_END_CRITERION 1

#define NUM_ALPHAS {num_alphas}


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
    #define SQP_MAX_ITER    60
    typedef double toplevel_return_type;
#else
    #define SQP_MAX_ITER    60
    typedef uint32_t toplevel_return_type;
#endif


#ifndef SQP_MAX_TIME_US
#define SQP_MAX_TIME_US {sqp_max_time_us}
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
        settings: Settings,
) -> None:
    print(f"Writing settings to for n={knot_points}", settings_file)
    time_linsys = int(settings.timing_mode != TimingMode.MINIMAL)
    fine_grained_timing = int(settings.timing_mode == TimingMode.FINE_GRAINED)



    const_update_freq = int(settings.sqp_sim_period != ADAPTIVE)
    # TODO: check if the sim period actually has no impact if const_update_freq is set to 0
    simulation_period = 2000 if settings.sqp_sim_period == ADAPTIVE else settings.sqp_sim_period
    sqp_max_time_us = settings.sqp_sim_period if settings.sqp_max_time_us is None else settings.sqp_max_time_us



    enable_preconditioning = int(settings.enable_preconditioning)
    if settings.pcg_max_iters == ADAPTIVE:
        adaptive_max_iters_str = ""
    else:
        adaptive_max_iters_str = "#define PCG_MAX_ITER {}".format(settings.pcg_max_iters)

    settings_str = settings_f_str.format(
        knot_points=knot_points,
        time_linsys=time_linsys,
        fine_grained_timing=fine_grained_timing,
        const_update_freq=const_update_freq,
        sqp_max_time_us=sqp_max_time_us,
        adaptive_max_iters=adaptive_max_iters_str,
        simulation_period=simulation_period,
        enable_preconditioning=enable_preconditioning,
        line_search_version=settings.line_search_version.value,
        num_alphas=settings.num_alphas
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

def run_over_knot_points(
    knot_points: Union[int, List[int]],
    settings: Settings,
    run_qdldl: bool = True,
):
    if results_tmp_dir.exists():
        print("Cleaning up previous results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"results_backup_{timestamp}"
        os.rename(tmp_dir / "results", tmp_dir / backup_name)

    results_tmp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(knot_points, int):
        knot_points = [knot_points]

    print("Settings:")
    print(str(settings))

    with open(results_tmp_dir / "settings.txt", 'w') as f:
        f.write(str(settings))
        f.write(f"\n\nknot_points: {knot_points}\n")

    # TODO: write settings to results_tmp_dir/settings.txt
    for n in knot_points:
        write_settings(
            n,
            settings
        )
        compile()
        run(run_qdldl=run_qdldl)


def run_expr(
        knot_points: Union[int, List[int]],
        settings: Settings,
        name: str | None = None,
        name_prefix: str = "",
        run_qdldl: bool = True,
    ):
    if name is None:
        name = settings.make_title()
    name = f"{name_prefix}_{name}"
    with expr(name):
        run_over_knot_points(
            knot_points=knot_points,
            settings=settings,
            run_qdldl=run_qdldl
        )


def init_runner():
    global has_been_initialized, dry_run

    if has_been_initialized:
        return

    has_been_initialized = True

    parser = ArgumentParser(description="Run the MPC experiment with various settings.")
    parser.add_argument("--dry-run", action="store_true", help="Only compile, do not run the executables.")
    args = parser.parse_args()

    dry_run = args.dry_run
