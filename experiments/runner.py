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

ADAPTIVE = "adaptive"
Adaptive: TypeAlias = Literal["adaptive"]
PCGMaxIters: TypeAlias = Union[int, Adaptive]

SimPeriod: TypeAlias = Union[int, Literal["adaptive"]]

class PCGResultReuse(Enum):
    DISABLE_INSTRUMENTATION = 0 # Default behaviour
    NO_RESULT_REUSE = 1 # No reuse of the previous result --> closeness measurement would be 0 anyways
    MEASURE_CLOSENESS_INITIAL_GUESS = 2 # Measuring closeness requires result reuse.

    def __str__(self):
        return self.name.lower()


@dataclass
class Settings:
    timing_mode: TimingMode
    pcg_max_iters: PCGMaxIters
    sqp_sim_period: SimPeriod = 2000
    enable_preconditioning: bool = True
    pcg_result_reuse: PCGResultReuse = PCGResultReuse.DISABLE_INSTRUMENTATION

    @classmethod
    def default(cls):
        return cls(
            timing_mode=TimingMode.MINIMAL,
            pcg_max_iters=ADAPTIVE,
            sqp_sim_period=2000,
            enable_preconditioning=True,
        )

    def __str__(self):
        return f"timing_mode={self.timing_mode}\npcg_max_iters={self.pcg_max_iters}\nsqp_sim_period={self.sqp_sim_period}\nenable_preconditioning={self.enable_preconditioning}\npcg_result_reuse={self.pcg_result_reuse}\n"

    def make_title(self) -> str:
        basic_title = f"TM={self.timing_mode}_PCG={self.pcg_max_iters}_SP={self.sqp_sim_period}"
        if not self.enable_preconditioning:
            basic_title += "_NO_PRECOND"
        if self.pcg_result_reuse != PCGResultReuse.DISABLE_INSTRUMENTATION:
            basic_title += f"_PCG_REUSE={self.pcg_result_reuse}"
        return basic_title


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
// My Settings
// ===============================================

// Default value is 1
#define PCG_RESULT_REUSE {pcg_result_reuse}
// This only makes sense if PCG_RESULT_REUSE is set to 1
#define MEASURE_PCG_CLOSENESS_INITIAL_GUESS {measure_pcg_closeness_initial_guess}

#if PCG_RESULT_REUSE == 0 and MEASURE_PCG_CLOSENESS_INITIAL_GUESS == 1
#error "PCG_RESULT_REUSE must be set to 1 if MEASURE_PCG_CLOSENESS_INITIAL_GUESS is set to 1"
#endif

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
        settings: Settings,
) -> None:
    print(f"Writing settings to for n={knot_points}", settings_file)
    time_linsys = int(settings.timing_mode != TimingMode.MINIMAL)
    fine_grained_timing = int(settings.timing_mode == TimingMode.FINE_GRAINED)
    const_update_freq = int(settings.sqp_sim_period != ADAPTIVE)
    # TODO: check if the sim period actually hasno impact if const_update_freq is set to 0
    simulation_period = 2000 if settings.sqp_sim_period == ADAPTIVE else settings.sqp_sim_period
    enable_preconditioning = int(settings.enable_preconditioning)
    if settings.pcg_max_iters == ADAPTIVE:
        adaptive_max_iters_str = ""
    else:
        adaptive_max_iters_str = "#define PCG_MAX_ITER {}".format(settings.pcg_max_iters)

    if settings.pcg_result_reuse == PCGResultReuse.DISABLE_INSTRUMENTATION:
        pcg_result_reuse = 1
        measure_pcg_closeness_initial_guess = 0
    elif settings.pcg_result_reuse == PCGResultReuse.NO_RESULT_REUSE:
        pcg_result_reuse = 0
        measure_pcg_closeness_initial_guess = 0
    elif settings.pcg_result_reuse == PCGResultReuse.MEASURE_CLOSENESS_INITIAL_GUESS:
        pcg_result_reuse = 1
        measure_pcg_closeness_initial_guess = 1
    else:
        assert False
        

    settings_str = settings_f_str.format(
        knot_points=knot_points,
        time_linsys=time_linsys,
        fine_grained_timing=fine_grained_timing,
        const_update_freq=const_update_freq,
        adaptive_max_iters=adaptive_max_iters_str,
        simulation_period=simulation_period,
        enable_preconditioning=enable_preconditioning,
        pcg_result_reuse=pcg_result_reuse,
        measure_pcg_closeness_initial_guess=measure_pcg_closeness_initial_guess,
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
