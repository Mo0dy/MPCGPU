#pragma once

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
#define LINE_SEARCH_QUADR_MINIMUM_50P 4
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_09_S_1_50P 5
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_08_S_1_50P 6
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_0999_S_1_50P 7
//for these settings use 100 sampling points (parallel streams)
#define LINE_SEARCH_QUADR_MINIMUM_100P 8
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_09_S_1_100P 9
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_08_S_1_100P 10
#define LINE_SEARCH_QUADR_WEIGHTED_BELL_A_0999_S_1_100P 11

#define LINE_SEARCH_EXP_GRID_Mod8_No_Bell 12
#define LINE_SEARCH_QUADRATIC_GRID_No_Bell 13
#define LINE_SEARCH_QUADRATIC_GRID_Bell 14
#define LINE_SEARCH_EXP_GRID_Mod8_Bell 15

#define LINE_SEARCH_VERSION 15

#define USE_TOL_END_CRITERION 1

//parallel streams can be set by using LINE_SEARCH_VERSION
#if LINE_SEARCH_VERSION < 4
#define NUM_ALPHAS 8
#elif LINE_SEARCH_VERSION < 8
//#define NUM_ALPHAS 50
#define NUM_ALPHAS 8
//can be replaced
#elif LINE_SEARCH_VERSION < 12
//#define NUM_ALPHAS 100
#define NUM_ALPHAS 8
#else
//set alpha here if wanting to run an experiment with same alphas for the last methods
//#define NUM_ALPHAS 32
#define NUM_ALPHAS 8
#endif


#ifndef KNOT_POINTS
#define KNOT_POINTS 128
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


#define CONST_UPDATE_FREQ 1

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

#define ENABLE_PRECONDITIONING 1

/* time_linsys = 1 to record linear system solve times.
time_linsys = 0 to record number of sqp iterations.
In both cases, the tracking error will also be recorded. */

#define TIME_LINSYS 1
#define FINE_GRAINED_TIMING 1

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
#define SQP_MAX_TIME_US 100000
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
