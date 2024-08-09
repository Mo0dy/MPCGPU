#pragma once

#ifndef KNOT_POINTS
#define KNOT_POINTS 32 
#endif

#ifndef STATE_SIZE
#define STATE_SIZE 14 //for kuka iiwa14
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

#ifndef SAVE_DATA //save stats from mpcsim to file
#define SAVE_DATA   0
#endif 

#ifndef TEST_ITERS //number of times to test each trajectory for track_iiwa_pcg.cu
#define TEST_ITERS 1
#endif

#ifndef USE_DOUBLES //change type of linsys_t
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

#ifndef CONST_UPDATE_FREQ //exits sqp solver early to reach a constant update frequency
#define CONST_UPDATE_FREQ 1
#endif

#ifndef SQP_MAX_TIME_US //max time to run sqp solver, if CONST_UPDATE_FREQ == 1 (us)
#define SQP_MAX_TIME_US 16000
#endif

#ifndef REMOVE_JITTERS // run sqp solver a bunch of times before starting to track
#define REMOVE_JITTERS 1
#endif


#ifndef SHIFT_THRESHOLD // this constant controls when xu and goal will be shifted, should be a fraction of a timestep
#define SHIFT_THRESHOLD (1 * timestep)
#endif

#ifndef SIMULATION_PERIOD // how long to simulate the system during mpcsim loop (us)
#define SIMULATION_PERIOD 16000
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
#ifndef TIME_LINSYS
#define TIME_LINSYS 1
#endif

#ifndef LINSYS_SOLVE //use pcg or qdldl
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
    #define SQP_MAX_ITER   20 //20 TODO: change back to 20
    typedef double toplevel_return_type;
#else
    #define SQP_MAX_ITER    40
    typedef uint32_t toplevel_return_type;
#endif

#ifndef KKT_THREADS
#define KKT_THREADS 128
#endif

#ifndef SCHUR_THREADS
#define SCHUR_THREADS 32
#endif

#ifndef PCG_NUM_THREADS
#define PCG_NUM_THREADS	32
#endif

#ifndef DZ_THREADS
#define DZ_THREADS 128
#endif 

#ifndef MERIT_THREADS
#define MERIT_THREADS 64
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