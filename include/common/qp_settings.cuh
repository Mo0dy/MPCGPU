#pragma once

#ifndef KNOT_POINTS
#define KNOT_POINTS 3
#endif

#ifndef STATE_SIZE
#define STATE_SIZE  2
#endif

/*******************************************************************************
 *                           Test Settings                               *
 *******************************************************************************/

#ifndef USE_DOUBLES
#define USE_DOUBLES 0
#endif

#if USE_DOUBLES
typedef double linsys_t;
#else
typedef float linsys_t;
#endif

/*******************************************************************************
 *                           Linsys Settings                               *
 *******************************************************************************/

#ifndef PCG_NUM_THREADS
#define PCG_NUM_THREADS	128
#endif


/* LINSYS_SOLVE = 1 uses pcg as the underlying linear system solver
LINSYS_SOLVE = 0 uses qdldl as the underlying linear system solver */

#ifndef LINSYS_SOLVE
#define LINSYS_SOLVE 1 
#endif

#ifndef PCG_MAX_ITER
#define PCG_MAX_ITER 100
#endif


/*******************************************************************************
 *                           QP Settings                               *
 *******************************************************************************/

#ifndef SCHUR_THREADS
#define SCHUR_THREADS       64
#endif 

#ifndef DZ_THREADS
#define DZ_THREADS          64
#endif 

#ifndef KKT_THREADS
#define KKT_THREADS         64
#endif
