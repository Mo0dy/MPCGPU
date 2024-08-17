#pragma once

#ifndef KNOT_POINTS
#define KNOT_POINTS 3
#endif

#ifndef STATE_SIZE
#define STATE_SIZE  2
#endif

#ifndef CONTROL_SIZE
#define CONTROL_SIZE  1
#endif

/*******************************************************************************
 *                           Linsys Settings                               *
 *******************************************************************************/

// only for transformed schur and pcg
#ifndef CHOL_OR_LDL
#define CHOL_OR_LDL    false
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

#include "common/dz.cuh"
