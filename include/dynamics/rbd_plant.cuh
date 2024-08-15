#pragma once

#ifndef GATO_PLANT
	#define GATO_PLANT 1
#endif

#include "gpuassert.cuh"
#include <cooperative_groups.h>

#if GATO_PLANT == 1
	// #include "iiwa_plant.cuh"
	#include "iiwa/iiwa_eepos_plant.cuh"
#else
	#include "pend.cuh"
#endif