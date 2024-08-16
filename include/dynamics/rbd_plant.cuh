#pragma once

#ifndef GATO_PLANT
	#define GATO_PLANT 1
#endif

#include "gpuassert.cuh"
#include <cooperative_groups.h>

#if GATO_PLANT == 1
	// #include "iiwa_plant.cuh"
	#include "iiwa/iiwa_eepos_plant.cuh"

	namespace grid {
		const int EE_POS_SIZE = 6;
		const int EE_POS_SIZE_COST = 3; // just xyz
	}
#else
	#include "pend.cuh"
	namespace grid {
		const int EE_POS_SIZE = 2;
		const int EE_POS_SIZE_COST = 2;
	}
#endif