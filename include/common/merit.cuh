#pragma once

#include "dynamics/rbd_plant.cuh"
#include "integrator.cuh"
#include "settings.cuh"
#include <cooperative_groups.h>
#include <cstdint>

// TODO: this
template <typename T>
size_t get_merit_smem_size(uint32_t state_size, uint32_t control_size) {
    return sizeof(T) *
           (6 + (2 * state_size + control_size) + ((int)1.5 * state_size) +
            gato_plant::forwardDynamics_TempMemSize_Shared());
}

// cost compute for line search
template <typename T>
__global__ void ls_gato_compute_merit(
    uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xs,
    T *d_xu, T *d_eePos_traj, T mu, T dt, void *d_dynMem_const, T *d_dz,
    uint32_t alpha_multiplier, T *d_merits_out, T *d_merit_temp) {

    grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *)d_dynMem_const;
    const cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_s_controls = state_size + control_size;

    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;



#if NUM_ALPHAS == 8
    const uint32_t num_alphas = 8;
#elif NUM_ALPHAS == 50
    const uint32_t num_alphas = 50;
#elif NUM_ALPHAS == 100
    const uint32_t num_alphas = 100;
#else
        assert(false && "Invalid NUM_ALPHAS");
#endif
    const T alpha_min = 1e-18;
    const T alpha_max = 2;

#if LINE_SEARCH_VERSION == LINE_SEARCH_EXP_GRID
    // @LSK_0: calculate alpha (exp grid)
    T alpha = -1.0 / (1 << alpha_multiplier);
#elif LINE_SEARCH_VERSION == LINE_SEARCH_ACADOS_BACKTRACKING
    // @LSK_0: calculate alpha (acados like)
    T alpha = -1.0 / (1 << alpha_multiplier);
#elif LINE_SEARCH_VERSION == LINE_SEARCH_FULLSTEP_01
        //full_step returns 8x (-1.0, merit(-1.0))
    T alpha = -1.0;

//every other implementation uses a quadratic spaced sampling of alpha
#elif LINE_SEARCH_VERSION >=4
        T to_add = (alpha_max-alpha_min)/((T) num_alphas);
        T alpha = (-1.0)*(alpha_min + to_add * (T)alpha_multiplier)*(alpha_min + to_add * (T)alpha_multiplier);
#else
    #error "LINE_SEARCH_VERSION not defined"
#endif //LINE_SEARCH_VERSION


    T *s_eePos_k_traj = s_xux_k + 2 * state_size + control_size;
    T *s_temp = s_eePos_k_traj + 6;

    // @LSK_1: sum integration error and tracking cost over (block of) knots
    for (unsigned knot = block_id; knot < knot_points; knot += num_blocks) {

        for (int i = thread_id;
             i < state_size + (knot < knot_points - 1) * (states_s_controls);
             i += num_threads) {
            s_xux_k[i] = d_xu[knot * states_s_controls + i] +
                         alpha * d_dz[knot * states_s_controls + i];
            if (i < 6) {
                s_eePos_k_traj[i] = d_eePos_traj[knot * 6 + i];
            }
        }
        block.sync();

        Jk = gato_plant::trackingcost<T>(state_size, control_size, knot_points,
                                         s_xux_k, s_eePos_k_traj, s_temp,
                                         d_robotModel);

        block.sync();
        if (knot < knot_points - 1) {
            ck = integratorError<T>(state_size, s_xux_k,
                                    &s_xux_k[states_s_controls], s_temp,
                                    d_robotModel, dt, block);
        } else {
            // diff xs vs xs_traj
            for (int i = threadIdx.x; i < state_size; i++) {
                s_temp[i] = abs((d_xu[i] + alpha * d_dz[i]) - d_xs[i]);
            }
            block.sync();
            glass::reduce<T>(state_size, s_temp);
            block.sync();
            ck = s_temp[0];
        }
        block.sync();

        if (thread_id == 0) {
            pointmerit = Jk + mu * ck;
            d_merit_temp[alpha_multiplier * knot_points + knot] = pointmerit;
            // printf("alpha: %f knot: %d reporting merit: %f\n", alpha, knot,
            // pointmerit);
        }
    }

    // @LSK_2: reduction
    cooperative_groups::this_grid().sync();
    if (block_id == 0) {
        glass::reduce<T>(knot_points,
                         &d_merit_temp[alpha_multiplier * knot_points]);

        if (thread_id == 0) {
            d_merits_out[alpha_multiplier] =
                d_merit_temp[alpha_multiplier * knot_points];
        }
    }
}

// zero merit out
// shared mem size get_merit_smem_size()
// cost compute for non line search
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__ void compute_merit(uint32_t state_size, uint32_t control_size,
                              uint32_t knot_points, T *d_xu, T *d_eePos_traj,
                              T mu, T dt, void *d_dynMem_const,
                              T *d_merit_out) {
    grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *)d_dynMem_const;
    const cooperative_groups::thread_block block =
        cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;

    const uint32_t states_s_controls = state_size + control_size;
    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;
    T *s_eePos_k_traj = s_xux_k + 2 * state_size + control_size;
    T *s_temp = s_eePos_k_traj + 6;

    for (unsigned knot = block_id; knot < knot_points; knot += gridDim.x) {

        for (int i = thread_id;
             i < state_size + (knot < knot_points - 1) * (states_s_controls);
             i += num_threads) {
            s_xux_k[i] = d_xu[knot * states_s_controls + i];
            if (i < 6) {
                s_eePos_k_traj[i] = d_eePos_traj[knot * 6 + i];
            }
        }

        block.sync();
        Jk = gato_plant::trackingcost<T>(state_size, control_size, knot_points,
                                         s_xux_k, s_eePos_k_traj, s_temp,
                                         d_robotModel);

        block.sync();
        if (knot < knot_points - 1) {
            ck = integratorError<T>(state_size, s_xux_k,
                                    &s_xux_k[states_s_controls], s_temp,
                                    d_robotModel, dt, block);
        } else {
            ck = 0;
        }
        block.sync();

        if (thread_id == 0) {
            pointmerit = Jk + mu * ck;
            atomicAdd(d_merit_out, pointmerit);
        }
    }
}
