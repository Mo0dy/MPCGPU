#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "dynamics/rbd_plant.cuh"
#include "utils/integrator.cuh"

// Custom atomicMin for floating-point types
template <typename T>
__device__ T atomicMinFloat(T* address, T val)
{
    T old = *address, assumed;
    do {
        assumed = old;
        old = __int_as_float(atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(val)));
    } while (assumed != old && val < old);
    return old;
}

template <typename T>
__global__ 
void find_alpha_kernel_batched(uint32_t solve_count, uint32_t num_alphas,
    const T* d_merit_news, const T* d_merit_initial, T* d_min_merit, uint32_t* d_line_search_step, T* d_alphafinal)
{
    const uint32_t solve_id = blockIdx.x;
    const uint32_t tid = threadIdx.x + blockIdx.y * blockDim.x;
    const uint32_t stride = blockDim.x * gridDim.y;
    
    if (solve_id >= solve_count) return;

    __shared__ T s_min_merit;
    __shared__ uint32_t s_best_alpha;

    // Initialize shared memory
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        s_min_merit = d_merit_initial[solve_id];
        s_best_alpha = 0;
    }
    __syncthreads();

    // Find alpha that minimizes merit function
    for (uint32_t alpha_id = tid; alpha_id < num_alphas; alpha_id += stride) {
        T current_merit = d_merit_news[solve_id * num_alphas + alpha_id];
        if (current_merit < s_min_merit) {
            T old_min = atomicMinFloat(&s_min_merit, current_merit);
            if (s_min_merit == current_merit) {
                atomicExch(&s_best_alpha, alpha_id);
            }
        }
    }
    __syncthreads();

    // Update global memory with best alpha and merit
    if (tid == 0) {
        d_min_merit[solve_id] = s_min_merit;
        d_line_search_step[solve_id] = s_best_alpha;
        d_alphafinal[solve_id] = -1.0 / (1 << s_best_alpha);
    }
}

template <typename T>
__global__ void update_xu_and_rho_kernel(
    uint32_t solve_count, uint32_t traj_size,
    const T* d_alphafinal, const uint32_t* d_line_search_step, const T* d_min_merit,
    const T* d_merit_initial, T* d_xu_tensor, const T* d_dz, T* d_rhos, T* d_drho_vec,
    uint32_t* d_sqp_iterations, bool* d_sqp_time_exit,
    T rho_factor, T rho_min, T rho_max, T rho_reset)
{
    const uint32_t solve_id = blockIdx.x;
    const uint32_t tid = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (solve_id >= solve_count) return;

    // Shared memory for improvement flag and alphafinal
    __shared__ bool s_improvement;
    __shared__ T s_alphafinal;

    // First thread in block handles scalar operations
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        T min_merit = d_min_merit[solve_id];
        s_improvement = (min_merit < d_merit_initial[solve_id]);
        s_alphafinal = d_alphafinal[solve_id];

        if (s_improvement) {
            // Update drho and rho
            d_drho_vec[solve_id] = min(d_drho_vec[solve_id] / rho_factor, T(1) / rho_factor);
            d_rhos[solve_id] = max(d_rhos[solve_id] * d_drho_vec[solve_id], rho_min);
        } else {
            // No improvement
            d_drho_vec[solve_id] = max(d_drho_vec[solve_id] * rho_factor, rho_factor);
            d_rhos[solve_id] = max(d_rhos[solve_id] * d_drho_vec[solve_id], rho_min);
            if (d_rhos[solve_id] > rho_max) {
                d_sqp_time_exit[solve_id] = false;
                d_rhos[solve_id] = rho_reset;
            }
        }

        // Increment SQP iterations
        d_sqp_iterations[solve_id]++;

        // Update merit for next iteration
        const_cast<T*>(d_merit_initial)[solve_id] = min_merit;
    }
    //__syncthreads();

    // Update xu with dz and alpha
    if (s_improvement) {
        for (uint32_t j = tid; j < traj_size; j += blockDim.x * gridDim.y) {
            uint32_t idx = solve_id * traj_size + j;
            d_xu_tensor[idx] += s_alphafinal * d_dz[idx];
        }
    }
}

template <typename T>
void find_alpha_batched(uint32_t solve_count, uint32_t num_alphas,
    const T* d_merit_news, const T* d_merit_initial, T* d_min_merit, uint32_t* d_line_search_step, T* d_alphafinal)
{
    dim3 grid(solve_count, num_alphas);
    dim3 block(MERIT_THREADS);
    find_alpha_kernel_batched<T><<<grid, block>>>(
        solve_count, num_alphas,
        d_merit_news, d_merit_initial, d_min_merit, d_line_search_step, d_alphafinal
    );
    cudaDeviceSynchronize();
}