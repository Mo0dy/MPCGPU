#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <math.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <tuple>
#include <time.h>

#include "qp_settings.cuh"
#include "linsys_block_setup.cuh"
#include "common/dz.cuh"
#include "gpu_pcg.cuh"

#define time_delta_us_timespec(start, end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

template<typename T, bool DECOMPOSITION_SQUARE_ROOT>
auto qpBlockSolvePcg(const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points,
                     T *h_G_dense,
                     T *h_C_dense,
                     T *h_g,
                     T *h_c,
                     T *h_dz,
                     pcg_config <T> &config) {

    T rho = static_cast<T>(0.0);
    // qp timing
    struct timespec qp_solve_start, qp_solve_end;
    clock_gettime(CLOCK_MONOTONIC, &qp_solve_start);

    const uint32_t triangular_state = (state_size + 1) * state_size / 2;
    const uint32_t states_sq = state_size * state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(
            ((states_sq + controls_sq) * knot_points - controls_sq) * sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq + states_p_controls) * (knot_points - 1) *
                                                                  sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES = static_cast<uint32_t>(((state_size + control_size) * knot_points - control_size) *
                                                            sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES = static_cast<uint32_t>((state_size * knot_points) * sizeof(T));
    const uint32_t DZ_SIZE_BYTES = static_cast<uint32_t>((states_s_controls * knot_points - control_size) * sizeof(T));


    T *d_G_dense, *d_C_dense, *d_g, *d_c, *d_Ginv_dense;

    gpuErrchk(cudaMalloc(&d_G_dense, KKT_G_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_C_dense, KKT_C_DENSE_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_g, KKT_g_SIZE_BYTES));
    gpuErrchk(cudaMalloc(&d_c, KKT_c_SIZE_BYTES));
    d_Ginv_dense = d_G_dense;

    // G_dense, C_dense, g, c are ready on the Host memory
    // copy them to Device memory
    cudaMemcpy(d_G_dense, h_G_dense, KKT_G_DENSE_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_dense, h_C_dense, KKT_C_DENSE_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, KKT_g_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, KKT_c_SIZE_BYTES, cudaMemcpyHostToDevice);

    T *d_T;
    gpuErrchk(cudaMalloc(&d_T, triangular_state * knot_points * sizeof(T)));

    T *d_Sdb, *d_Sob, *d_gamma, *d_lambda;
    gpuErrchk(cudaMalloc(&d_Sdb, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_Sob, 2 * states_sq * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_gamma, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_lambda, state_size * knot_points * sizeof(T)));

    T *d_dz;
    gpuErrchk(cudaMalloc(&d_dz, DZ_SIZE_BYTES));

    // pcg things
    T *d_Pinvdb, *d_Pinvob;
    gpuErrchk(cudaMalloc(&d_Pinvdb, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_Pinvob, 2 * states_sq * knot_points * sizeof(T)));

    /*   PCG vars   */
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;// *d_r_tilde, *d_upsilon;
    gpuErrchk(cudaMalloc(&d_r, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knot_points * sizeof(T)));

    void *pcg_kernel = (void *) pcgBlock<T, STATE_SIZE, KNOT_POINTS>;
    uint32_t pcg_iters;
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool pcg_exit;
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));

    void *pcgKernelArgs[] = {
            (void *) &d_Sdb,
            (void *) &d_Sob,
            (void *) &d_Pinvdb,
            (void *) &d_Pinvob,
            (void *) &d_gamma,
            (void *) &d_lambda,
            (void *) &d_r,
            (void *) &d_p,
            (void *) &d_v_temp,
            (void *) &d_eta_new_temp,
            (void *) &d_pcg_iters,
            (void *) &d_pcg_exit,
            (void *) &config.pcg_max_iter,
            (void *) &config.pcg_exit_tol
    };
    size_t ppcg_kernel_smem_size = pcgBlockSharedMemSize<T>(state_size, knot_points);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    struct timespec linsys_start, linsys_end;
    double linsys_time;

    // form the Schur complement system (S, Pinv, gamma) from
    // the given KKT matrix (G_dense, C_dense, g, c)
    form_schur_system_block<T, DECOMPOSITION_SQUARE_ROOT>(
            state_size,
            control_size,
            knot_points,
            d_G_dense,
            d_C_dense,
            d_g,
            d_c,
            d_Sdb,
            d_Sob,
            d_Pinvdb,
            d_Pinvob,
            d_T,
            d_gamma,
            rho
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // start linear system solver timer
    clock_gettime(CLOCK_MONOTONIC, &linsys_start);

    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, PCG_NUM_THREADS, pcgKernelArgs,
                                          ppcg_kernel_smem_size));
    gpuErrchk(cudaMemcpy(&pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&pcg_exit, d_pcg_exit, sizeof(bool), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // stop linear system solver timer
    clock_gettime(CLOCK_MONOTONIC, &linsys_end);
    linsys_time = time_delta_us_timespec(linsys_start, linsys_end);

    // need to transform d_lambda using d_T
    transform_lamdba<T>(state_size, knot_points, d_T, d_lambda);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // recover dz
    compute_dz(
            state_size,
            control_size,
            knot_points,
            d_Ginv_dense,
            d_C_dense,
            d_g,
            d_lambda,
            d_dz
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // d_dz ready in Device memory
    // copy it to Host memory
    cudaMemcpy(h_dz, d_dz, DZ_SIZE_BYTES, cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &qp_solve_end);
    double qp_solve_time = time_delta_us_timespec(qp_solve_start, qp_solve_end);

    // free all allocated memory on Device
    gpuErrchk(cudaFree(d_G_dense));
    gpuErrchk(cudaFree(d_C_dense));
    gpuErrchk(cudaFree(d_g));
    gpuErrchk(cudaFree(d_c));
    gpuErrchk(cudaFree(d_Sdb));
    gpuErrchk(cudaFree(d_Sob));
    gpuErrchk(cudaFree(d_gamma));
    gpuErrchk(cudaFree(d_dz));
    gpuErrchk(cudaFree(d_pcg_iters));
    gpuErrchk(cudaFree(d_pcg_exit));
    gpuErrchk(cudaFree(d_Pinvdb));
    gpuErrchk(cudaFree(d_Pinvob));
    gpuErrchk(cudaFree(d_r));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_v_temp));
    gpuErrchk(cudaFree(d_eta_new_temp));

    return std::make_tuple(pcg_iters, linsys_time, qp_solve_time);
}
