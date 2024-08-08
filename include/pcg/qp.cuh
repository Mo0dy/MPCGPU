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
#include "linsys_setup.cuh"
#include "gpu_pcg.cuh"

#define time_delta_us_timespec(start, end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

template<typename T>
auto qpSolvePcg(const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points,
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

    T *d_S, *d_gamma, *d_lambda;
    gpuErrchk(cudaMalloc(&d_S, 3 * states_sq * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_gamma, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_lambda, state_size * knot_points * sizeof(T)));

    T *d_dz;
    gpuErrchk(cudaMalloc(&d_dz, DZ_SIZE_BYTES));

    // preconditioner(s)
    T *d_Pinv;
    gpuErrchk(cudaMalloc(&d_Pinv, 3 * states_sq * knot_points * sizeof(T)));
    T *d_I_H = NULL;

    struct timespec linsys_start, linsys_end;
    double linsys_time;

    // form the Schur complement system (S, Pinv, gamma) from
    // the given KKT matrix (G_dense, C_dense, g, c)
    form_schur_system<T>(
            state_size,
            control_size,
            knot_points,
            d_G_dense,
            d_C_dense,
            d_g,
            d_c,
            d_S,
            d_Pinv,
            d_gamma,
            rho
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // start linear system solver timer
    clock_gettime(CLOCK_MONOTONIC, &linsys_start);

    uint32_t pcg_iters = solvePCG<T>(d_S,
                                     d_Pinv,
                                     d_I_H,
                                     d_gamma,
                                     d_lambda,
                                     state_size,
                                     knot_points,
                                     &config);

    // stop linear system solver timer
    clock_gettime(CLOCK_MONOTONIC, &linsys_end);
    linsys_time = time_delta_us_timespec(linsys_start, linsys_end);

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
    gpuErrchk(cudaFree(d_S));
    gpuErrchk(cudaFree(d_Pinv));
    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_gamma));
    gpuErrchk(cudaFree(d_dz));

    return std::make_tuple(pcg_iters, linsys_time, qp_solve_time);
}
