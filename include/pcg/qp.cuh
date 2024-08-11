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
#include "linsys_setup_all.cuh"
#include "gpu_pcg.cuh"

#define time_delta_us_timespec(start, end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

template<typename T>
auto qpSolvePcg(const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points,
                T *h_G_dense,
                T *h_C_dense,
                T *h_g,
                T *h_c,
                T *h_dz,
                bool chol_or_ldl,
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

    const uint32_t KKT_G_DENSE_SIZE_BYTES = ((states_sq + controls_sq) * knot_points - controls_sq) * sizeof(T);
    const uint32_t KKT_C_DENSE_SIZE_BYTES = (states_sq + states_p_controls) * (knot_points - 1) * sizeof(T);
    const uint32_t KKT_g_SIZE_BYTES = ((state_size + control_size) * knot_points - control_size) * sizeof(T);
    const uint32_t KKT_c_SIZE_BYTES = (state_size * knot_points) * sizeof(T);
    const uint32_t DZ_SIZE_BYTES = (states_s_controls * knot_points - control_size) * sizeof(T);

    const uint32_t Nnx_T = KKT_c_SIZE_BYTES;
    const uint32_t Nnx2_T = knot_points * states_sq * sizeof(T);

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

    T *d_S, *d_Pinv;
    T *d_T = NULL;
    T *d_H = NULL;
    if (config.pcg_org_trans) {
        // TRANS
        gpuErrchk(cudaMalloc(&d_S, 2 * Nnx2_T + Nnx_T));
        gpuErrchk(cudaMalloc(&d_Pinv, 2 * Nnx2_T + Nnx_T));
        gpuErrchk(cudaMalloc(&d_T, triangular_state * knot_points * sizeof(T)));
    } else {
        // ORG
        gpuErrchk(cudaMalloc(&d_S, 3 * Nnx2_T));
        gpuErrchk(cudaMalloc(&d_Pinv, 3 * Nnx2_T));
    }

    bool use_H = config.pcg_poly_order > 0;
    if (use_H) {
        gpuErrchk(cudaMalloc(&d_H, 3 * Nnx2_T));
    }

    T *d_gamma, *d_lambda;
    gpuErrchk(cudaMalloc(&d_gamma, Nnx_T));
    // TODO: initialize d_lambda to zero vector
    gpuErrchk(cudaMalloc(&d_lambda, Nnx_T));

    T *d_dz;
    gpuErrchk(cudaMalloc(&d_dz, DZ_SIZE_BYTES));

    /*   PCG vars   */
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, Nnx_T));
    d_p = d_r;                      // share N*nx
    gpuErrchk(cudaMalloc(&d_v_temp, knot_points * sizeof(T)));
    d_eta_new_temp = d_v_temp;      // share N

    struct timespec linsys_start, linsys_end;
    double linsys_time;

    // form the Schur complement system (S, Pinv, gamma) from
    // the given KKT matrix (G_dense, C_dense, g, c)
    form_schur_system_block<T>(state_size, control_size, knot_points,
                               d_G_dense, d_C_dense, d_g, d_c,
                               d_S, d_Pinv, d_H, d_T,
                               d_gamma,
                               rho,
                               config.pcg_org_trans, chol_or_ldl, use_H);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // start linear system solver timer
    clock_gettime(CLOCK_MONOTONIC, &linsys_start);

    uint32_t pcg_iters = solvePCGCooperativeKernel<T>(
            state_size,
            knot_points,
            d_S,
            d_Pinv,
            d_H,
            d_gamma,
            d_lambda,
            d_r,
            d_p,
            d_v_temp,
            d_eta_new_temp,
            &config);

    // stop linear system solver timer
    clock_gettime(CLOCK_MONOTONIC, &linsys_end);
    linsys_time = time_delta_us_timespec(linsys_start, linsys_end);

    if (config.pcg_org_trans) {
        // TRANS
        // need to transform d_lambda back using d_T
        transform_lamdba<T>(state_size, knot_points, d_T, d_lambda);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // recover dz
    compute_dz(state_size, control_size, knot_points,
               d_Ginv_dense, d_C_dense, d_g,
               d_lambda, d_dz);
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
    gpuErrchk(cudaFree(d_gamma));
    gpuErrchk(cudaFree(d_lambda));

    gpuErrchk(cudaFree(d_r));
    gpuErrchk(cudaFree(d_v_temp));

    gpuErrchk(cudaFree(d_dz));

    return std::make_tuple(pcg_iters, linsys_time, qp_solve_time);
}
