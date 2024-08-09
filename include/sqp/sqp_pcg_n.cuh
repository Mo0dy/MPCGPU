#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <math.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <time.h>

#include "settings.cuh"
#include "kernels/batch/setup_kkt_n.cuh" // kkt
#include "kernels/batch/setup_schur_pcg_n.cuh" // schur
#include "gpu_pcg.cuh" // pcg
#include "kernels/batch/pcg_n.cuh" // pcg
#include "kernels/single/compute_dz.cuh" // dz
#include "kernels/batch/merit_n.cuh" // merit function
#include "kernels/batch/line_search_n.cuh" // line search

/**
 * @brief Solve trajectory optimization problem using sequential quadratic programming (SQP) with preconditioned conjugate gradient (PCG) method.
 * 
 * @tparam T float or double
 * @param solve_count number of solves to run in parallel
 * @param state_size state size (joint angles and velocities)
 * @param control_size control size (torques)
 * @param knot_points number of knot points in trajectory
 * @param timestep timestep between knot points
 * @param d_eePos_goal_tensor end effector goal trajectory (6 * knot_points * solve_count)
 * @param d_lambdas initial guess for lambdas (state_size * knot_points * solve_count)
 * @param d_xu_tensor initial guess for state and control trajectory ((state_size + control_size) * knot_points - control_size) * solve_count
 * @param d_dynmem pointer to dynamics memory
 * @param config 
 * @param d_rhos 
 * @param rho_reset 
 * @return auto 
 */
template <typename T>
auto sqpSolvePcgN(const uint32_t solve_count, const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points, float timestep, 
                    T* d_eePos_goal_tensor, 
                    T* d_lambdas, 
                    T* d_xu_tensor, 
                    void* d_dynmem, 
                    pcg_config<T>& config, 
                    T* d_rhos, 
                    T rho_reset) {
    
    const uint32_t traj_size = (state_size + control_size) * knot_points - control_size;
    const uint32_t G_size = (state_size * state_size + control_size * control_size) * knot_points - control_size * control_size;
    const uint32_t C_size = (state_size * state_size + state_size * control_size) * (knot_points - 1);

    // data storage for return values
    std::vector<std::vector<int>> pcg_iters_matrix(solve_count);
    std::vector<double> pcg_times_vec;
    //float sqp_solve_time = 0.0;
    std::vector<uint32_t> sqp_iterations_vec(solve_count, 0);
    std::vector<char> sqp_time_exit_vec(solve_count, true);
    std::vector<std::vector<bool>> pcg_exits_matrix(solve_count);

    uint32_t *d_sqp_iterations_vec;
    bool *d_sqp_time_exit_vec;
    gpuErrchk(cudaMalloc(&d_sqp_iterations_vec, solve_count * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_sqp_time_exit_vec, solve_count * sizeof(bool)));
    

    // rho parameters
    const T rho_factor = RHO_FACTOR;
    const T rho_max = RHO_MAX;
    const T rho_min = RHO_MIN;
    std::vector<T> drho_vec(solve_count, 1.0);
    T *d_drho_vec;
    gpuErrchk(cudaMalloc(&d_drho_vec, solve_count * sizeof(T)));
    gpuErrchk(cudaMemset(d_drho_vec, 1.0, solve_count * sizeof(T)));
    // current state
    T *d_xs;
    gpuErrchk(cudaMalloc(&d_xs, solve_count * state_size * sizeof(T)));
    for (uint32_t i = 0; i < solve_count; ++i) {
        gpuErrchk(cudaMemcpy(d_xs + i * state_size, d_xu_tensor + i * traj_size, state_size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    // KKT system
    T *d_G_dense, *d_C_dense, *d_g, *d_c, *d_dz;
    gpuErrchk(cudaMalloc(&d_G_dense, solve_count * G_size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_C_dense, solve_count * C_size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_g, solve_count * traj_size * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_c, solve_count * state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_dz, solve_count * traj_size * sizeof(T)));
    
    // Schur system
    T *d_S, *d_Pinv, *d_gamma;
    gpuErrchk(cudaMalloc(&d_S, solve_count * state_size * state_size * knot_points * sizeof(T) * 3));
    gpuErrchk(cudaMalloc(&d_Pinv, solve_count * state_size * state_size * knot_points * sizeof(T) * 3));
    gpuErrchk(cudaMalloc(&d_gamma, solve_count * state_size * knot_points * sizeof(T)));

    // PCG
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, solve_count * state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, solve_count * state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, solve_count * (knot_points + 1) * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, solve_count * (knot_points + 1) * sizeof(T)));

    std::vector<char> pcg_exit(solve_count, true);
    uint32_t *d_pcg_iters; // number of PCG iterations for each solve (return value)
    bool *d_pcg_exit; // whether PCG converged for each solve (return value)
    gpuErrchk(cudaMalloc(&d_pcg_iters, solve_count * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_pcg_exit, solve_count * sizeof(bool)));
    gpuErrchk(cudaMemset(d_pcg_exit, false, solve_count * sizeof(bool)));

    //void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;
    //const size_t pcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Line search
    const float mu = 10.0f;
    const uint32_t num_alphas = 8;
    T h_merit_news[solve_count * num_alphas];
    T h_merit_initial[solve_count];

    T *d_min_merit;
    std::vector<T> h_min_merit(solve_count, 0);
    std::vector<uint32_t> h_line_search_step(solve_count, 0);
    std::vector<T> h_alphafinal(solve_count, 0);
    T *d_merit_initial;
    T *d_merit_news;
    T *d_merit_temp;
    T *d_alphafinal;
    uint32_t *d_line_search_step;
    gpuErrchk(cudaMalloc(&d_min_merit, solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_initial, solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_news, solve_count * num_alphas * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_merit_temp, num_alphas * knot_points * solve_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_line_search_step, solve_count * sizeof(uint32_t)));
    gpuErrchk(cudaMalloc(&d_alphafinal, solve_count * sizeof(T)));

    
    gpuErrchk(cudaMemset(d_merit_initial, 0, solve_count * sizeof(T)));
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);

    // Create CUDA streams
    cudaStream_t streams[num_alphas];
    for(uint32_t str = 0; str < num_alphas; str++){
        cudaStreamCreate(&streams[str]);
    }
    gpuErrchk(cudaPeekAtLastError());

    // Initialize cuBLAS
    cublasHandle_t handle; 
    cublasCreate(&handle);
    
    gpuErrchk(cudaDeviceSynchronize());

    // Timing
    // cudaEvent_t kkt_start, kkt_stop, schur_start, schur_stop, pcg_start, pcg_stop, dz_start, dz_stop, line_search_start, line_search_stop, sqp_start, sqp_stop;
    // cudaEventCreate(&kkt_start);
    // cudaEventCreate(&kkt_stop);
    // cudaEventCreate(&schur_start);
    // cudaEventCreate(&schur_stop);
    // cudaEventCreate(&pcg_start);
    // cudaEventCreate(&pcg_stop);
    // cudaEventCreate(&dz_start);
    // cudaEventCreate(&dz_stop);
    // cudaEventCreate(&line_search_start);
    // cudaEventCreate(&line_search_stop);
    // cudaEventCreate(&sqp_start);
    // cudaEventCreate(&sqp_stop);

    struct timespec sqp_solve_start, sqp_solve_end;
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);
    
    //cudaEventRecord(sqp_start);
    struct timespec linsys_start, linsys_end;
    double linsys_time;

#if CONST_UPDATE_FREQ
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start,sqp_cur) > SQP_MAX_TIME_US;
    };
#else
    auto sqpTimecheck = [&]() { return false; };
#endif

    // ------------------ Compute Initial Merit --------------

    compute_merit<T>(solve_count, state_size, control_size, knot_points, 
        d_xu_tensor, d_eePos_goal_tensor, static_cast<T>(10), timestep, d_dynmem, d_merit_initial
    );
    
    gpuErrchk(cudaMemcpyAsync(&h_merit_initial, d_merit_initial, solve_count*sizeof(T), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());

    // ------------------ SQP loop --------------

    for (uint32_t iter = 0; iter < SQP_MAX_ITER; ++iter) {
        
        // ------------------ KKT Matrices --------------
        
        //cudaEventRecord(kkt_start);

        generate_kkt_system_batched<T>(solve_count, knot_points, state_size, control_size,
            d_G_dense, d_C_dense, d_g, d_c, d_dynmem, timestep, d_eePos_goal_tensor, d_xs, d_xu_tensor
        );
        gpuErrchk(cudaPeekAtLastError());        
        if (sqpTimecheck()){ break; }
        //cudaEventRecord(kkt_stop);
        //cudaEventSynchronize(kkt_stop);
        //float milliseconds = 0;
        //cudaEventElapsedTime(&milliseconds, kkt_start, kkt_stop);
        //std::cout << "Time elapsed for forming KKT system: " << milliseconds << std::endl;

        // ------------------ Schur Matrices --------------

        //cudaEventRecord(schur_start);

        form_schur_system_batched<T>(solve_count, state_size, control_size, knot_points, 
            d_G_dense, d_C_dense, d_g, d_c, d_S, d_Pinv, d_gamma, d_rhos
        );
        gpuErrchk(cudaPeekAtLastError());
        if (sqpTimecheck()){ break; }
        //cudaEventRecord(schur_stop);
        //cudaEventSynchronize(schur_stop);
        //cudaEventElapsedTime(&milliseconds, schur_start, schur_stop);
        //std::cout << "Time elapsed for forming Schur system: " << milliseconds << std::endl;
        gpuErrchk(cudaDeviceSynchronize());
        if (sqpTimecheck()){ break; }
        // ------------------ PCG --------------

        //cudaEventRecord(pcg_start);
        clock_gettime(CLOCK_MONOTONIC,&linsys_start);

        pcg_batched(solve_count, state_size, knot_points,
            d_S, d_Pinv, d_gamma, d_lambdas,
            d_r, d_p, d_v_temp, d_eta_new_temp,
            d_pcg_iters, d_pcg_exit, &config
        );

        std::vector<uint32_t> h_pcg_iters(solve_count);
        std::vector<char> h_pcg_exit(solve_count);

        gpuErrchk(cudaMemcpy(h_pcg_iters.data(), d_pcg_iters, solve_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_pcg_exit.data(), d_pcg_exit, solve_count * sizeof(char), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&linsys_end);

        for (uint32_t i = 0; i < solve_count; ++i) {
            pcg_iters_matrix[i].push_back(h_pcg_iters[i]);
            pcg_exits_matrix[i].push_back(h_pcg_exit[i]);
        }
        if (sqpTimecheck()){ break; }

        // void *pcgKernelArgs[] = {
        //     (void *)&d_S,
        //     (void *)&d_Pinv,
        //     (void *)&d_gamma,
        //     (void *)&d_lambdas,
        //     (void *)&d_r,
        //     (void *)&d_p,
        //     (void *)&d_v_temp,
        //     (void *)&d_eta_new_temp,
        //     (void *)&d_pcg_iters,
        //     (void *)&d_pcg_exit,
        //     (void *)&config.pcg_max_iter,
        //     (void *)&config.pcg_exit_tol
        // };
        // uint32_t pcg_iters = 0;
        // bool pcg_exit = true;
        // gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, PCG_NUM_THREADS, pcgKernelArgs, ppcg_kernel_smem_size));    
        // gpuErrchk(cudaMemcpy(&pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        // gpuErrchk(cudaMemcpy(&pcg_exit, d_pcg_exit, sizeof(bool), cudaMemcpyDeviceToHost));


        //cudaEventRecord(pcg_stop);
        //cudaEventSynchronize(pcg_stop);
        //cudaEventElapsedTime(&milliseconds, pcg_start, pcg_stop);
        linsys_time = time_delta_us_timespec(linsys_start,linsys_end);
        pcg_times_vec.push_back(linsys_time);
        //std::cout << "Time elapsed for PCG: " << milliseconds << std::endl;

        // ------------------ Recover dz --------------
        //cudaEventRecord(dz_start);

        for (uint32_t i = 0; i < solve_count; ++i) {
            compute_dz<T>(state_size, control_size, knot_points, 
                        d_G_dense + i * G_size, 
                        d_C_dense + i * C_size, 
                        d_g + i * traj_size, 
                        d_lambdas + i * state_size * knot_points, 
                        d_dz + i * traj_size);
            gpuErrchk(cudaPeekAtLastError());
        }
        if (sqpTimecheck()){ break; }

        //cudaEventRecord(dz_stop);
        //cudaEventSynchronize(dz_stop);
        //cudaEventElapsedTime(&milliseconds, dz_start, dz_stop);
        //std::cout << "Time elapsed for computing dz: " << milliseconds << std::endl;

        // ------------------ Line Search --------------

        //cudaEventRecord(line_search_start);

        ls_compute_merit_batched<T>(solve_count, state_size, control_size, knot_points, 
            d_xs, d_xu_tensor, d_eePos_goal_tensor, mu, timestep, d_dynmem, d_dz, num_alphas, d_merit_news, d_merit_temp);
        if (sqpTimecheck()){ break; }    
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        find_alpha_batched(solve_count, num_alphas,
            d_merit_news, d_merit_initial, d_min_merit, d_line_search_step, d_alphafinal
        );

        if (sqpTimecheck()){ break; }

        // update xu and rho
        for (uint32_t i = 0; i < solve_count; ++i) {
            T alphafinal = h_alphafinal[i];
            uint32_t line_search_step = h_line_search_step[i];
            T min_merit = h_min_merit[i];
        
            // Check if there's an improvement
            if (min_merit < h_merit_initial[i]) {
                // Update xu with dz and alpha                    
#if USE_DOUBLES
                cublasDaxpy(handle, 
                    traj_size,
                    &alphafinal,
                    d_dz + i * traj_size, 1,
                    d_xu_tensor + i * traj_size, 1
                );
#else
                cublasSaxpy(handle, 
                    traj_size,
                    &alphafinal,
                    d_dz + i * traj_size, 1,
                    d_xu_tensor + i * traj_size, 1
                );
#endif
                // Update drho and rho
                T d_rho;
                gpuErrchk(cudaMemcpy(&d_rho, d_rhos + i, sizeof(T), cudaMemcpyDeviceToHost));
                drho_vec[i] = std::min(drho_vec[i] / rho_factor, 1 / rho_factor);
                T rho = std::max(d_rho * drho_vec[i], rho_min);
                gpuErrchk(cudaMemcpy(&d_rhos[i], &rho, sizeof(T), cudaMemcpyHostToDevice));
            } else {
                // No improvement
                T d_rho;
                gpuErrchk(cudaMemcpy(&d_rho, d_rhos + i, sizeof(T), cudaMemcpyDeviceToHost));
                drho_vec[i] = std::max(drho_vec[i] * rho_factor, rho_factor);
                T rho = std::max(d_rho * drho_vec[i], rho_min);
                if (rho > rho_max) {
                    sqp_time_exit_vec[i] = false;
                    rho = rho_reset;
                }
                gpuErrchk(cudaMemcpy(&d_rhos[i], &rho, sizeof(T), cudaMemcpyHostToDevice));
            }
        
            // Increment SQP iterations
            sqp_iterations_vec[i]++;
        
            // Update merit for next iteration
            h_merit_initial[i] = min_merit;
        }
        
        gpuErrchk(cudaMemcpy(d_merit_initial, h_merit_initial, solve_count * sizeof(T), cudaMemcpyHostToDevice));
        // dim3 grid(solve_count, traj_size);
        // dim3 block(MERIT_THREADS);
        // update_xu_and_rho_kernel<<<grid, block>>>(
        //     solve_count, traj_size,
        //     d_alphafinal, d_line_search_step, d_min_merit,
        //     d_merit_initial, d_xu_tensor, d_dz, d_rhos, d_drho_vec,
        //     d_sqp_iterations_vec, d_sqp_time_exit_vec,
        //     rho_factor, rho_min, rho_max, rho_reset
        // );
        
        // Check for errors
        if (sqpTimecheck()){ break; }
        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(h_min_merit.data(), d_min_merit, solve_count * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_line_search_step.data(), d_line_search_step, solve_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_alphafinal.data(), d_alphafinal, solve_count * sizeof(T), cudaMemcpyDeviceToHost);

        // gpuErrchk(cudaMemcpy(d_merit_initial, d_min_merit, solve_count * sizeof(T), cudaMemcpyDeviceToDevice));
        // // gpuErrchk(cudaMemcpy(h_merit_news, d_merit_news, solve_count * num_alphas * sizeof(T), cudaMemcpyDeviceToHost));
        //cudaEventRecord(line_search_stop);
        //cudaEventSynchronize(line_search_stop);
        //cudaEventElapsedTime(&milliseconds, line_search_start, line_search_stop);
        //std::cout << "Time elapsed for merit-function and line-search: " << milliseconds << std::endl;


        // ------------------ Check convergence --------------

        bool all_converged = true;
        for (uint32_t i = 0; i < solve_count; ++i) {
            if (sqp_time_exit_vec[i]) {
                all_converged = false;
                break;
            }
        }
        if (all_converged) { break; } //TODO: test this
    }

    // ------------------ End of SQP loop --------------

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_end);
    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);
    //cudaEventRecord(sqp_stop);
    //cudaEventSynchronize(sqp_stop);
    //cudaEventElapsedTime(&sqp_solve_time, sqp_start, sqp_stop);
    //std::cout << "Time elapsed for SQP: " << sqp_solve_time << std::endl;
    //std::cout << "Average time per iteration: " << sqp_solve_time / SQP_MAX_ITER << std::endl;

    // ------------------ Clean up --------------

    cudaFree(d_xs);
    cudaFree(d_G_dense);
    cudaFree(d_C_dense);
    cudaFree(d_g);
    cudaFree(d_c);
    cudaFree(d_dz);
    cudaFree(d_S);
    cudaFree(d_Pinv);
    cudaFree(d_gamma);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_v_temp);
    cudaFree(d_eta_new_temp);
    cudaFree(d_pcg_iters);
    cudaFree(d_pcg_exit);
    cudaFree(d_merit_news);
    cudaFree(d_merit_temp);

    // cudaEventDestroy(kkt_start);
    // cudaEventDestroy(kkt_stop);
    // cudaEventDestroy(schur_start);
    // cudaEventDestroy(schur_stop);
    // cudaEventDestroy(pcg_start);
    // cudaEventDestroy(pcg_stop);
    // cudaEventDestroy(line_search_start);
    // cudaEventDestroy(line_search_stop);
    // cudaEventDestroy(sqp_start);
    // cudaEventDestroy(sqp_stop);

    for (uint32_t str = 0; str < num_alphas; str++) {
        cudaStreamDestroy(streams[str]);
    }

    cublasDestroy(handle);

    // ------------------ Return values --------------

    return std::make_tuple(pcg_iters_matrix, pcg_times_vec, sqp_solve_time, sqp_iterations_vec, sqp_time_exit_vec, pcg_exits_matrix);
}
