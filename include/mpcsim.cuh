#pragma once
#include <iomanip>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cublas_v2.h>
#include <math.h>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <tuple>
#include <time.h>
#include "integrator.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpuassert.cuh"

#if LINSYS_SOLVE == 1
#include "pcg/sqp.cuh"
#else 
#include "qdldl/sqp.cuh"
#endif



template <typename T>
__global__
void compute_tracking_error_kernel(T *d_tracking_error, uint32_t state_size, T *d_xu_goal, T *d_xs){
    
    T err;
    
    for(int ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        err = abs(d_xs[ind] - d_xu_goal[ind]);
        atomicAdd(d_tracking_error, err);
    }
}


template <typename T>
T compute_tracking_error(uint32_t state_size, T *d_xu_goal, T *d_xs){

    T h_tracking_error = 0.0f;
    T *d_tracking_error;
    gpuErrchk(cudaMalloc(&d_tracking_error, sizeof(T)));
    gpuErrchk(cudaMemcpy(d_tracking_error, &h_tracking_error, sizeof(T), cudaMemcpyHostToDevice));

    compute_tracking_error_kernel<T><<<1,32>>>(d_tracking_error, state_size, d_xu_goal, d_xs);

    gpuErrchk(cudaMemcpy(&h_tracking_error, d_tracking_error, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_tracking_error));
    return h_tracking_error;
}


template <typename T>
void dump_tracking_data(
    std::vector<int> *pcg_iters,
    std::vector<bool> *pcg_exits,
    std::vector<double> *linsys_times,
    std::vector<double> *sqp_times,
    std::vector<uint32_t> *sqp_iters,
    std::vector<bool> *sqp_exits,
    std::vector<T> *tracking_errors,
    std::vector<std::vector<T>> *tracking_path,
    uint32_t timesteps_taken,
    uint32_t control_updates_taken,
    uint32_t start_state_ind,
    uint32_t goal_state_ind,
    uint32_t test_iter,
    std::string filename_prefix,
    std::vector<double> *ktt_time_vec,
    std::vector<double> *shur_time_vec,
    std::vector<double> *dz_time_vec,
    std::vector<double> *line_search_time_vec
){
    // Helper function to create file names
    auto createFileName = [&](const std::string& data_type) {
        std::string filename = filename_prefix + "_" + std::to_string(test_iter) + "_" + data_type + ".result";
        return filename;
    };
    
    // Helper function to dump single-dimension vector data
    auto dumpVectorData = [&](const auto& data, const std::string& data_type) {
        std::ofstream file(createFileName(data_type));
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_type << " file.\n";
            return;
        }
        for (const auto& item : *data) {
            file << item << '\n';
        }
        file.close();
    };

    // Dump single-dimension vector data
    dumpVectorData(pcg_iters, "pcg_iters");
    dumpVectorData(linsys_times, "linsys_times");
    dumpVectorData(sqp_times, "sqp_times");
    dumpVectorData(sqp_iters, "sqp_iters");
    dumpVectorData(sqp_exits, "sqp_exits");
    dumpVectorData(tracking_errors, "tracking_errors");
    dumpVectorData(pcg_exits, "pcg_exits");

    // for FINE_GRAINED_TIMING
#if LINSYS_SOLVE == 1 && FINE_GRAINED_TIMING
    dumpVectorData(ktt_time_vec, "ktt_times");
    dumpVectorData(shur_time_vec, "shur_times");
    dumpVectorData(dz_time_vec, "dz_times");
    dumpVectorData(line_search_time_vec, "line_search_times");
#endif // #if LINSYS_SOLVE == 1 && FINE_GRAINED_TIMING

    // Dump two-dimension vector data (tracking_path)
    std::ofstream file(createFileName("tracking_path"));
    if (!file.is_open()) {
        std::cerr << "Failed to open tracking_path file.\n";
        return;
    }
    for (const auto& outerItem : *tracking_path) {
        for (const auto& innerItem : outerItem) {
            file << innerItem << ',';
        }
        file << '\n';
    }
    file.close();

    std::ofstream statsfile(createFileName("stats"));
    if (!statsfile.is_open()) {
        std::cerr << "Failed to open stats file.\n";
        return;
    }
    statsfile << "timesteps: " << timesteps_taken << "\n";
    statsfile << "control_updates: " << control_updates_taken << "\n";
    // printStatsToFile<double>(&linsys_times, )
    
    statsfile.close();
}


void print_test_config(){
    std::cout << "Knot points: " << KNOT_POINTS << "\n";
    std::cout << "State size: " << STATE_SIZE << "\n";
    std::cout << "Datatype: " << (USE_DOUBLES ? "DOUBLE" : "FLOAT") << "\n";
    std::cout << "Sqp exits condition: " << (CONST_UPDATE_FREQ ? "CONSTANT TIME" : "CONSTANT ITERS") << "\n";
    std::cout << "QD COST: " << QD_COST << "\n";
    std::cout << "R COST: " << R_COST << "\n";
    std::cout << "Rho factor: " << RHO_FACTOR << "\n";
    std::cout << "Rho max: " << RHO_MAX << "\n";
    std::cout << "Test iters: " << TEST_ITERS << "\n";
#if CONST_UPDATE_FREQ
    std::cout << "Max sqp time: " << SQP_MAX_TIME_US << "\n";
#else
    std::cout << "Max sqp iter: " << SQP_MAX_ITER << "\n";
#endif
    std::cout << "Solver: " << ( (LINSYS_SOLVE == 1) ? "PCG" : "QDLDL") << "\n";
#if LINSYS_SOLVE == 1
    std::cout << "Max pcg iter: " << PCG_MAX_ITER << "\n";
    // std::cout << "pcg exit tol: " << PCG_EXIT_TOL << "\n";
#endif
    std::cout << "Save data: " << (SAVE_DATA ? "ON" : "OFF") << "\n";
    std::cout << "Jitters: " << (REMOVE_JITTERS ? "ON" : "OFF") << "\n";

    std::cout << "\n\n";
}


template <typename T, typename return_type>
std::tuple<std::vector<toplevel_return_type>, std::vector<linsys_t>, linsys_t> simulateMPC(const uint32_t state_size, const uint32_t control_size, const uint32_t knot_points, const uint32_t traj_steps, 
            float timestep, T *d_eePos_traj, T *d_xu_traj, T *d_xs, uint32_t start_state_ind, uint32_t goal_state_ind, uint32_t test_iter, T linsys_exit_tol,
            std::string test_output_prefix){

    const uint32_t traj_len = (state_size+control_size)*knot_points-control_size;

    const T shift_threshold = SHIFT_THRESHOLD;
    const int max_control_updates = 100000;
    
    
    // struct timespec solve_start, solve_end;
    double sqp_solve_time_us = 0;               // current sqp solve time
    double simulation_time = 0;                 // current simulation time
    double prev_simulation_time = 0;            // last simulation time
    double time_since_timestep = 0;             // time since last timestep of original trajectory
    bool shifted = false;                       // has xu been shifted
    uint32_t traj_offset = 0;                        // current goal states of original trajectory


    // vars for recording data
    // NOTE: @Felix the current vectors are never really freed ... should not matter that much right?
    std::vector<std::vector<T>> tracking_path;      // list of traversed traj
    std::vector<int> linsys_iters;
    std::vector<double> linsys_times;
    std::vector<double> sqp_times;
    std::vector<uint32_t> sqp_iters;
    std::vector<bool> sqp_exits;
    std::vector<bool> linsys_exits;
    std::vector<T> tracking_errors;
    std::vector<int> cur_linsys_iters;
    std::vector<bool> cur_linsys_exits;
    std::vector<double> cur_linsys_times;
    std::tuple<
        std::vector<int>,
        std::vector<double>,
        double,
        uint32_t,
        bool,
        std::vector<bool>,
        std::vector<double>,
        std::vector<double>,
        std::vector<double>,
        std::vector<double>> sqp_stats;
    uint32_t cur_sqp_iters;
    T cur_tracking_error;
    int control_update_step;

    // For potentially recording fine-grained timing
    std::vector<double> ktt_time_vec, shur_time_vec, dz_time_vec, line_search_time_vec;
    std::vector<double> cur_ktt_time_vec, cur_shur_time_vec, cur_dz_time_vec, cur_line_search_time_vec;

    // mpc iterates
    T *d_lambda, *d_eePos_goal, *d_xu, *d_xu_old;
    gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu_old, traj_len*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eePos_goal, 6*knot_points*sizeof(T)));
    gpuErrchk(cudaMemset(d_lambda, 0, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMemcpy(d_eePos_goal, d_eePos_traj, 6*knot_points*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu_old, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


    void *d_dynmem = gato_plant::initializeDynamicsConstMem<T>();


    // temp host memory
    T h_xs[state_size];
    gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
    tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));    
    gpuErrchk(cudaPeekAtLastError());
    T h_eePos[6];
    T h_eePos_goal[6];


    // temp device memory
    T *d_eePos;
    gpuErrchk(cudaMalloc(&d_eePos, 6*sizeof(T)));

#if LINSYS_SOLVE == 1
    pcg_config<T> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = linsys_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;
#endif

    T rho = 1e-3;
    T rho_reset = 1e-3;

#if REMOVE_JITTERS
	#if LINSYS_SOLVE == 1
    config.pcg_exit_tol = 1e-11;
    config.pcg_max_iter = 10000;
    
    for(int j = 0; j < 100; j++){
        sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, 1e-3);
        gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    rho = 1e-3;
    config.pcg_exit_tol = linsys_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;
	#else
    for(int j = 0; j < 100; j++){
        sqpSolveQdldl<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, rho, 1e-3);
        gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    rho = 1e-3;
	#endif

#endif // #if REMOVE_JITTERS



    //
    // MPC tracking loop
    //
    for(control_update_step = 0; control_update_step < max_control_updates; control_update_step++){
        

        if (traj_offset == traj_steps){ break; }



#if LIVE_PRINT_PATH
        grid::end_effector_positions_kernel<T><<<1,128,144*sizeof(T)>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
        gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < 6; i++){
            std::cout << h_eePos[i] << (i < 5 ? " " : "\n");
        }
#endif // #if LIVE_PRINT_PATH
        


#if LINSYS_SOLVE == 1
        sqp_stats = sqpSolvePcg<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, config, rho, rho_reset);
#else 
	    sqp_stats = sqpSolveQdldl<T>(state_size, control_size, knot_points, timestep, d_eePos_goal, d_lambda, d_xu, d_dynmem, rho, rho_reset);
#endif

        cur_linsys_iters = std::get<0>(sqp_stats);
        cur_linsys_times = std::get<1>(sqp_stats);
        sqp_solve_time_us = std::get<2>(sqp_stats);
        cur_sqp_iters = std::get<3>(sqp_stats);
        sqp_exits.push_back(std::get<4>(sqp_stats));
        cur_linsys_exits = std::get<5>(sqp_stats);

    #if LINSYS_SOLVE == 1
        // For FINE_GRAINED_TIMING
        // @Felix weird location ... but we just copy what they did in the original code
        cur_ktt_time_vec = std::get<6>(sqp_stats),
        cur_shur_time_vec = std::get<7>(sqp_stats),
        cur_dz_time_vec = std::get<8>(sqp_stats),
        cur_line_search_time_vec = std::get<9>(sqp_stats),
    #endif // LINSYS_SOLVE == 1


#if CONST_UPDATE_FREQ
        simulation_time = SIMULATION_PERIOD;
#else
        simulation_time = sqp_solve_time_us;
#endif
        

        // simulate traj for current solve time, offset by previous solve time
        simple_simulate<T>(state_size, control_size, knot_points, d_xs, d_xu_old, d_dynmem, timestep, prev_simulation_time, simulation_time);

        // old xu = new xu
        gpuErrchk(cudaMemcpy(d_xu_old, d_xu, traj_len*sizeof(T), cudaMemcpyDeviceToDevice));


        time_since_timestep += simulation_time * 1e-6;

        // if shift_threshold% through timestep
        if (!shifted && time_since_timestep > shift_threshold){
            
            // record tracking error
            grid::end_effector_positions_kernel<T><<<1,128,144*sizeof(T)>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
            gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
            cur_tracking_error = 0.0;
            for(uint32_t i=0; i < 3; i++){
                cur_tracking_error += abs(h_eePos[i] - h_eePos_goal[i]);
            }
            // std::cout << cur_tracking_error << std::endl;;
            tracking_errors.push_back(cur_tracking_error);                                            
            
            traj_offset++;

            // shift xu
            just_shift<T>(state_size, control_size, knot_points, d_xu);             // shift everything over one
            if (traj_offset + knot_points < traj_steps){
                // if within precomputed traj, fill in last state, control with precompute
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - (state_size + control_size)], &d_xu_traj[(state_size+control_size)*traj_offset - control_size], sizeof(T)*(state_size+control_size), cudaMemcpyDeviceToDevice));     // last state filled from precomputed trajectory
            }
            else{
                // fill in last state with goal position, zero velocity, last control with zero control
                gpuErrchk(cudaMemcpy(&d_xu[traj_len - state_size], &d_xu_traj[(traj_steps-1)*(state_size+control_size)], (state_size/2)*sizeof(T), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemset(&d_xu[traj_len - state_size / 2], 0, (state_size/2) * sizeof(T)));
                gpuErrchk(cudaMemset(&d_xu[traj_len - (state_size+control_size)], 0, control_size * sizeof(T)));
            }
            
            // shift goal
            just_shift(6, 0, knot_points, d_eePos_goal);
            if (traj_offset + knot_points < traj_steps){
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_offset+knot_points-1) * (6)], 6*sizeof(T), cudaMemcpyDeviceToDevice));
            }
            else{
                // fill in last goal state with goal state and zero velocity
                gpuErrchk(cudaMemcpy(&d_eePos_goal[(knot_points-1)*(6)], &d_eePos_traj[(traj_steps-1)*(6)], (6)*sizeof(T), cudaMemcpyDeviceToDevice));
                // gpuErrchk(cudaMemset(&d_eePos_goal[(knot_points-1)*(6) + state_size / 2], 0, (state_size/2) * sizeof(T)));
            }
            
            // shift lambda
            just_shift(state_size, 0, knot_points, d_lambda);
                // gpuErrchk(cudaMemset(&lambdas[i][state_size*(knot_points-1)], 0, state_size*sizeof(T)));
            
            shifted = true;
        }

        if (time_since_timestep > timestep){
            // std::cout << "shifted to offset: " << traj_offset + 1 << std::endl;
            shifted = false;
            time_since_timestep = std::fmod(time_since_timestep, timestep);
        }
        gpuErrchk(cudaMemcpy(d_xu, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToDevice));


        
        prev_simulation_time = simulation_time;

        gpuErrchk(cudaPeekAtLastError());

        
        // record data
        linsys_iters.insert(linsys_iters.end(), cur_linsys_iters.begin(), cur_linsys_iters.end());                      // linsys iters
        linsys_times.insert(linsys_times.end(), cur_linsys_times.begin(), cur_linsys_times.end());          // linsys times
        linsys_exits.insert(linsys_exits.end(), cur_linsys_exits.begin(), cur_linsys_exits.end());
        gpuErrchk(cudaMemcpy(h_xs, d_xs, state_size*sizeof(T), cudaMemcpyDeviceToHost));
        tracking_path.push_back(std::vector<T>(h_xs, &h_xs[state_size]));                                   // next state
        sqp_times.push_back(sqp_solve_time_us);
        sqp_iters.push_back(cur_sqp_iters);

    #if LINSYS_SOLVE == 1
        // For FINE_GRAINED_TIMING
        ktt_time_vec.insert(ktt_time_vec.end(), cur_ktt_time_vec.begin(), cur_ktt_time_vec.end());
        shur_time_vec.insert(shur_time_vec.end(), cur_shur_time_vec.begin(), cur_shur_time_vec.end());
        dz_time_vec.insert(dz_time_vec.end(), cur_dz_time_vec.begin(), cur_dz_time_vec.end());
        line_search_time_vec.insert(line_search_time_vec.end(), cur_line_search_time_vec.begin(), cur_line_search_time_vec.end());
    #endif // LINSYS_SOLVE == 1


#if LIVE_PRINT_STATS
        if (control_update_step % 1000 == 50){
            for (uint32_t i = 0; i < state_size; i++){
                std::cout << h_xs[i] << (i < state_size-1 ? " " : "\n");
            }
    #if TIME_LINSYS == 1
            std::cout << "linear system solve time:" << std::endl;
            printStats<double>(&linsys_times);
    #endif // #if TIME_LINSYS
            std::cout << "goal offset [" << traj_offset << "]\n";
            std::cout << "sqp iters" << std::endl;
            printStats<uint32_t>(&sqp_iters);
            std::cout << "sqp times" << std::endl;
            printStats<double>(&sqp_times);
            
            int totalOnes = std::accumulate(linsys_exits.begin(), linsys_exits.end(), 0);
            double max_iter_pct = (static_cast<double>(totalOnes) / linsys_exits.size());
            std::cout << "linsys exits for max iter: " << max_iter_pct * 100 << "% of the time\n";
            if (max_iter_pct > 0.5) {
               std::cout << "WARNING: PCG exiting for max iter over 50% of the time" << std::endl;
            }
            
            std::cout << "avg tracking error: " << std::accumulate(tracking_errors.begin(), tracking_errors.end(), 0.0f) / traj_offset << " current error: " << cur_tracking_error << "\n";
            std::cout << std::endl;

        }

#endif


    }
#if SAVE_DATA
    dump_tracking_data(
        &linsys_iters,
        &linsys_exits,
        &linsys_times,
        &sqp_times,
        &sqp_iters,
        &sqp_exits,
        &tracking_errors,
        &tracking_path,
        traj_offset,
        control_update_step,
        start_state_ind,
        goal_state_ind,
        test_iter,
        test_output_prefix,
        &ktt_time_vec,
        &shur_time_vec,
        &dz_time_vec,
        &line_search_time_vec
    );
#endif
    

    grid::end_effector_positions_kernel<T><<<1,128,144*sizeof(T)>>>(d_eePos, d_xs, grid::NUM_JOINTS, (grid::robotModel<T> *) d_dynmem, 1);
    gpuErrchk(cudaMemcpy(h_eePos, d_eePos, 6*sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_eePos_goal, d_eePos_goal, 6*sizeof(T), cudaMemcpyDeviceToHost));
    cur_tracking_error = 0.0;
    for(uint32_t i=0; i < 3; i++){
        cur_tracking_error += abs(h_eePos[i] - h_eePos_goal[i]);
    }

    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));
    gpuErrchk(cudaFree(d_eePos_goal));
    gpuErrchk(cudaFree(d_xu_old));

    gpuErrchk(cudaFree(d_eePos));

    #if TIME_LINSYS == 1 
        return std::make_tuple(linsys_times, tracking_errors, cur_tracking_error);
    #else
        return std::make_tuple(sqp_iters, tracking_errors, cur_tracking_error);
    #endif
}
