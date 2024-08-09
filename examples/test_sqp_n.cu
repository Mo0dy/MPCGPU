// track_iiwa_pcg_n.cu
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>
#include "sqp/sqp_pcg_DEV.cuh"
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "utils/io.cuh"
#include "gpu_pcg.cuh"

void print_test_config_n(const uint32_t solve_count){
    std::cout << "\nSolve count: " << solve_count << "\n";
    std::cout << "Knot points: " << KNOT_POINTS << "\n";
    std::cout << "State size: " << STATE_SIZE << "\n";
    std::cout << "Datatype: " << (USE_DOUBLES ? "DOUBLE" : "FLOAT") << "\n";
    std::cout << "SQP exit condition: " << (CONST_UPDATE_FREQ ? "CONSTANT TIME" : "CONSTANT ITERS") << "\n";
#if CONST_UPDATE_FREQ
    std::cout << "Max sqp time: " << SQP_MAX_TIME_US << "\n";
#else
    std::cout << "Max sqp iter: " << SQP_MAX_ITER << "\n";
#endif
    std::cout << "Max pcg iter: " << PCG_MAX_ITER << "\n";
    std::cout << "\n\n";
}

int main(){
    std::cout << "Testing Multisolve\n\n";
    // ----------------- Constants -----------------
    const uint32_t solve_count = 1;
    constexpr uint32_t state_size = grid::NUM_JOINTS*2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = 32;
    const linsys_t timestep = .015625; // 1/64 s
    const uint32_t traj_size = (state_size + control_size) * knot_points - control_size;

    pcg_config<linsys_t> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = 5e-6;      //1e-5, 7.5e-6, 5e-6, 2.5e-6, 1e-6
    config.pcg_max_iter = PCG_MAX_ITER;
    checkPcgOccupancy<linsys_t>((void *) pcg<linsys_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);   // TODO: change for batched PCG solver
    
    print_test_config_n(solve_count);

    // ----------------- Device Memory -----------------

    void *d_dynmem = gato_plant::initializeDynamicsConstMem<linsys_t>();

    linsys_t *d_eePos_trajs; 
    linsys_t *d_xu_trajs; 
    linsys_t *d_lambdas;
    linsys_t *d_rhos;
    gpuErrchk(cudaMalloc(&d_eePos_trajs, 6 * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_xu_trajs, traj_size * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_lambdas, state_size * knot_points * solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMalloc(&d_rhos, solve_count * sizeof(linsys_t)));
    gpuErrchk(cudaMemset(d_rhos, 1e-3, solve_count * sizeof(linsys_t)));

    //read in input goal end effector position trajectory
    auto eePos_traj2d = readCSVToVecVec<linsys_t>("trajfiles/0_0_eepos.traj"); 
    auto xu_traj2d = readCSVToVecVec<linsys_t>("trajfiles/0_0_traj.csv"); 
    if(eePos_traj2d.size() < knot_points){ std::cout << "precomputed traj length < knotpoints, not implemented\n"; return 1; }

    std::vector<std::vector<linsys_t>> h_eePos_trajs(solve_count);
    std::vector<std::vector<linsys_t>> h_xu_trajs(solve_count);
    for (uint32_t i = 0; i < solve_count; ++i) { // Duplicate the trajectory data for each solve (for now)
        
        for (int j = 0; j < KNOT_POINTS; ++j) {
            h_eePos_trajs[i].insert(h_eePos_trajs[i].end(), eePos_traj2d[j].begin(), eePos_traj2d[j].end());
            h_xu_trajs[i].insert(h_xu_trajs[i].end(), xu_traj2d[j].begin(), xu_traj2d[j].end());
        }   

        //copy to device
        gpuErrchk(cudaMemcpy(d_eePos_trajs + i * 6, h_eePos_trajs[i].data(), 6 * knot_points * sizeof(linsys_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_xu_trajs + i * traj_size, h_xu_trajs[i].data(), traj_size * sizeof(linsys_t), cudaMemcpyHostToDevice));
    }

    // ----------------- SOLVE -----------------

    std::tuple<std::vector<std::vector<int>>, std::vector<double>, float, std::vector<uint32_t>, std::vector<char>, std::vector<std::vector<bool>>> sqp_stats;
    sqp_stats = sqpSolvePcgN<linsys_t>(solve_count, state_size, control_size, knot_points, timestep, d_eePos_trajs, d_lambdas, d_xu_trajs, d_dynmem, config, d_rhos, 1e-3);

    // ----------------- Get Results -----------------

    //copy xu trajectories back to host
    std::vector<std::vector<linsys_t>> h_xu_trajs_out(solve_count);
    for (uint32_t i = 0; i < solve_count; ++i) {
        h_xu_trajs_out[i].resize(traj_size);
        gpuErrchk(cudaMemcpy(h_xu_trajs_out[i].data(), d_xu_trajs + i * traj_size, traj_size * sizeof(linsys_t), cudaMemcpyDeviceToHost));
    }

    std::vector<std::vector<int>> pcg_iters_matrix(solve_count);
    std::vector<double> pcg_times_vec;
    float sqp_solve_time = 0.0;
    std::vector<uint32_t> sqp_iterations_vec(solve_count);
    std::vector<char> sqp_exit_vec(solve_count);
    std::vector<std::vector<bool>> pcg_exits_matrix(solve_count);

    pcg_iters_matrix = std::get<0>(sqp_stats);
    pcg_times_vec = std::get<1>(sqp_stats);
    sqp_solve_time = std::get<2>(sqp_stats);
    sqp_iterations_vec = std::get<3>(sqp_stats);
    sqp_exit_vec = std::get<4>(sqp_stats);
    pcg_exits_matrix = std::get<5>(sqp_stats);

    // print everything
    std::cout << "\n\nResults:\n";
    for (uint32_t i = 0; i < solve_count; ++i) {
        std::cout << "Solve " << i + 1  << ":\n";
        std::cout << "SQP iterations: " << sqp_iterations_vec[i] << "\n";
        std::cout << "SQP exit: " << sqp_exit_vec[i] << "\n";
        std::cout << "PCG exits: ";
        for (unsigned long j = 0; j < pcg_exits_matrix[i].size(); ++j) {
            std::cout << pcg_exits_matrix[i][j] << " ";
        }
        std::cout << "\nPCG iters: ";
        for (unsigned long j = 0; j < pcg_iters_matrix[i].size(); ++j) {
            std::cout << pcg_iters_matrix[i][j] << " ";
        }
        //print updated xu trajectory
        std::cout << "\n-----\n";
    }
    std::cout << "\n\nPCG times: ";
    for (unsigned long i = 0; i < pcg_times_vec.size(); ++i) {
        std::cout <<  i << ": " << pcg_times_vec[i] << "us   ";
    }
    std::cout << "\n";
    std::cout << "SQP solve time: " << sqp_solve_time << "us\n";

    // ----------------- Free Memory -----------------
    gato_plant::freeDynamicsConstMem<linsys_t>(d_dynmem);

    gpuErrchk(cudaFree(d_xu_trajs));
    gpuErrchk(cudaFree(d_eePos_trajs));
    gpuErrchk(cudaFree(d_lambdas));
    gpuErrchk(cudaFree(d_rhos));
    
    gpuErrchk(cudaPeekAtLastError());

    return 0;

}



