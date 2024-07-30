#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <tuple>
#include <filesystem>
#include "mpcsim.cuh"
#include "dynamics/rbd_plant.cuh"
#include "settings.cuh"
#include "utils/experiment.cuh"
#include "gpu_pcg.cuh"


int main(){

    constexpr uint32_t state_size = grid::NUM_JOINTS*2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = 32;
    const linsys_t timestep = .015625;
    const uint32_t traj_length = (state_size + control_size) * knot_points - control_size;

    // checks GPU space for pcg
    checkPcgOccupancy<linsys_t>((void *) pcg<linsys_t, state_size, knot_points>, PCG_NUM_THREADS, state_size, knot_points);    

    void *d_dynmem_const = gato_plant::initializeDynamicsConstMem<linsys_t>();

    pcg_config<linsys_t> config;
    config.pcg_block = PCG_NUM_THREADS; //128
    config.pcg_max_iter = PCG_MAX_ITER; //173 (?)

    std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, bool, std::vector<bool>> sqp_stats;

    linsys_t *d_eePos_traj, *d_xu, *d_lambda;

    uint32_t num_exit_vals = 5;
    float pcg_exit_vals[num_exit_vals];
    pcg_exit_vals[0] = 5e-6;
    pcg_exit_vals[1] = 7.5e-6;
    pcg_exit_vals[2] = 5e-6;
    pcg_exit_vals[3] = 2.5e-6;
    pcg_exit_vals[4] = 1e-6;



    for (uint32_t pcg_exit_ind = 0; pcg_exit_ind < num_exit_vals; pcg_exit_ind++){

        float pcg_exit_tol = pcg_exit_vals[pcg_exit_ind];
        config.pcg_exit_tol = pcg_exit_tol;

        linsys_t rho = 1e-3;
        linsys_t rho_reset = 1e-3;

        std::vector<linsys_t> h_eePos_traj;
        std::vector<linsys_t> h_xu_traj;
        //read in precomputed end effector position
        auto eePos_traj2d = readCSVToVecVec<linsys_t>("examples/trajfiles/figure8_traj_eePos_meters.csv"); //eePos_traj2d is a 2D vector of 6 positions for all knot points
        for (const auto& vec : eePos_traj2d) { h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end()); } //flatten 2D vector into 1D vector
        //now onto state and control trajectory
        auto xu_traj2d = readCSVToVecVec<linsys_t>("examples/trajfiles/figure8_traj_xu_input.csv"); //xu_traj2d is a 2D vector of states and controls for all knot points
        for (const auto& xu_vec : xu_traj2d) { h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end()); }

        if(eePos_traj2d.size() < knot_points){ std::cout << "precomputed traj length < knotpoints, not implemented\n"; exit(1); }


        
        gpuErrchk(cudaMalloc(&d_eePos_traj, 6 * knot_points * sizeof(linsys_t)));
        gpuErrchk(cudaMemcpy(d_eePos_traj, h_eePos_traj.data(), 6 * knot_points * sizeof(linsys_t), cudaMemcpyHostToDevice)); // initialize with start of trajectory
        
        gpuErrchk(cudaMalloc(&d_xu, traj_length * sizeof(linsys_t)));
        gpuErrchk(cudaMemcpy(d_xu, h_xu_traj.data(), traj_length*sizeof(linsys_t), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc(&d_lambda, state_size * knot_points * sizeof(linsys_t)));
        gpuErrchk(cudaMemset(d_lambda, 0, state_size * knot_points * sizeof(linsys_t)));

        
        sqp_stats = sqpSolvePcg<linsys_t>(state_size, control_size, 32, timestep, d_eePos_traj, d_lambda, d_xu, d_dynmem_const, config, rho, rho_reset);

        auto [pcg_iters, linsys_times, sqp_solve_time, sqp_iters, sqp_exit, pcg_exits] = sqp_stats;

        printf("Results for exit tol %f:\n", pcg_exit_tol);
        printf("PCG iters: %d\n", pcg_iters);
        printf("SQP iters: %d\n", sqp_iters);
        printf("SQP solve time: %f\n", sqp_solve_time);
        printf("PCG exits: ");
        for (int i = 0; i < pcg_exits.size(); i++){
            printf("%d ", pcg_exits[i]);
        }
        printf("\n");
        printf("PCG times: ");
        for (int i = 0; i < linsys_times.size(); i++){
            printf("%f ", linsys_times[i]);
        }   
        printf("\n");

        printf("freeing memory\n");
        gato_plant::freeDynamicsConstMem<linsys_t>(d_dynmem_const);

        gpuErrchk(cudaFree(d_xu));
        gpuErrchk(cudaFree(d_eePos_traj));
        gpuErrchk(cudaFree(d_lambda));
        gpuErrchk(cudaPeekAtLastError());
        
        std::cout << "Completed at " << getCurrentTimestamp() << std::endl;

	}

    return 0;
}


