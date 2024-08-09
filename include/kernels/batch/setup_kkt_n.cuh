
#include "settings.cuh"
#include "dynamics/rbd_plant.cuh"
#include "kernels/single/merit.cuh"
#include "kernels/single/setup_kkt.cuh"

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void generate_kkt_submatrices_batched(int solve_count, int knot_points,
                                      uint32_t state_size, 
                                      uint32_t control_size, 
                                      T *d_G_dense, 
                                      T *d_C_dense, 
                                      T *d_g, 
                                      T *d_c,
                                      void *d_dynMem_const, 
                                      T timestep,
                                      T *d_eePos_traj, 
                                      T *d_xs, 
                                      T *d_xu)
{
    const cgrps::thread_block block = cgrps::this_thread_block();
    int traj_id = blockIdx.y;
    int num_blocks = gridDim.x * gridDim.y;
    
    const uint32_t states_sq = state_size * state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t traj_size = states_s_controls * knot_points - control_size;
    const uint32_t G_size = (states_sq + controls_sq) * knot_points - controls_sq;
    const uint32_t C_size = (states_sq + states_p_controls) * (knot_points - 1);

    extern __shared__ T s_temp[];

    T *s_xux = s_temp;
    T *s_eePos_traj = s_xux + 2*state_size + control_size;
    T *s_Qk = s_eePos_traj + 6;
    T *s_Rk = s_Qk + states_sq;
    T *s_qk = s_Rk + controls_sq;
    T *s_rk = s_qk + state_size;
    T *s_end = s_rk + control_size;

    // Offset pointers for this trajectory
    d_G_dense += traj_id * G_size;
    d_C_dense += traj_id * C_size;
    d_g += traj_id * traj_size;
    d_c += traj_id * state_size * knot_points;
    d_eePos_traj += traj_id * 6 * knot_points;
    d_xs += traj_id * state_size;
    d_xu += traj_id * traj_size;

    for (unsigned knot_id = blockIdx.x + blockIdx.y * gridDim.x; knot_id < knot_points-1; knot_id+=num_blocks) {
        
        glass::copy<T>(2 * state_size + control_size, &d_xu[knot_id * states_s_controls], s_xux);
        glass::copy<T>(2 * 6, &d_eePos_traj[knot_id * 6], s_eePos_traj);

        block.sync();

        if (knot_id == knot_points - 2) {
            // Last knot point
            T* s_Ak = s_end;
            T* s_Bk = s_Ak + states_sq;
            T* s_Qkp1 = s_Bk + states_p_controls;
            T* s_qkp1 = s_Qkp1 + states_sq;
            T* s_integrator_error = s_qkp1 + state_size;
            T* s_extra_temp = s_integrator_error + state_size;

            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
                state_size, control_size,
                s_xux,
                s_Ak,
                s_Bk,
                s_integrator_error,
                s_extra_temp,
                d_dynMem_const,
                timestep,
                block
            );
            block.sync();
            
            gato_plant::trackingCostGradientAndHessian_lastblock<T>(
                state_size,
                control_size,
                s_xux,
                s_eePos_traj,
                s_Qk,
                s_qk,
                s_Rk,
                s_rk,
                s_Qkp1,
                s_qkp1,
                s_extra_temp,
                d_dynMem_const
            );
            block.sync();

            for (int i = threadIdx.x; i < state_size; i += blockDim.x) {
                d_c[i] = d_xu[i] - d_xs[i];
            }
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq + controls_sq) * knot_id]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq + controls_sq) * knot_id + states_sq]);
            glass::copy<T>(states_sq, s_Qkp1, &d_G_dense[(states_sq + controls_sq) * (knot_id + 1)]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls * knot_id]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls * knot_id + state_size]);
            glass::copy<T>(state_size, s_qkp1, &d_g[states_s_controls * (knot_id + 1)]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq + states_p_controls) * knot_id]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq + states_p_controls) * knot_id + states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size * (knot_id + 1)]);
        }
        else {
            // Not last knot point
            T* s_Ak = s_end;
            T* s_Bk = s_Ak + states_sq;
            T* s_integrator_error = s_Bk + states_p_controls;
            T* s_extra_temp = s_integrator_error + state_size;

            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
                state_size, control_size,
                s_xux,
                s_Ak,
                s_Bk,
                s_integrator_error,
                s_extra_temp,
                d_dynMem_const,
                timestep,
                block
            );
            block.sync();
            
            gato_plant::trackingCostGradientAndHessian<T>(
                state_size,
                control_size,
                s_xux,
                s_eePos_traj,
                s_Qk,
                s_qk,
                s_Rk,
                s_rk,
                s_extra_temp,
                d_dynMem_const
            );
            block.sync();


            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq + controls_sq) * knot_id]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq + controls_sq) * knot_id + states_sq]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls * knot_id]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls * knot_id + state_size]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq + states_p_controls) * knot_id]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq + states_p_controls) * knot_id + states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size * (knot_id + 1)]);
        }
    }
}

/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/

template <typename T>
void generate_kkt_system_batched(
    int solve_count,
    int knot_points,
    uint32_t state_size,
    uint32_t control_size,
    T *d_G_dense,
    T *d_C_dense,
    T *d_g,
    T *d_c,
    void *d_dynMem_const,
    T timestep,
    T *d_eePos_traj,
    T *d_xs,
    T *d_xu
){
    const uint32_t kkt_smem_size = 2 * get_kkt_smem_size<T>(state_size, control_size);
    dim3 block(KKT_THREADS);
    dim3 grid(knot_points, solve_count, 1);

    void *kernel = (void*)generate_kkt_submatrices_batched<T>;
    void *args[] = {
        &solve_count,
        &knot_points,
        &state_size,
        &control_size,
        &d_G_dense,
        &d_C_dense,
        &d_g,
        &d_c,
        &d_dynMem_const,
        &timestep,
        &d_eePos_traj,
        &d_xs,
        &d_xu
    };

    gpuErrchk(cudaLaunchKernel(kernel, grid, block, args, kkt_smem_size));
}







// // Function to get optimal launch configuration
// void get_optimal_launch_config(dim3& block, dim3& grid, int solve_count, int knot_points) {
//     cudaDeviceProp prop;
//     int device;
//     cudaGetDevice(&device);
//     cudaGetDeviceProperties(&prop, device);

//     int max_threads_per_block = prop.maxThreadsPerBlock;
//     int num_sms = prop.multiProcessorCount;
//     int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;

//     // Set block size
//     block.x = min(max_threads_per_block, 128);
//     block.y = 1;
//     block.z = 1;

//     // Set grid size
//     int max_blocks = num_sms * max_blocks_per_sm;
//     grid.x = min(knot_points, max_blocks);
//     grid.y = min(solve_count, max_blocks / grid.x);
//     grid.z = 1;
// }

// template <typename... Args>
// void launch_kernel(int solve_count, int knot_points, size_t shared_mem_size, cudaStream_t stream, Args... args) {
//     dim3 block, grid;
//     get_optimal_launch_config(block, grid, solve_count, knot_points);
    
//     printf("Launching with block: (%d, %d, %d), grid: (%d, %d, %d)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);

//     generate_kkt_submatrices_batched<<<grid, block, shared_mem_size, stream>>>(solve_count, knot_points, args...);
// }



















