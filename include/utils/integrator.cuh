#pragma once
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include "dynamics/rbd_plant.cuh"
#include "glass.cuh"

namespace cgrps = cooperative_groups;

//---------- Utils ----------

/**
 * @brief Wraps an angle to range [-pi, pi]
 * @tparam T Data type of the angle
 * @param input Input angle
 * @return T Wrapped angle
 */
template<typename T>
__host__ __device__ 
T angleWrap(T input){
    const T pi = static_cast<T>(M_PI);
    if(input > pi) {input -= 2 * pi;}
    else if (input < -pi) {input += 2 * pi;}
    return input;
}

// ---------- Execution functions for integrator ----------

/**
 * @brief Executes system integrator to find next state and velocity
 * 
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator (0: Euler, 1: Semi-Implicit Euler)
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of the state vector
 * @param s_qkp1, s_qdkp1 next state, next velocity
 * @param s_q, s_qd, s_qdd linearized state, velocity, acceleration
 * @param dt Time step
 * @param block Thread block
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator(uint32_t state_size, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block){

    for (unsigned ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            s_qkp1[ind] = s_q[ind] + dt*s_qd[ind];
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
            s_qkp1[ind] = s_q[ind] + dt*s_qdkp1[ind];
        }
        else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){
            s_qkp1[ind] = angleWrap(s_qkp1[ind]);
        }
    }
}

/**
* @brief Execute integrator to find gradients of next state wrt current state and control
* @tparam T Data type
* @tparam INTEGRATOR_TYPE Type of integrator (0: Euler, 1: Semi-Implicit Euler)
* @param state_size Size of the state vector
* @param control_size Size of the control vector
* @param s_Ak, s_Bk df
* @param s_dqdd Acceleration gradient
* @param dt Time step
* @param block Thread block
*/
template <typename T, unsigned INTEGRATOR_TYPE = 0>
__device__
void exec_integrator_gradient(uint32_t state_size, uint32_t control_size, T *s_Ak, T *s_Bk, T *s_dqdd, T dt, cgrps::thread_block block){
        
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;

    // and finally A and B
    if (INTEGRATOR_TYPE == 0){
        // then apply the euler rule -- xkp1 = xk + dt*dxk thus AB = [I_{state},0_{control}] + dt*dxd
        // where dxd = [ 0, I, 0; dqdd/dq, dqdd/dqd, dqdd/du]
        for (unsigned ind = thread_id; ind < state_size*(state_size + control_size); ind += block_dim){
            int c = ind / state_size; int r = ind % state_size;
            T *dst = (c < state_size)? &s_Ak[ind] : &s_Bk[ind - state_size*state_size]; // dst
            T val = (r == c) * static_cast<T>(1); // first term (non-branching)
            val += (r < state_size/2 && r == c - state_size/2) * dt; // first dxd term (non-branching)
            if(r >= state_size/2) { val += dt * s_dqdd[c*state_size/2 + r - state_size/2]; }
            ///TODO: EMRE why didn't this error before?
            // val += (r >= state_size/2) * dt * s_dqdd[c*state_size/2 + r - state_size/2]; // second dxd term (non-branching)
            *dst = val;
        }
    }
    else if (INTEGRATOR_TYPE == 1){
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1 = qk + dt*qdk + dt^2*qddk
        // dxkp1 = [Ix | 0u ] + dt*[[0q, Iqd, 0u] + dt*dqdd
        //                                             dqdd]
        // Ak = I + dt * [[0,I] + dt*dqdd/dx; dqdd/dx]
        // Bk = [dt*dqdd/du; dqdd/du]
        for (unsigned ind = thread_id; ind < state_size*state_size; ind += block_dim){
            int c = ind / state_size; int r = ind % state_size; int rdqdd = r % (state_size/2);
            T dtVal = static_cast<T>((r == rdqdd)*dt + (r != rdqdd));
            s_Ak[ind] = static_cast<T>((r == c) + dt*(r == c - state_size/2)) +
                        dt * s_dqdd[c*state_size/2 + rdqdd] * dtVal;
            if(c < control_size){
                s_Bk[ind] = dt * s_dqdd[state_size*state_size/2 + c*state_size/2 + rdqdd] * dtVal;
            }
        }
    }
    else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}
}

/**
* @brief Executes system integrator and computes error
* @tparam T Data type
* @tparam INTEGRATOR_TYPE Type of integrator (0: Euler, 1: Semi-Implicit Euler)
* @tparam ANGLE_WRAP Whether to wrap angles
* @param state_size Size of the state vector
* @param s_err Output error
* @param s_qkp1, s_qdkp1 Next state
* @param s_q, s_qd, s_qdd Current state and acceleration
* @param dt Time step
* @param block Thread block
* @param absval Whether to compute absolute value of error
*/
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator_error(uint32_t state_size, T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block, bool absval = false){
    
    T new_qkp1; T new_qdkp1;
    for (unsigned ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            new_qkp1 = s_q[ind] + dt*s_qd[ind];
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
            new_qkp1 = s_q[ind] + dt*new_qdkp1;
        } else {printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){ printf("ANGLE_WRAP!\n");
            new_qkp1 = angleWrap(new_qkp1);
        }

        // then computre error
        if(absval){
            s_err[ind] = abs(s_qkp1[ind] - new_qkp1);
            s_err[ind + state_size/2] = abs(s_qdkp1[ind] - new_qdkp1);    
        } else {
            s_err[ind] = s_qkp1[ind] - new_qkp1;
            s_err[ind + state_size/2] = s_qdkp1[ind] - new_qdkp1;
        }
        // printf("err[%f] with new qkp1[%f] vs orig[%f] and new qdkp1[%f] vs orig[%f] with qk[%f] qdk[%f] qddk[%f] and dt[%f]\n",s_err[ind],new_qkp1,s_qkp1[ind],new_qdkp1,s_qdkp1[ind],s_q[ind],s_qd[ind],s_qdd[ind],dt);
    }
}

// ---------- Integrator for simulation ----------

/**
 * @brief Compute forward dynamics and integrate to find next state
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of the state vector
 * @param s_xkp1 Output next state
 * @param s_xuk Input state and control
 * @param s_temp Temporary storage
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 * @param block Thread block
 */
 template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
 __device__ 
 void integrator(uint32_t state_size, T *s_xkp1, T *s_xuk, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
     T *s_q = s_xuk; 					
     T *s_qd = s_q + state_size/2; 				
     T *s_u = s_qd + state_size/2;
     T *s_qkp1 = s_xkp1; 				
     T *s_qdkp1 = s_qkp1 + state_size/2;
 
     T *s_qdd = s_temp; 					
     T *s_extra_temp = s_qdd + state_size/2;
 
     //first compute qdd
     gato_plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
     block.sync();
     exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block);
 }


// ---------- Integrator and gradients for KKT system ----------

/**
 * @brief Integrates to find next state and computes the gradient of next state wrt current state and control
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @tparam COMPUTE_INTEGRATOR_ERROR Whether to compute integrator error
 * @param state_size Size of the state vector
 * @param control_size Size of the control vector
 * @param s_xux Input state and control
 * @param s_Ak, s_Bk Output matrices
 * @param s_xnew_err Output new state or error
 * @param s_temp Temporary storage, size: (state_size/2*(state_size + control_size + 1) + DYNAMICS_TEMP)
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 * @param block Thread block
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void integratorAndGradient(uint32_t state_size, uint32_t control_size, T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
    
    T *s_q = s_xux; 	
    T *s_qd = s_q + state_size/2; 		
    T *s_u = s_qd + state_size/2;
    
    T *s_qdd = s_temp; // linearized acceleration
    T *s_dqdd = s_qdd + state_size/2;
    T *s_extra_temp = s_dqdd + (state_size/2)*(state_size+control_size);

    // first compute qdd and dqdd
    gato_plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    block.sync();

    // then compute xnew or error
    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xux[state_size+control_size], &s_xux[state_size+control_size+state_size/2], s_q, s_qd, s_qdd, dt, block);
    } else {
        exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xnew_err[state_size/2], s_q, s_qd, s_qdd, dt, block);
    }
    
    // then compute gradients to form Ak and Bk
    exec_integrator_gradient<T,INTEGRATOR_TYPE>(state_size, control_size, s_Ak, s_Bk, s_dqdd, dt, block);
}

// ---------- Integrator and error computation for line search ----------

/**
 * @brief Computes error after integrating the system
 * 
 * s_temp of size: (3*state_size/2 + DYNAMICS_TEMP)
 *
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of the state vector
 * @param s_xuk Input state and control
 * @param s_xkp1 Next state
 * @param s_temp Temporary storage
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 * @param block Thread block
 * @return T Computed error
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
T integratorError(uint32_t state_size, T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){

    T *s_q = s_xuk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;

    T *s_qdd = s_temp; 					
    T *s_err = s_qdd + state_size/2;
    T *s_extra_temp = s_err + state_size/2;

    // first compute qdd
    gato_plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();

    // then apply the integrator and compute error
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block, true);
    block.sync();

    // finish off forming the error
    glass::reduce<T>(state_size, s_err);
    block.sync();
    // if(GATO_LEAD_THREAD){printf("in integratorError with reduced error of [%f]\n",s_err[0]);}
    return s_err[0];
}

// ---------- Integrator kernel for host ----------

/**
 * @brief kernel for integrating directly from host
 * @tparam T Data type
 * @tparam INTEGRATOR_TYPE Type of integrator
 * @tparam ANGLE_WRAP Whether to wrap angles
 * @param state_size Size of the state vector
 * @param control_size Size of the control vector
 * @param d_xkp1 Output next state
 * @param d_xuk Input state and control
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 */
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void integrator_kernel(uint32_t state_size, uint32_t control_size, T *d_xkp1, T *d_xuk, void *d_dynMem_const, T dt){
    extern __shared__ T s_smem[];
    T *s_xkp1 = s_smem;
    T *s_xuk = s_xkp1 + state_size; 
    T *s_temp = s_xuk + state_size + control_size;
    cgrps::thread_block block = cgrps::this_thread_block();	  
    cgrps::grid_group grid = cgrps::this_grid();
    for (unsigned ind = threadIdx.x; ind < state_size + control_size; ind += blockDim.x){
        s_xuk[ind] = d_xuk[ind];
    }

    block.sync();
    integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xkp1, s_xuk, s_temp, d_dynMem_const, dt, block);
    block.sync();

    for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
        d_xkp1[ind] = s_xkp1[ind];
    }
}



/**
 * @brief Host function for integrator
 *
 * take start state from h_xs, and control input from h_xu, and update h_xs
 *
 * @tparam T Data type
 * @param state_size Size of the state vector
 * @param control_size Size of the control vector
 * @param d_xs State vector
 * @param d_xu State and control vector
 * @param d_dynMem_const Dynamics memory
 * @param dt Time step
 */
template <typename T>
void integrator_host(uint32_t state_size, uint32_t control_size, T *d_xs, T *d_xu, void *d_dynMem_const, T dt){
    // T *d_xu;
    // T *d_xs_new;
    // gpuErrchk(cudaMalloc(&d_xu, xu_size));
    // gpuErrchk(cudaMalloc(&d_xs_new, xs_size));

    // gpuErrchk(cudaMemcpy(d_xu, h_xs, state_size*sizeof(T), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_xu + state_size, h_xu + state_size, control_size*sizeof(T), cudaMemcpyHostToDevice));
    //TODO: needs sync?

    const size_t integrator_kernel_smem_size = sizeof(T)*(2*state_size + control_size + state_size/2 + gato_plant::forwardDynamics_TempMemSize_Shared());
    //TODO: one block one thread? Why?
    integrator_kernel<T><<<1,1, integrator_kernel_smem_size>>>(state_size, control_size, d_xs, d_xu, d_dynMem_const, dt);

    //TODO: needs sync?
    // gpuErrchk(cudaMemcpy(h_xs, d_xs_new, xs_size, cudaMemcpyDeviceToHost));

    // gpuErrchk(cudaFree(d_xu));
    // gpuErrchk(cudaFree(d_xs_new));
}
