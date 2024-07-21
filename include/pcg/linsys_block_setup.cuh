#pragma once

#include <cstdint>
#include "gpuassert.cuh"
#include "glass.cuh"
#include "utils/matrix.cuh"

// d_Pinv: diagonal blocks ready, off-diagonal blocks to be computed
// d_S: diagonal blocks ready, off-diagonal blocks are half-ready
// d_T: all ready
// d_gamma: not used, so not passed in

template<typename T>
__device__
void
complete_SS_Pinv_block_blockrow(uint32_t state_size, uint32_t knot_points,
                                T *d_Sdb, T *d_Sob, T *d_Pinvdb, T *d_Pinvob, T *d_T,
                                T *s_temp,
                                unsigned blockrow) {

    const uint32_t states_sq = state_size * state_size;
    const uint32_t triangular_state = (state_size + 1) * state_size / 2;

    T *s_T_k = s_temp;
    T *s_T_km1 = s_T_k + triangular_state;
    T *s_phi_k = s_T_km1 + triangular_state;
    T *s_phi_kp1_T = s_phi_k + states_sq;
    T *s_O_k = s_phi_kp1_T + states_sq;
    T *S_O_km1_T = s_O_k + states_sq;
    T *s_DInv_k = S_O_km1_T + states_sq;
    T *s_DInv_km1 = s_DInv_k + state_size;
    T *s_DInv_kp1 = s_DInv_km1 + state_size;
    T *s_PhiInv_k_R = s_DInv_kp1 + state_size;
    T *s_PhiInv_k_L = s_PhiInv_k_R + states_sq;
    T *s_scratch = s_PhiInv_k_L + states_sq;

    const unsigned lastrow = knot_points - 1;

    if (blockrow != lastrow) {
        // load s_T_k
        for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
            unsigned offset = blockrow * triangular_state + ind;
            s_T_k[ind] = d_T[offset];
        }
        // load s_phi_kp1_T
        load_block_ob<T>(state_size, knot_points,
                         d_Sob,              // src
                         s_phi_kp1_T,        // dst
                         0,                  // block column (0 or 1)
                         blockrow + 1,       // blockrow
                         true                // transpose
        );

        // compute s_O_k
        __syncthreads();//----------------------------------------------------------------
        glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1), s_T_k, s_phi_kp1_T, s_O_k);

        // save s_O_k to S (right diagonal)
        store_block_ob<T>(state_size, knot_points,
                          s_O_k,        // src
                          d_Sob,        // dst
                          1,            // block column (0 or 1)
                          blockrow,     // blockrow
                          -1            // negative
        );

        // load s_DInv_kp1
        load_block_db<T>(state_size, knot_points,
                         d_Pinvdb,      // src
                         s_DInv_kp1,    // dst
                         blockrow + 1   // blockrow
        );
    }

    if (blockrow != 0) {
        // load s_T_km1
        for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
            unsigned offset = (blockrow - 1) * triangular_state + ind;
            s_T_km1[ind] = d_T[offset];
        }
        // load phi_k
        load_block_ob<T>(state_size, knot_points,
                         d_Sob,     // src
                         s_phi_k,   // dst
                         0,         // block column (0 or 1)
                         blockrow   // block row
        );

        // compute S_O_km1_T
        __syncthreads();//----------------------------------------------------------------
        glass::trmm_right<T, true>(state_size, state_size, static_cast<T>(1), s_T_km1, s_phi_k, S_O_km1_T);

        // save s_O_km1_T to S (left diagonal)
        store_block_ob<T>(state_size, knot_points,
                          S_O_km1_T,    // src
                          d_Sob,        // dst
                          0,            // block column (0 or 1)
                          blockrow,     // blockrow
                          -1            // negative
        );

        // load s_DInv_km1
        load_block_db<T>(state_size, knot_points,
                         d_Pinvdb,      // src
                         s_DInv_km1,    // dst
                         blockrow - 1   // blockrow
        );
    }

    // load s_DInv_k
    load_block_db<T>(state_size, knot_points,
                     d_Pinvdb,       // src
                     s_DInv_k,       // dst
                     blockrow        // blockrow
    );

    if (blockrow != 0) {
        __syncthreads();//----------------------------------------------------------------

        // compute left off diag for Pinv
        // no need to __syncthreads() between 2 dimm because s_DInv_k & s_DInv_km1 are both diagonal so order doesn't matter
        // use s_phi_k for scratching
        glass::dimm_left<T>(state_size, state_size, static_cast<T>(1.0), s_DInv_k, S_O_km1_T, s_phi_k);
        glass::dimm_right<T>(state_size, state_size, static_cast<T>(1.0), s_DInv_km1, s_phi_k, s_PhiInv_k_L);

        // store left diagonal in Pinv
        store_block_ob<T>(state_size, knot_points,
                          s_PhiInv_k_L,     // src
                          d_Pinvob,         // dst
                          0,                // block column (0 or 1)
                          blockrow,         // blockrow
                          1                 // positive
        );
    }

    if (blockrow != lastrow) {
        __syncthreads();//----------------------------------------------------------------

        // calculate right diag in Pinv
        // no need to __syncthreads() between 2 dimm because s_DInv_k & s_DInv_kp1 are both diagonal so order doesn't matter
        // use s_phi_k for scratching
        glass::dimm_left<T>(state_size, state_size, static_cast<T>(1.0), s_DInv_k, s_O_k, s_phi_k);
        glass::dimm_right<T>(state_size, state_size, static_cast<T>(1.0), s_DInv_kp1, s_phi_k, s_PhiInv_k_R);

        // store right diagonal in Pinv
        store_block_ob<T>(state_size, knot_points,
                          s_PhiInv_k_R,     // src
                          d_Pinvob,         // dst
                          1,                // block column (0 or 1)
                          blockrow,         // blockrow
                          1                 // positive
        );
    }
}

// this function assumes d_G, d_C, d_g, d_c are dense and ready
// this function fills in
//      diagonal blocks of d_S, and off-diagonal blocks of d_S partially
//      diagonal blocks of d_Pinv
//      all of d_gamma, d_T

template<typename T>
__device__
void
form_S_gamma_and_jacobi_Pinv_block_blockrow(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                                            T *d_G, T *d_C, T *d_g, T *d_c,
                                            T *d_Sdb, T *d_Sob, T *d_Pinvdb, T *d_Pinvob, T *d_T,
                                            T *d_gamma, T rho, T *s_temp,
                                            unsigned blockrow) {

    //  SPACE ALLOCATION IN SHARED MEM
    //  | phi_k | theta_k | D_k |     T_k     | gamma_k | block-specific...
    //     s^2      s^2      s      s(s+1)/2       s
    uint32_t triangular_state = (1 + state_size) * state_size / 2;

    T *s_phi_k = s_temp;
    T *s_theta_k = s_phi_k + state_size * state_size;           // s_phi_k = O_k            nx^2
    T *s_D_k = s_theta_k + state_size * state_size;             // s_theta_k = D_k          nx^2
    T *s_T_k = s_D_k + state_size;                              // s_D_k = \tilde{D}_k      nx
    T *s_gamma_k = s_T_k + triangular_state;                    // s_T_k                    (nx+1)nx/2
    T *s_end_main = s_gamma_k + state_size;                     // s_gamma_k = gamma_k      nx

    if (blockrow == 0) {

        // need to save Q0 to S and Q0_i to Pinv
        // need to save gamma_0

        T *s_Q0 = s_end_main;
        T *s_Q0_i = s_Q0 + state_size * state_size;
        T *s_q0 = s_Q0_i + state_size * state_size;
        T *s_end = s_q0 + state_size;

        // scratch space
        T *s_extra_temp = s_end;

        glass::copy<T>(state_size * state_size, d_G, s_Q0);
        glass::copy<T>(state_size, d_g, s_q0);

        __syncthreads();//----------------------------------------------------------------
        add_identity(s_Q0, state_size, rho);

        // invert Q_0
        loadIdentity<T>(state_size, s_Q0_i);
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<T>(state_size, s_Q0, s_extra_temp);

        // compute Q0^{-1}q0  - IntegratorError in gamma
        __syncthreads();//----------------------------------------------------------------
        mat_vec_prod<T>(state_size, state_size,
                        s_Q0_i,
                        s_q0,
                        s_gamma_k
        );
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_gamma_k[i] -= d_c[(blockrow * state_size) + i];
        }

        // Attention: loading of Q0 and computation of gamma_k is finished

        // do LDL' here on s_theta_k = s_Q0_i = D_0 = D_k
        glass::ldl_InPlace<T>(state_size, s_Q0_i, s_D_k);
        // note: s_Q0_i is dense but its lower triangular part is Lk = L0
        // note: s_D_k = \tilde{D}_k is diagonal

        // save s_D_k = \tilde{D}_k to main diagonal S
        store_block_db<T>(state_size, knot_points,
                          s_D_k,
                          d_Sdb,
                          blockrow,
                          1
        );

        // invert the unit lower triangular Lk = L0
        // s_T_k = T_k = inv(Lk)
        glass::loadIdentityTriangular<T>(state_size, s_T_k);
        __syncthreads();//----------------------------------------------------------------
        glass::trsm_triangular<T, true>(state_size, s_Q0_i, s_T_k);

        // save s_T_k to d_T
        __syncthreads();//----------------------------------------------------------------
        for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
            d_T[ind] = s_T_k[ind];
        }

        // calculate inv(\tilde{D}_k)
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_D_k[i] = static_cast<T>(1) / s_D_k[i];
        }

        // save -inv(\tilde{D}_k) to main diagonal Pinv
        __syncthreads();//----------------------------------------------------------------
        store_block_db<T>(state_size, knot_points,
                          s_D_k,
                          d_Pinvdb,
                          blockrow,
                          1
        );

        // calculate T_k gamma_k
        glass::trmv<T, false>(state_size, static_cast<T>(1), s_T_k, s_gamma_k, s_extra_temp);

        // save T_k gamma_k in gamma
        __syncthreads();//----------------------------------------------------------------
        for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x) {
            d_gamma[ind] = s_extra_temp[ind];
        }
    } else {                       // blockrow!=LEAD_BLOCK

        const unsigned C_set_size = state_size * state_size + state_size * control_size;
        const unsigned G_set_size = state_size * state_size + control_size * control_size;

        //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
        //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
        //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

        T *s_Ak = s_end_main;
        T *s_Bk = s_Ak + state_size * state_size;
        T *s_Qk = s_Bk + state_size * control_size;
        T *s_Qk_i = s_Qk + state_size * state_size;
        T *s_Qkp1 = s_Qk_i + state_size * state_size;
        T *s_Qkp1_i = s_Qkp1 + state_size * state_size;
        T *s_Rk = s_Qkp1_i + state_size * state_size;
        T *s_Rk_i = s_Rk + control_size * control_size;
        T *s_qk = s_Rk_i + control_size * control_size;
        T *s_qkp1 = s_qk + state_size;
        T *s_rk = s_qkp1 + state_size;
        T *s_end = s_rk + control_size;

        // scratch
        T *s_extra_temp = s_end;

        glass::copy<T>(state_size * state_size, d_C + (blockrow - 1) * C_set_size, s_Ak);
        glass::copy<T>(state_size * control_size, d_C + (blockrow - 1) * C_set_size + state_size * state_size, s_Bk);
        // note: since kkt.cuh stores Ak, Bk with minus sign, s_Ak & s_Bk are actually -Ak, -Bk
        glass::copy<T>(state_size * state_size, d_G + (blockrow - 1) * G_set_size, s_Qk);
        glass::copy<T>(state_size * state_size, d_G + (blockrow * G_set_size), s_Qkp1);
        glass::copy<T>(control_size * control_size, d_G + ((blockrow - 1) * G_set_size + state_size * state_size),
                       s_Rk);
        glass::copy<T>(state_size, d_g + (blockrow - 1) * (state_size + control_size), s_qk);
        glass::copy<T>(state_size, d_g + (blockrow) * (state_size + control_size), s_qkp1);
        glass::copy<T>(control_size, d_g + ((blockrow - 1) * (state_size + control_size) + state_size), s_rk);

        __syncthreads();//----------------------------------------------------------------

        add_identity(s_Qk, state_size, rho);
        add_identity(s_Qkp1, state_size, rho);
        add_identity(s_Rk, control_size, rho);

        // Invert Q, Qp1, R 
        loadIdentity<T>(state_size, state_size, control_size,
                        s_Qk_i,
                        s_Qkp1_i,
                        s_Rk_i
        );
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<T>(state_size, state_size, control_size, state_size,
                        s_Qk,
                        s_Qkp1,
                        s_Rk,
                        s_extra_temp
        );

        // save Qk_i into G (now Ginv) for calculating dz
        glass::copy<T>(state_size * state_size, s_Qk_i, d_G + (blockrow - 1) * G_set_size);

        // save Rk_i into G (now Ginv) for calculating dz
        glass::copy<T>(control_size * control_size, s_Rk_i,
                       d_G + (blockrow - 1) * G_set_size + state_size * state_size);

        if (blockrow == knot_points - 1) {
            // save Qkp1_i into G (now Ginv) for calculating dz
            glass::copy<T>(state_size * state_size, s_Qkp1_i, d_G + (blockrow) * G_set_size);
        }

        // Compute -AQ^{-1} in phi
        __syncthreads();//----------------------------------------------------------------
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_Ak, s_Qk_i, s_phi_k);

        // Compute -BR^{-1} in Qkp1
        glass::gemm<T>(state_size, control_size, control_size, static_cast<T>(1.0), s_Bk, s_Rk_i, s_Qkp1);

        // compute Q_{k+1}^{-1}q_{k+1} - IntegratorError in gamma
        mat_vec_prod<T>(state_size, state_size,
                        s_Qkp1_i,
                        s_qkp1,
                        s_gamma_k
        );
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_gamma_k[i] -= d_c[(blockrow * state_size) + i];
        }

        // compute -AQ^{-1}q for gamma         temp storage in extra temp
        __syncthreads();//----------------------------------------------------------------
        mat_vec_prod<T>(state_size, state_size,
                        s_phi_k,
                        s_qk,
                        s_extra_temp
        );

        // compute -BR^{-1}r for gamma           temp storage in extra temp + states
        mat_vec_prod<T>(state_size, control_size,
                        s_Qkp1,
                        s_rk,
                        s_extra_temp + state_size
        );

        // gamma computation done
        __syncthreads();//----------------------------------------------------------------
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_gamma_k[i] += s_extra_temp[state_size + i] + s_extra_temp[i];
        }

        // compute -AQ^{-1}AT for theta
        glass::gemm<T, true>(
                state_size,
                state_size,
                state_size,
                static_cast<T>(1.0),
                s_phi_k,
                s_Ak,
                s_theta_k
        );

        // add Qkp1^{-1} to theta
        __syncthreads();//----------------------------------------------------------------
        for (unsigned i = threadIdx.x; i < state_size * state_size; i += blockDim.x) {
            s_theta_k[i] += s_Qkp1_i[i];
        }

        // compute -BR^{-1}BT for theta            temp storage in QKp1{-1}
        __syncthreads();//----------------------------------------------------------------
        glass::gemm<T, true>(
                state_size,
                control_size,
                state_size,
                static_cast<T>(1.0),
                s_Qkp1,
                s_Bk,
                s_Qkp1_i
        );

        // add -BR^{-1}BT to theta
        __syncthreads();//----------------------------------------------------------------
        for (unsigned i = threadIdx.x; i < state_size * state_size; i += blockDim.x) {
            s_theta_k[i] += s_Qkp1_i[i];
        }

        // Attention: computation of phi_k, theta_k, gamma_k is finished

        // do LDL' here on s_theta_k = D_k
        __syncthreads();//----------------------------------------------------------------
        glass::ldl_InPlace<T>(state_size, s_theta_k, s_D_k);
        // note: s_theta_k is dense but its lower triangular part is Lk
        // note: s_D_k = \tilde{D}_k is diagonal

        // save s_D_k = \tilde{D}_k to main diagonal S
        store_block_db<T>(state_size, knot_points,
                          s_D_k,
                          d_Sdb,
                          blockrow,
                          1
        );

        // invert the unit lower triangular Lk
        // s_T_k = T_k = inv(Lk)
        glass::loadIdentityTriangular<T>(state_size, s_T_k);
        __syncthreads();//----------------------------------------------------------------
        glass::trsm_triangular<T, true>(state_size, s_theta_k, s_T_k);

        // save s_T_k to d_T
        __syncthreads();//----------------------------------------------------------------
        for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
            unsigned offset = blockrow * triangular_state + ind;
            d_T[offset] = s_T_k[ind];
        }

        // calculate inv(\tilde{D}_k)
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_D_k[i] = static_cast<T>(1) / s_D_k[i];
        }

        // save -inv(\tilde{D}_k) to main diagonal Pinv
        __syncthreads();//----------------------------------------------------------------
        store_block_db<T>(state_size, knot_points,
                          s_D_k,
                          d_Pinvdb,
                          blockrow,
                          1
        );

        // calculate T_k gamma_k
        glass::trmv<T, false>(state_size, static_cast<T>(1), s_T_k, s_gamma_k, s_extra_temp);

        // save T_k gamma_k in gamma
        __syncthreads();//----------------------------------------------------------------
        for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x) {
            unsigned offset = (blockrow) * state_size + ind;
            d_gamma[offset] = s_extra_temp[ind];
        }

        // save T_k * phi_k into left off-diagonal of S
        glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1), s_T_k, s_phi_k, s_Qkp1);
        __syncthreads();//----------------------------------------------------------------
        store_block_ob<T>(state_size, knot_points,
                          s_Qkp1,                        // src
                          d_Sob,                          // dst
                          0,                              // col = 0 or 1
                          blockrow,                       // blockrow
                          -1
        );

        // transpose phi_k
        loadIdentity<T>(state_size, s_Ak);
        __syncthreads();//----------------------------------------------------------------
        glass::gemm<T, true>(
                state_size,
                state_size,
                state_size,
                static_cast<T>(1.0),
                s_Ak,
                s_phi_k,
                s_Qkp1
        );

        // save phi_k_T * T_k' into right off-diagonal of S
        __syncthreads();//----------------------------------------------------------------
        glass::trmm_right<T, true>(state_size, state_size, static_cast<T>(1), s_T_k, s_Qkp1, s_phi_k);
        __syncthreads();//----------------------------------------------------------------
        store_block_ob<T>(state_size, knot_points,
                          s_phi_k,                       // src
                          d_Sob,                         // dst
                          1,                             // col = 0 or 1
                          blockrow - 1,                  // blockrow
                          -1
        );
    }
}


template<typename T>
__global__
void form_S_gamma_Pinv_block_kernel(
        uint32_t state_size,
        uint32_t control_size,
        uint32_t knot_points,
        T *d_G,
        T *d_C,
        T *d_g,
        T *d_c,
        T *d_Sdb,
        T *d_Sob,
        T *d_Pinvdb,
        T *d_Pinvob,
        T *d_T,
        T *d_gamma,
        T rho
) {

    extern __shared__ T s_temp[];

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        form_S_gamma_and_jacobi_Pinv_block_blockrow<T>(
                state_size,
                control_size,
                knot_points,
                d_G,
                d_C,
                d_g,
                d_c,
                d_Sdb,
                d_Sob,
                d_Pinvdb,
                d_Pinvob, // should not be used
                d_T,
                d_gamma,
                rho,
                s_temp,
                blockrow
        );
    }
    cgrps::this_grid().sync();

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        complete_SS_Pinv_block_blockrow<T>(
                state_size, knot_points,
                d_Sdb,
                d_Sob,
                d_Pinvdb,
                d_Pinvob,
                d_T,
                s_temp,
                blockrow
        );
    }
}

template<typename T>
__global__
void transform_lambda_kernel(uint32_t state_size,
                             uint32_t knot_points,
                             T *d_T,
                             T *d_lambda) {

    extern __shared__ T s_mem[];
    const uint32_t triangular_state = (state_size + 1) * state_size / 2;

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {

        T *s_T_k = s_mem;
        T *s_lambda_k = s_T_k + triangular_state;
        T *s_lambdaNew_k = s_lambda_k + state_size;
        T *s_end = s_lambdaNew_k + state_size;

        glass::copy<T>(triangular_state, d_T + blockrow * triangular_state, s_T_k);
        glass::copy<T>(state_size, d_lambda + blockrow * state_size, s_lambda_k);

        // calculate T_k' * lambda_k
        __syncthreads();//----------------------------------------------------------------
        glass::trmv<T, true>(state_size, static_cast<T>(1), s_T_k, s_lambda_k, s_lambdaNew_k);

        // save s_lambdaNew_k to d_lambda
        for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x) {
            unsigned offset = blockrow * state_size + ind;
            d_lambda[offset] = s_lambdaNew_k[ind];
        }
    }
}

/*******************************************************************************
 *                           Interface Functions                                *
 *******************************************************************************/

template<typename T>
void transform_lamdba(uint32_t state_size,
                      uint32_t knot_points,
                      T *d_T,
                      T *d_lambda) {

    const uint32_t s_temp_size = sizeof(T) * (state_size * (state_size + 1) / 2 + 2 * state_size);
    transform_lambda_kernel<<<knot_points, DZ_THREADS, s_temp_size>>>(
            state_size,
            knot_points,
            d_T,
            d_lambda
    );
}

template<typename T>
void form_schur_system_block(
        uint32_t state_size,
        uint32_t control_size,
        uint32_t knot_points,
        T *d_G_dense,
        T *d_C_dense,
        T *d_g,
        T *d_c,
        T *d_Sdb,
        T *d_Sob,
        T *d_Pinvdb,
        T *d_Pinvob,
        T *d_T,
        T *d_gamma,
        T rho
) {
    const uint32_t s_temp_size = sizeof(T) * (7 * state_size * state_size +
                                              state_size * (state_size + 1) / 2 +
                                              8 * state_size +
                                              state_size * control_size +
                                              3 * control_size +
                                              2 * control_size * control_size +
                                              3);

    void *kernel = (void *) form_S_gamma_Pinv_block_kernel<T>;
    void *args[] = {
            (void *) &state_size,
            (void *) &control_size,
            (void *) &knot_points,
            (void *) &d_G_dense,
            (void *) &d_C_dense,
            (void *) &d_g,
            (void *) &d_c,
            (void *) &d_Sdb,
            (void *) &d_Sob,
            (void *) &d_Pinvdb,
            (void *) &d_Pinvob,
            (void *) &d_T,
            (void *) &d_gamma,
            (void *) &rho
    };

    gpuErrchk(cudaLaunchCooperativeKernel(kernel, knot_points, SCHUR_THREADS, args, s_temp_size));
}