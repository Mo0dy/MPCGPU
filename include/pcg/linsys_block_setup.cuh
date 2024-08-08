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
void complete_SS_Pinv_block_blockrow(uint32_t state_size, uint32_t knot_points,
                                     T *d_Sdb, T *d_Sob, T *d_Pinvdb, T *d_Pinvob, T *d_T,
                                     T *s_temp, unsigned blockrow) {

    const uint32_t states_sq = state_size * state_size;
    const uint32_t triangular_state = (state_size + 1) * state_size / 2;

    // shared block memory usage: 3nx^2 + nx(nx+1)/2 + 2nx

    T *s_T_k = s_temp;
    T *s_T_km1 = s_T_k;                         // T_k and T_km1 share (nx+1)nx/2
    T *s_phi_k = s_T_km1 + triangular_state;
    T *s_phi_kp1_T = s_phi_k;                   // phi_k and phi_kp1_T share nx^2
    T *s_O_k = s_phi_kp1_T + states_sq;
    T *S_O_km1_T = s_O_k;                       // O_k and O_km1_T share nx^2
    T *s_DInv_k = S_O_km1_T + states_sq;
    T *s_DInv_km1 = s_DInv_k + state_size;      // DInv_k is not shared, size = nx
    T *s_DInv_kp1 = s_DInv_km1;                 // DInv_km1 and DInv_kp1 share nx
    T *s_PhiInv_k_R = s_DInv_kp1 + state_size;
    T *s_PhiInv_k_L = s_PhiInv_k_R;             // PhiInv_k_R and PhiInv_k_L share nx^2
    T *s_scratch = s_PhiInv_k_L + states_sq;

    const unsigned lastrow = knot_points - 1;

    // load s_DInv_k
    load_block_db<T>(state_size, knot_points,
                     d_Pinvdb,       // src
                     s_DInv_k,       // dst
                     blockrow        // blockrow
    );

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
        __syncthreads();
        glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1), s_T_k, s_phi_kp1_T, s_O_k);

        // save s_O_k to S (right diagonal)
        store_block_ob<T>(state_size, knot_points,
                          s_O_k,        // src
                          d_Sob,        // dst
                          1,            // block column (0 or 1)
                          blockrow,     // blockrow
                          1             // positive
        );

        // load s_DInv_kp1
        load_block_db<T>(state_size, knot_points,
                         d_Pinvdb,      // src
                         s_DInv_kp1,    // dst
                         blockrow + 1   // blockrow
        );

        __syncthreads();

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
                          -1                // negative
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

        // compute s_O_km1_T
        __syncthreads();
        glass::trmm_right<T, true>(state_size, state_size, static_cast<T>(1), s_T_km1, s_phi_k, S_O_km1_T);

        // save s_O_km1_T to S (left diagonal)
        store_block_ob<T>(state_size, knot_points,
                          S_O_km1_T,    // src
                          d_Sob,        // dst
                          0,            // block column (0 or 1)
                          blockrow,     // blockrow
                          1             // positive
        );

        // load s_DInv_km1
        load_block_db<T>(state_size, knot_points,
                         d_Pinvdb,      // src
                         s_DInv_km1,    // dst
                         blockrow - 1   // blockrow
        );

        __syncthreads();

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
                          -1                // negative
        );
    }

}

// this function assumes d_G, d_C, d_g, d_c are dense and ready
// this function fills in
//      diagonal blocks of d_S, and off-diagonal blocks of d_S partially
//      diagonal blocks of d_Pinv
//      all of d_gamma, d_T

// if chol_or_ldl, use chol
// if !chol_or_ldl, use LDL'

template<typename T>
__device__
void form_S_gamma_and_jacobi_Pinv_block_blockrow(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                                                 T *d_G, T *d_C, T *d_g, T *d_c,
                                                 T *d_Sdb, T *d_Sob, T *d_Pinvdb, T *d_Pinvob, T *d_T, T *d_gamma,
                                                 T rho, T *s_temp, unsigned blockrow, bool chol_or_ldl) {

    // note: kkt.cuh stores Ak, Bk with minus sign, so the Ak, Bk here are actually -Ak, -Bk

    const uint32_t triangular_state = (1 + state_size) * state_size / 2;
    const uint32_t state_sq = state_size * state_size;
    const uint32_t control_sq = control_size * control_size;
    const uint32_t state_dot_control = state_size * control_size;

    // shared block memory usage: 4nx^2 + 3nx + 1

    T *s_M1 = s_temp;
    T *s_M2 = s_M1 + state_sq;          // 4 matrices of size nx^2
    T *s_M3 = s_M2 + state_sq;
    T *s_M4 = s_M3 + state_sq;
    T *s_v1 = s_M4 + state_sq;
    T *s_v2 = s_v1 + state_size;        // 3 vectors of size nx
    T *s_v3 = s_v2 + state_size + 1;    // +1 is very important due to matrix inversion needs 2nx+1 tmp
    T *s_end = s_v3 + state_size;

    if (blockrow == 0) {

        // need to save Q0 to S and Q0_i to Pinv
        // need to save gamma_0

        // load Q0 to M1
        glass::copy<T>(state_sq, d_G, s_M1);
        __syncthreads();
        add_identity(s_M1, state_size, rho);

        // invert M1 = Q0, M2 <- inv(M1) = Q0_i
        loadIdentity<T>(state_size, s_M2);
        __syncthreads();
        invertMatrix<T>(state_size, s_M1, s_v1);

        // load q0 to v1
        glass::copy<T>(state_size, d_g, s_v1);

        // compute Q0^{-1}q0  - IntegratorError in gamma
        // v2 <- M2 * v1 = Q0_i * q0
        __syncthreads();
        mat_vec_prod<T>(state_size, state_size, s_M2, s_v1, s_v2);

        // v2 <- v2 - d_c = Q0_i * q0 - d_c
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_v2[i] -= d_c[(blockrow * state_size) + i];
        }

        // Attention: loading of Q0 and computation of gamma_0 is finished
        // v2 = gamma_0, M2 = Q0_i

        // do Cholesky or LDL' here on M2 = Q0_i
        // note: Q0_i is dense but its lower triangular part is L0
        // note: v1 = \tilde{D}_k is diagonal
        if (chol_or_ldl) {
            // Cholesky, with square root
            glass::chol_InPlace<T>(state_size, s_M2);
            // v1 identity
            for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
                s_v1[i] = static_cast<T>(1);
            }
        } else {
            // square root free Cholesky = LDL'
            glass::ldl_InPlace<T>(state_size, s_M2, s_v1);
        }

        // save v1 = \tilde{D}_k to main diagonal S
        store_block_db<T>(state_size, knot_points,
                          s_v1,
                          d_Sdb,
                          blockrow,
                          1
        );

        // invert the unit lower triangular L0
        // T0 = M3 <- inv(M2) = inv(L0)
        glass::loadIdentityTriangular<T>(state_size, s_M3);
        __syncthreads();
        glass::trsm_triangular<T, true>(state_size, s_M2, s_M3);

        // save M3 = T0 to d_T
        __syncthreads();
        for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
            d_T[ind] = s_M3[ind];
        }

        // v1 <- inv(v1), calculate inv(\tilde{D}_0)
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_v1[i] = static_cast<T>(1) / s_v1[i];
        }

        // save v1 = inv(\tilde{D}_0) to main diagonal Pinv
        __syncthreads();
        store_block_db<T>(state_size, knot_points,
                          s_v1,
                          d_Pinvdb,
                          blockrow,
                          1
        );

        // v3 <- M3 * v2 = T0 * gamma_0
        glass::trmv<T, false>(state_size, static_cast<T>(1), s_M3, s_v2, s_v3);

        // save v3 = T0 * gamma_0 in gamma
        __syncthreads();
        for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x) {
            d_gamma[ind] = s_v3[ind];
        }
    } else {                       // blockrow!=LEAD_BLOCK

        const unsigned C_set_size = state_sq + state_dot_control;
        const unsigned G_set_size = state_sq + control_sq;

        //-------------------------- Qkp1 & qkp1 starts --------------------------

        // load Qkp1 to M1
        glass::copy<T>(state_sq, d_G + (blockrow * G_set_size), s_M1);
        __syncthreads();
        add_identity(s_M1, state_size, rho);

        // load identity to M2
        loadIdentity<T>(state_size, s_M2);

        // invert M1, M2 <- inv(M1) = Qkp1_i
        __syncthreads();
        invertMatrix<T>(state_size, s_M1, s_v1);

        if (blockrow == knot_points - 1) {
            // save M2 = Qkp1_i to d_G (now Ginv) for calculating dz
            __syncthreads();
            glass::copy<T>(state_sq, s_M2, d_G + (blockrow) * G_set_size);
        }

        // save M2 = Qkp1_i to M4
        __syncthreads();
        glass::copy<T>(state_sq, s_M2, s_M4);

        // load qkp1 to v1
        glass::copy<T>(state_size, d_g + (blockrow) * (state_size + control_size), s_v1);

        // v3 <- M2 * v1 = Qkp1_i * qkp1
        __syncthreads();
        mat_vec_prod<T>(state_size, state_size, s_M2, s_v1, s_v3);

        //-------------------------- Qkp1 & qkp1 ends --------------------------

        // as of now, M4 = Qkp1_i
        //            v3 = Qkp1_i * qkp1

        //-------------------------- Rk & rk & -Bk starts --------------------------

        // load Rk to M1. Note M1 is declared as nx * nx but here only nu * nu is used.
        glass::copy<T>(control_sq, d_G + ((blockrow - 1) * G_set_size + state_sq),
                       s_M1);
        __syncthreads();
        add_identity(s_M1, control_size, rho);

        // load -Bk to M3. Note M3 is declared as nx * nx but here only nx * nu is used.
        glass::copy<T>(state_dot_control, d_C + (blockrow - 1) * C_set_size + state_sq, s_M3);

        // load identity to M2_tmp. Address of M2 is changed to M2_tmp due to size change of M1.
        T *s_M2_tmp = s_M1 + control_sq;
        loadIdentity<T>(control_size, s_M2_tmp);

        // invert M1, M2_tmp <- inv(M1) = Rk_i
        __syncthreads();
        invertMatrix<T>(control_size, s_M1, s_v1);

        // load rk to v1. Note v1 is declared as nx but here only nu is used.
        glass::copy<T>(control_size, d_g + ((blockrow - 1) * (state_size + control_size) + state_size), s_v1);

        // save M2_tmp = Rk_i to d_G (now Ginv) for calculating dz
        glass::copy<T>(control_sq, s_M2_tmp,
                       d_G + (blockrow - 1) * G_set_size + state_sq);

        // M1 <- M3 * M2_tmp = -Bk * Rk_i
        __syncthreads();
        glass::gemm<T>(state_size, control_size, control_size, static_cast<T>(1.0), s_M3, s_M2_tmp, s_M1);

        // v2 <- M1 * v1 = -Bk * Rk_i * rk
        __syncthreads();
        mat_vec_prod<T>(state_size, control_size, s_M1, s_v1, s_v2);

        // v3 <- v3 + v2 = Qkp1_i * qkp1 - Bk * Rk_i * rk
        __syncthreads();
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_v3[i] += s_v2[i];
        }

        // M2 <- M1 * M3' = Bk * Rk_i * Bk'
        glass::gemm<T, true>(state_size, control_size, state_size, static_cast<T>(1.0), s_M1, s_M3, s_M2);

        // M4 <- M4 + M2 = Qkp1_i + Bk * Rk_i * Bk'
        __syncthreads();
        for (unsigned i = threadIdx.x; i < state_sq; i += blockDim.x) {
            s_M4[i] += s_M2[i];
        }

        //-------------------------- Rk & rk & -Bk ends --------------------------

        // as of now, M4 = Qkp1_i + Bk * Rk_i * Bk'
        //            v3 = Qkp1_i * qkp1 - Bk * Rk_i * rk

        //-------------------------- Qk & qk * -Ak starts --------------------------

        // load Qk to M1
        glass::copy<T>(state_sq, d_G + (blockrow - 1) * G_set_size, s_M1);
        __syncthreads();
        add_identity(s_M1, state_size, rho);

        // laod -Ak to M3
        glass::copy<T>(state_sq, d_C + (blockrow - 1) * C_set_size, s_M3);

        // load identity to M2
        loadIdentity<T>(state_size, s_M2);

        // invert M1, M2 <- inv(M1) = Qk_i
        __syncthreads();
        invertMatrix<T>(state_size, s_M1, s_v1);

        // load qk to v1
        __syncthreads();
        glass::copy<T>(state_size, d_g + (blockrow - 1) * (state_size + control_size), s_v1);

        // save M2 = Qk_i to d_G (now Ginv) for calculating dz
        glass::copy<T>(state_sq, s_M2, d_G + (blockrow - 1) * G_set_size);

        // M1 <- M3 * M2 = -Ak * Qk_i
        __syncthreads();
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M3, s_M2, s_M1);

        // v2 <- M1 * v1 = -Ak * Qk_i * qk
        __syncthreads();
        mat_vec_prod<T>(state_size, state_size, s_M1, s_v1, s_v2);

        // v3 <- v3 + v2 = Qkp1_i * qkp1 - Bk * Rk_i * rk - Ak * Qk_i * qk
        __syncthreads();
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_v3[i] += s_v2[i];
        }

        // v3 <- v3 - d_c = Qkp1_i * qkp1 - Bk * Rk_i * rk - Ak * Qk_i * qk - d_c
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_v3[i] -= d_c[(blockrow * state_size) + i];
        }

        // M2 <- M1 * M3' = Ak * Qk_i * Ak'
        glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M3, s_M2);

        // M4 <- M4 + M2 = Qkp1_i + Bk * Rk_i * Bk' + Ak * Qk_i * Ak'
        __syncthreads();
        for (unsigned i = threadIdx.x; i < state_sq; i += blockDim.x) {
            s_M4[i] += s_M2[i];
        }

        //-------------------------- Qk & qk * -Ak ends --------------------------

        // as of now, M4 = Qkp1_i + Bk * Rk_i * Bk' + Ak * Qk_i * Ak' = theta_k
        //            v3 = Qkp1_i * qkp1 - Bk * Rk_i * rk - Ak * Qk_i * qk - d_c = gamma_k
        //            M1 = -Ak * QK_i = phi_k
        // Attention: computation of phi_k, theta_k, gamma_k is finished

        __syncthreads();
        // do Cholesky or LDL' here on M4
        // note: M4 is dense but its lower triangular part is Lk
        // note: v1 = \tilde{D}_k is a diagonal matrix

        if (chol_or_ldl) {
            // Cholesky, with square root
            glass::chol_InPlace<T>(state_size, s_M4);
            // v1 identity
            for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
                s_v1[i] = static_cast<T>(1);
            }
        } else {
            // square root free Cholesky = LDL'
            glass::ldl_InPlace<T>(state_size, s_M4, s_v1);
        }

        // save v1 = \tilde{D}_k to main diagonal S
        store_block_db<T>(state_size, knot_points,
                          s_v1,
                          d_Sdb,
                          blockrow,
                          1
        );

        // invert the unit lower triangular Lk
        // M2 <- Tk = inv(Lk)
        glass::loadIdentityTriangular<T>(state_size, s_M2);
        __syncthreads();
        glass::trsm_triangular<T, true>(state_size, s_M4, s_M2);

        // save M2 = Tk to d_T
        __syncthreads();
        for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
            unsigned offset = blockrow * triangular_state + ind;
            d_T[offset] = s_M2[ind];
        }

        // v1 <- inv(v1)
        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            s_v1[i] = static_cast<T>(1) / s_v1[i];
        }

        // save v1 = inv(\tilde{D}_k) to main diagonal Pinv
        __syncthreads();
        store_block_db<T>(state_size, knot_points,
                          s_v1,
                          d_Pinvdb,
                          blockrow,
                          1
        );

        // v1 <- M2 * v3 = Tk * gamma_k
        glass::trmv<T, false>(state_size, static_cast<T>(1), s_M2, s_v3, s_v1);

        // save v1 = Tk * gamma_k to d_gamma
        __syncthreads();
        for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x) {
            unsigned offset = (blockrow) * state_size + ind;
            d_gamma[offset] = s_v1[ind];
        }

        // M3 <- M2 * M1 = Tk * phi_k
        glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1), s_M2, s_M1, s_M3);
        __syncthreads();

        // save M3 = Tk * phi_k into left off-diagonal of S
        store_block_ob<T>(state_size, knot_points,
                          s_M3,         // src
                          d_Sob,        // dst
                          0,            // col = 0 or 1
                          blockrow,     // blockrow
                          1             // positive
        );

        // load identity to M1
        loadIdentity<T>(state_size, s_M1);
        __syncthreads();
        // M2 <- M1 * M3' = I * (Tk * phi_k)' = phi_k' * Tk'
        glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M3, s_M2);

        // save phi_k' * T_k' into right off-diagonal of S
        __syncthreads();
        store_block_ob<T>(state_size, knot_points,
                          s_M2,         // src
                          d_Sob,        // dst
                          1,            // col = 0 or 1
                          blockrow - 1, // blockrow
                          1             // positive
        );
    }
}


template<typename T>
__global__
void form_S_gamma_Pinv_block_kernel(
        uint32_t state_size, uint32_t control_size, uint32_t knot_points,
        T *d_G, T *d_C, T *d_g, T *d_c,
        T *d_Sdb, T *d_Sob, T *d_Pinvdb, T *d_Pinvob, T *d_T, T *d_gamma,
        T rho, bool chol_or_ldl) {

    extern __shared__ T s_temp[];

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        form_S_gamma_and_jacobi_Pinv_block_blockrow<T>(
                state_size, control_size, knot_points,
                d_G, d_C, d_g, d_c,
                d_Sdb, d_Sob, d_Pinvdb, d_Pinvob, d_T, d_gamma,
                rho, s_temp, blockrow, chol_or_ldl);
    }
    cgrps::this_grid().sync();

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        complete_SS_Pinv_block_blockrow<T>(
                state_size, knot_points,
                d_Sdb, d_Sob, d_Pinvdb, d_Pinvob, d_T,
                s_temp, blockrow);
    }
}

template<typename T>
__global__
void transform_lambda_kernel(uint32_t state_size, uint32_t knot_points,
                             T *d_T, T *d_lambda) {

    extern __shared__ T s_mem[];
    const uint32_t triangular_state = (state_size + 1) * state_size / 2;

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {

        // shared block memory usage: nx(nx+1)/2 + 2nx

        T *s_T_k = s_mem;
        T *s_lambda_k = s_T_k + triangular_state;
        T *s_lambdaNew_k = s_lambda_k + state_size;
        T *s_end = s_lambdaNew_k + state_size;

        glass::copy<T>(triangular_state, d_T + blockrow * triangular_state, s_T_k);
        glass::copy<T>(state_size, d_lambda + blockrow * state_size, s_lambda_k);

        // calculate T_k' * lambda_k
        __syncthreads();
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
void transform_lamdba(uint32_t state_size, uint32_t knot_points,
                      T *d_T, T *d_lambda) {

    const uint32_t s_temp_size = sizeof(T) * (state_size * (state_size + 1) / 2 + 2 * state_size);
    transform_lambda_kernel<<<knot_points, DZ_THREADS, s_temp_size>>>(
            state_size, knot_points,
            d_T, d_lambda);
}

template<typename T>
void form_schur_system_block(
        uint32_t state_size, uint32_t control_size, uint32_t knot_points,
        T *d_G_dense, T *d_C_dense, T *d_g, T *d_c,
        T *d_Sdb, T *d_Sob, T *d_Pinvdb, T *d_Pinvob, T *d_T, T *d_gamma,
        T rho, bool chol_or_ldl) {
    // old shared block memory size                     7nx^2 + nx(nx+1)/2 + 8nx + nxnu + 3nu + 2nu^2 +3
    // form_S_gamma_and_jacobi_Pinv_block_blockrow      4nx^2 + 3nx + 1
    // complete_SS_Pinv_block_blockrow                  3nx^2 + nx(nx+1)/2 + 2nx

    // it is assumed here that nx > nu always holds
    const uint32_t s_temp_size = sizeof(T) * (4 * state_size * state_size +
                                              3 * state_size + 1);

    void *kernel = (void *) form_S_gamma_Pinv_block_kernel<T>;
    void *args[] = {
            (void *) &state_size, (void *) &control_size, (void *) &knot_points,
            (void *) &d_G_dense, (void *) &d_C_dense, (void *) &d_g, (void *) &d_c,
            (void *) &d_Sdb, (void *) &d_Sob, (void *) &d_Pinvdb, (void *) &d_Pinvob, (void *) &d_T, (void *) &d_gamma,
            (void *) &rho, (void *) &chol_or_ldl
    };

    gpuErrchk(cudaLaunchCooperativeKernel(kernel, knot_points, SCHUR_THREADS, args, s_temp_size));
}