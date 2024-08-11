#pragma once

#include <cstdint>
#include "gpuassert.cuh"
#include "glass.cuh"
#include "utils/matrix.cuh"

// d_Pinv: diagonal blocks ready, off-diagonal blocks to be computed
// d_S: diagonal blocks ready, off-diagonal blocks are half-ready (for TRANS) or ready (for ORG)
// d_T: all ready (for TRANS, ORG not applicable)
// d_gamma: not used, so not passed in

template<typename T>
__device__
void complete_SS_Pinv_block_blockrow(uint32_t state_size, uint32_t knot_points,
                                     T *d_S, T *d_Pinv, T *d_H, T *d_T,
                                     T *s_temp, unsigned blockrow, bool org_trans, bool use_H) {

    const uint32_t states_sq = state_size * state_size;
    const unsigned lastrow = knot_points - 1;

    if (org_trans) {
        // TRANS
        const uint32_t triangular_state = (state_size + 1) * state_size / 2;

        T *d_Sob = d_S + knot_points * state_size;
        T *d_Pinvdb = d_Pinv;
        T *d_Pinvob = d_Pinv + knot_points * state_size;

        // shared block memory usage: 4nx^2 + 3nx

        T *s_M1 = s_temp;
        T *s_M2 = s_M1 + states_sq;
        T *s_M3 = s_M2 + states_sq;
        T *s_M4 = s_M3 + states_sq;
        T *s_v1 = s_M4 + states_sq;
        T *s_v2 = s_v1 + state_size;
        T *s_v3 = s_v2 + state_size;
        T *s_end = s_v3 + state_size;

        // load tilde_Dk_inv to v1
        load_block_db<T>(state_size, knot_points,
                         d_Pinvdb,          // src
                         s_v1,              // dst
                         blockrow           // blockrow
        );

        if (blockrow != lastrow) {
            // load Tk to M1. Note only the first triangular_state part of M1 is used.
            for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
                unsigned offset = blockrow * triangular_state + ind;
                s_M1[ind] = d_T[offset];
            }
            // load Ok * Tkp1' to M2
            load_block_ob<T>(state_size, knot_points,
                             d_Sob,         // src
                             s_M2,          // dst
                             0,             // left block column
                             blockrow + 1,  // blockrow
                             true           // transpose
            );

            // M3 <- M1 * M2 = Tk * Ok * Tkp1' = tilde_Ok
            __syncthreads();
            glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1), s_M1, s_M2, s_M3);

            // save M3 = tilde_Ok to S (right diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M3,         // src
                              d_Sob,        // dst
                              1,            // right block column
                              blockrow,     // blockrow
                              1             // positive
            );

            // load tilde_Dkp1_inv to v2
            load_block_db<T>(state_size, knot_points,
                             d_Pinvdb,      // src
                             s_v2,          // dst
                             blockrow + 1   // blockrow
            );

            // M2 <- v1 * M3 (v1 is diagonal matrix) = tilde_Dk_inv * tilde_Ok
            glass::dimm_left<T>(state_size, state_size, static_cast<T>(1.0), s_v1, s_M3, s_M2);
            // M4 <- v2 * M2 (v2 is diagonal matrix) = tilde_Dk_inv * tilde_Ok * tilde_Dkp1_inv = tilde_Ek
            __syncthreads();
            glass::dimm_right<T>(state_size, state_size, static_cast<T>(1.0), s_v2, s_M2, s_M4);

            // save -M4 = -tilde_Ek to Pinv (right diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M4,         // src
                              d_Pinvob,     // dst
                              1,            // right block column
                              blockrow,     // blockrow
                              -1            // negative
            );
        }

        if (blockrow != 0) {
            // load Tkm1 to M1. Note only the first triangular_state part of M1 is used.
            for (unsigned ind = threadIdx.x; ind < triangular_state; ind += blockDim.x) {
                unsigned offset = (blockrow - 1) * triangular_state + ind;
                s_M1[ind] = d_T[offset];
            }
            // load Tk * Okm1' to M2
            load_block_ob<T>(state_size, knot_points,
                             d_Sob,         // src
                             s_M2,          // dst
                             0,             // left block column
                             blockrow       // block row
            );

            // M3 <- M2 * M1 = Tk * Okm1' * Tkm1' = tilde_Okm1'. Note trmm_right and transpose=true.
            __syncthreads();
            glass::trmm_right<T, true>(state_size, state_size, static_cast<T>(1), s_M1, s_M2, s_M3);

            // save M3 = tilde_Okm1' to S (left diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M3,         // src
                              d_Sob,        // dst
                              0,            // left block column
                              blockrow,     // blockrow
                              1             // positive
            );

            // load tilde_Dkm1_inv to v3
            load_block_db<T>(state_size, knot_points,
                             d_Pinvdb,      // src
                             s_v3,          // dst
                             blockrow - 1   // blockrow
            );

            // M2 <- v1 * M3 (v1 is diagonal matrix) = tilde_Dk_inv * tilde_Okm1'
            glass::dimm_left<T>(state_size, state_size, static_cast<T>(1.0), s_v1, s_M3, s_M2);
            // M4 <- v3 * M2 (v3 is diagonal matrix) = tilde_Dk_inv * tilde_Okm1' * tilde_Dkm1_inv = tilde_Ekm1'
            __syncthreads();
            glass::dimm_right<T>(state_size, state_size, static_cast<T>(1.0), s_v3, s_M2, s_M4);

            // save -M4 = -tilde_Ekm1' to Pinv (left diagonal)
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M4,         // src
                              d_Pinvob,     // dst
                              0,            // left block column
                              blockrow,     // blockrow
                              -1            // negative
            );
        }
    } else {
        // ORG
        // shared block memory usage: 4nx^2

        T *s_M1 = s_temp;
        T *s_M2 = s_M1 + states_sq;
        T *s_M3 = s_M2 + states_sq;
        T *s_M4 = s_M3 + states_sq;
        T *s_end = s_M4 + states_sq;

        if (blockrow != lastrow) {
            // load thetaInv_k = Dk_inv to M1
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow       // blockrow
            );

            // load phi_kp1_T = Ok to M2
            load_block_bd<T>(state_size, knot_points,
                             d_S,           // src
                             s_M2,          // dst
                             0,             // left block column
                             blockrow + 1,  // block row
                             true           // transpose
            );

            // M3 <- M1 * M2 = thetaInv_k * phi_kp1_T = Dk_inv * Ok
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M2, s_M3);

            // load thetaInv_kp1 = Dkp1_inv to M1
            __syncthreads();
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow + 1   // blockrow
            );

            // M4 <- M3 * M1 = thetaInv_k * phi_kp1_T * thetaInv_kp1
            // M4 = Ek = Dk_inv * Ok * Dkp1_inv
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M3, s_M1, s_M4);

            // save -M4 = -Ek to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M4,         // src
                              d_Pinv,       // dst
                              2,            // right block column
                              blockrow,     // blockrow
                              -1            // negative
            );

            if (use_H) {
                // M1 <- M4 * M2' = Ek * Ok'
                glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M1);

                // save M1 = Ek * Ok' to H, will be read again in the general case
                __syncthreads();
                store_block_bd<T>(state_size, knot_points,
                                  s_M1,     // src
                                  d_H,      // dst
                                  1,        // middle block column
                                  blockrow, // blockrow
                                  1         // positive
                );

                if (blockrow != lastrow - 1) {
                    // load phi_kp2_T = Okp1 to M2
                    load_block_bd<T>(state_size, knot_points,
                                     d_S,           // src
                                     s_M2,          // dst
                                     0,             // left block column
                                     blockrow + 2,  // block row
                                     true           // transpose
                    );

                    // M3 <- M4 * M2 = Ek * Okp1
                    __syncthreads();
                    glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M3);

                    // save M3 = Ek * Okp1 to H
                    __syncthreads();
                    store_block_bd<T>(state_size, knot_points,
                                      s_M3,         // src
                                      d_H,          // dst
                                      2,            // right block column
                                      blockrow,     // blockrow
                                      1             // positive
                    );
                }
            }
        }

        if (blockrow != 0) {
            // load thetaInv_k = Dk_inv to M1
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow       // blockrow
            );

            // load phi_k = Okm1_T to M2
            load_block_bd<T>(state_size, knot_points,
                             d_S,       // src
                             s_M2,      // dst
                             0,         // left block column
                             blockrow   // blockrow
            );

            // M3 <- M1 * M2 = thetaInv_k * phi_k = Dk_inv * Okm1_T
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M2, s_M3);

            // load thetaInv_km1 = Dkm1_inv to M1
            __syncthreads();
            load_block_bd<T>(state_size, knot_points,
                             d_Pinv,        // src
                             s_M1,          // dst
                             1,             // middle block column
                             blockrow - 1   // blockrow
            );

            // M4 <- M3 * M1 = thetaInv_k * phi_k * thetaInv_km1
            // M4 = Ekm1_T = Dk_inv * Okm1_T * Dkm1_inv
            __syncthreads();
            glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M3, s_M1, s_M4);

            // save -M4 = -Ekm1_T to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M4,     // src
                              d_Pinv,   // dst
                              0,        // left block column
                              blockrow, // blockrow
                              -1        // negative
            );

            if (use_H) {
                // M1 <- M4 * M2' = Ekm1_T * Okm1
                glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M1);

                if (blockrow != lastrow) {
                    // load M2 = Ek * Ok' (has been calculated before) from H
                    __syncthreads();
                    load_block_bd<T>(state_size, knot_points,
                                     d_H,       // src
                                     s_M2,      // dst
                                     1,         // middle block column
                                     blockrow   // blockrow
                    );
                    // M1 <- M1 + M2 = Ekm1_T * Okm1 + Ek * Ok'
                    for (unsigned ind = threadIdx.x; ind < states_sq; ind += blockDim.x) {
                        s_M1[ind] += s_M2[ind];
                    }
                }

                // save M1 = Ekm1_T * Okm1 (lastrow) or Ekm1_T * Okm1 + Ek * Ok' to H
                __syncthreads();
                store_block_bd<T>(state_size, knot_points,
                                  s_M1,     // src
                                  d_H,      // dst
                                  1,        // middle block column
                                  blockrow, // blockrow
                                  1         // positive
                );

                if (blockrow != 1) {
                    // load phi_km1 = Okm2_T to M2
                    load_block_bd<T>(state_size, knot_points,
                                     d_S,           // src
                                     s_M2,          // dst
                                     0,             // left block column
                                     blockrow - 1   // blockrow
                    );

                    // M3 <- M4 * M2 = Ekm1_T * Okm2_T
                    __syncthreads();
                    glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_M4, s_M2, s_M3);

                    // save M3 = Ekm1_T * Okm2_T to H
                    __syncthreads();
                    store_block_bd<T>(state_size, knot_points,
                                      s_M3,     // src
                                      d_H,      // dst
                                      0,        // left block column
                                      blockrow, // blockrow
                                      1         // positive
                    );
                }
            }
        }
    }
}

// this function assumes d_G, d_C, d_g, d_c are dense and ready
// this function fills in
//      diagonal blocks of d_S, and off-diagonal blocks of d_S partially
//      diagonal blocks of d_Pinv
//      all of d_gamma, d_T

// if org_trans, then everything in transformed coordinate (tilde)
// if !org_trans, then everything in original coordinate

// if chol_or_ldl, use chol
// if !chol_or_ldl, use LDL'

template<typename T>
__device__
void form_S_gamma_and_jacobi_Pinv_block_blockrow(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                                                 T *d_G, T *d_C, T *d_g, T *d_c,
                                                 T *d_S, T *d_Pinv, T *d_T, T *d_gamma,
                                                 T rho, T *s_temp, unsigned blockrow,
                                                 bool org_trans, bool chol_or_ldl) {

    T *d_Sdb = d_S;
    T *d_Sob = d_S + knot_points * state_size;
    T *d_Pinvdb = d_Pinv;
    T *d_Pinvob = d_Pinv + knot_points * state_size;

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
        __syncthreads();

        // Attention: loading of Q0 and computation of gamma_0 is finished
        // v2 = gamma_0, M2 = Q0_i

        if (org_trans) {
            // TRANS

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
            __syncthreads();
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
        } else {
            // ORG
            // note: after inversion, M1 is no longer Q0, so reload.
            glass::copy<T>(state_sq, d_G, s_M1);

            // save M1 = Q0 to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M1,         // src
                              d_Pinv,       // dst
                              1,            // col
                              blockrow,     // blockrow
                              1             // positive
            );

            // save M2 = Q0_i to S
            store_block_bd<T>(state_size, knot_points,
                              s_M2,         // src
                              d_S,          // dst
                              1,            // col
                              blockrow,     // blockrow
                              1             // positive
            );

            // save v2 = gamma_0 to gamma
            for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x) {
                d_gamma[ind] = s_v2[ind];
            }
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

        if (org_trans) {
            // TRANS

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
            __syncthreads();
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

            // M3 <- M2 * M1 = Tk * phi_k = Tk * Okm1'
            glass::trmm_left<T, false>(state_size, state_size, static_cast<T>(1), s_M2, s_M1, s_M3);
            __syncthreads();

            // save M3 = Tk * phi_k = Tk * Okm1' into left off-diagonal of S
            store_block_ob<T>(state_size, knot_points,
                              s_M3,         // src
                              d_Sob,        // dst
                              0,            // left block column
                              blockrow,     // blockrow
                              1             // positive
            );

            // load identity to M1
            loadIdentity<T>(state_size, s_M1);
            __syncthreads();
            // M2 <- M1 * M3' = I * (Tk * phi_k)' = phi_k' * Tk' = Okm1 * Tk'
            glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M1, s_M3, s_M2);

            // save phi_k' * T_k' = Okm1 * Tk' into right off-diagonal of S
            __syncthreads();
            store_block_ob<T>(state_size, knot_points,
                              s_M2,         // src
                              d_Sob,        // dst
                              1,            // right block column
                              blockrow - 1, // blockrow
                              1             // positive
            );
        } else {
            // ORG

            // save M1 = phi_k = Okm1' to S
            store_block_bd<T>(state_size, knot_points,
                              s_M1,         // src
                              d_S,          // dst
                              0,            // left block column
                              blockrow,     // blockrow
                              1             // positive
            );

            loadIdentity<T>(state_size, s_M2);
            __syncthreads();
            // M3 <- M2 * M1' = I * phi_k' = phi_k' = Okm1
            glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_M2, s_M1, s_M3);
            __syncthreads();

            // save M3 = phi_k' = Okm1 to S
            store_block_bd<T>(state_size, knot_points,
                              s_M3,         // src
                              d_S,          // dst
                              2,            // right block column
                              blockrow - 1, // blockrow
                              1             // positive
            );

            // save M4 = theta_k = Dk to S
            store_block_bd<T>(state_size, knot_points,
                              s_M4,         // src
                              d_S,          // dst
                              1,            // middle block column
                              blockrow,     // blockrow
                              1             // positive
            );

            loadIdentity<T>(state_size, s_M2);
            // M1 <- M4 = theta_k
            for (unsigned ind = threadIdx.x; ind < state_sq; ind += blockDim.x) {
                s_M1[ind] = s_M4[ind];
            }
            // invert M1 = theta_k, M2 <- inv(M1) = theta_k_inv = Dk_inv
            __syncthreads();
            invertMatrix<T>(state_size, s_M1, s_v1);

            // save M2 = theta_k_inv = Dk_inv to Pinv
            __syncthreads();
            store_block_bd<T>(state_size, knot_points,
                              s_M2,         // src
                              d_Pinv,       // dst
                              1,            // middle block column
                              blockrow,     // blockrow
                              1             // positive
            );

            // save v3 = gamma_k to gamma
            for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x) {
                unsigned offset = (blockrow) * state_size + ind;
                d_gamma[offset] = s_v3[ind];
            }
        }

    }
}


template<typename T>
__global__
void form_S_gamma_Pinv_block_kernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                                    T *d_G, T *d_C, T *d_g, T *d_c,
                                    T *d_S, T *d_Pinv, T *d_H, T *d_T, T *d_gamma,
                                    T rho, bool org_trans, bool chol_or_ldl, bool use_H) {

    extern __shared__ T s_temp[];

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        form_S_gamma_and_jacobi_Pinv_block_blockrow<T>(
                state_size, control_size, knot_points,
                d_G, d_C, d_g, d_c,
                d_S, d_Pinv, d_T, d_gamma,
                rho, s_temp, blockrow, org_trans, chol_or_ldl);
    }
    cgrps::this_grid().sync();

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        complete_SS_Pinv_block_blockrow<T>(
                state_size, knot_points,
                d_S, d_Pinv, d_H, d_T,
                s_temp, blockrow, org_trans, use_H);
    }
}

// only needed when org_trans == true
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
void form_schur_system_block(uint32_t state_size, uint32_t control_size, uint32_t knot_points,
                             T *d_G_dense, T *d_C_dense, T *d_g, T *d_c,
                             T *d_S, T *d_Pinv, T *d_H, T *d_T, T *d_gamma,
                             T rho, bool org_trans, bool chol_or_ldl, bool use_H) {
    // shared block memory summary (timeline & types)

    // previous setup
    // form_schur_system         8nx^2 + 7nx + nxnu + 3nu + 2nu^2 + 3               -> ORG
    // form_schur_system_block   7nx^2 + nx(nx+1)/2 + 8nx + nxnu + 3nu + 2nu^2 + 3  -> TRANS

    // current setup
    // form_S_gamma_and_jacobi_Pinv_block_blockrow      4nx^2 + 3nx + 1 (both ORG & TRANS)
    // complete_SS_Pinv_block_blockrow                  3nx^2 + nx(nx+1)/2 + 2nx  -> TRANS
    //                                                  4nx^2                     -> ORG

    // it is assumed here that nx > nu always holds
    const uint32_t s_temp_size = sizeof(T) * (4 * state_size * state_size +
                                              3 * state_size + 1);

    void *kernel = (void *) form_S_gamma_Pinv_block_kernel<T>;
    void *args[] = {(void *) &state_size, (void *) &control_size, (void *) &knot_points,
                    (void *) &d_G_dense, (void *) &d_C_dense, (void *) &d_g, (void *) &d_c,
                    (void *) &d_S, (void *) &d_Pinv, (void *) &d_H, (void *) &d_T, (void *) &d_gamma,
                    (void *) &rho, (void *) &org_trans, (void *) &chol_or_ldl, (void *) &use_H
    };

    gpuErrchk(cudaLaunchCooperativeKernel(kernel, knot_points, SCHUR_THREADS, args, s_temp_size));
}