#pragma once
#include <cstdint>
#include "gpuassert.cuh"
#include "glass.cuh"
#include "utils/matrix.cuh"
#include "settings.cuh"

/*
 * This version of the Schur–complement construction **removes all
 * pre‑conditioner (Jacobi / symmetric‑stair) related calculations**.
 *
 * In practice that means:
 *   • **No inverse blocks (theta⁻¹, Q⁻¹, R⁻¹, …) are ever written out to
 *     d_Pinv.**  All of those computations – including the secondary
 *     kernel pass that stitched the block‑tridiagonal Φ⁻¹ – have been
 *     deleted.
 *   • The few inverses that are still required *internally* (e.g. Q⁻¹,
 *     R⁻¹ for γ and for saving G⁻¹ in d_G) are kept, but they stay in
 *     shared memory and are not copied to any preconditioner buffer.
 *
 * The public API therefore shrinks – every function loses the d_Pinv
 * argument, and the launch kernel has been renamed accordingly.
 */

/*******************************************************************************
 *                      Helpers (unchanged, just forward)                      *
 *******************************************************************************/

template <typename T>
__device__
void store_block_bd(
    uint32_t state_size,
    uint32_t knot_points,
    const T *src,
    T *dst,
    unsigned block_col,  // 0: left, 1: diag, 2: right
    unsigned block_row,
    int8_t multiplier = 1);

template <typename T>
__device__
void load_block_bd(
    uint32_t state_size,
    uint32_t knot_points,
    const T *src,
    T *dst,
    unsigned block_col,
    unsigned block_row,
    bool transpose = false);

/*******************************************************************************
 *          Per block‑row Schur complement construction **without Φ⁻¹**        *
 *******************************************************************************/

template <typename T>
__device__
void form_S_gamma_blockrow(
    uint32_t state_size,
    uint32_t control_size,
    uint32_t knot_points,
    T *d_G,
    T *d_C,
    T *d_g,
    T *d_c,
    T *d_S,
    T *d_gamma,
    T rho,
    T *s_temp,
    unsigned blockrow)
{
    // -------------------------------------------------------------------------
    // Shared‑memory layout (identical to the original file but without any of
    // the *_Inv or PhiInv_* buffers).
    // -------------------------------------------------------------------------

    // | ϕ_k | θ_k | γ_k |   scratch…
    //   s²     s²    s    (everything else is allocated on demand to maximize
    //                      reuse of the same large buffer)

    T *s_phi_k      = s_temp;                              // state²
    T *s_theta_k    = s_phi_k   + state_size*state_size;    // state²
    T *s_gamma_k    = s_theta_k + state_size*state_size;    // state
    T *s_end_main   = s_gamma_k + state_size;               // <- scratch ptr

    // Convenience lambdas -----------------------------------------------------
    auto add_identity = [=] __device__ (T *A, uint32_t dim, T coef)
    {
        for (unsigned i = threadIdx.x; i < dim; i += blockDim.x)
            A[i*dim + i] += coef;
    };

    // -------------------------------------------------------------------------
    // LEADING BLOCK ROW (k = 0)  — handles Q₀/Qᴺ set‑up, γ₀ etc.
    // -------------------------------------------------------------------------
    if (blockrow == 0) {
        // --- Additional shared memory blocks --------------------------------
        T *s_QN      = s_end_main;                                   // s²
        T *s_QN_i    = s_QN      + state_size*state_size;            // s²
        T *s_qN      = s_QN_i    + state_size*state_size;            // s
        T *s_Q0      = s_qN      + state_size;                       // s²
        T *s_Q0_i    = s_Q0      + state_size*state_size;            // s²
        T *s_q0      = s_Q0_i    + state_size*state_size;            // s
        T *s_extra   = s_q0      + state_size;                       // scratch

        __syncthreads();

        // Copy dense cost blocks from global.
        glass::copy<T>(state_size*state_size, d_G,                                                     s_Q0);
        glass::copy<T>(state_size*state_size, d_G+(knot_points-1)*(state_size*state_size+control_size*control_size), s_QN);
        glass::copy<T>(state_size,            d_g,                                                     s_q0);
        glass::copy<T>(state_size,            d_g+(knot_points-1)*(state_size+control_size),           s_qN);

        __syncthreads();

        add_identity(s_Q0, state_size, rho);
        add_identity(s_QN, state_size, rho);

        // Invert Q₀, Qᴺ — still needed later for γ and for writing G^{-1}.
        loadIdentity<T>(state_size, state_size, s_Q0_i, s_QN_i);
        __syncthreads();
        invertMatrix<T>(state_size, state_size, state_size, s_Q0, s_QN, s_extra);
        __syncthreads();

        // γ₀ = -(Q₀^{-1} q₀)
        mat_vec_prod<T>(state_size, state_size, s_Q0_i, s_q0, s_gamma_k);
        __syncthreads();

        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x)
            d_gamma[i] = -s_gamma_k[i];

        // Store -Q₀ into Schur S (block diag, col = 1, row = 0)
        store_block_bd<T>(state_size, knot_points, s_Q0, d_S, 1, blockrow, -1);
    }
    // -------------------------------------------------------------------------
    // GENERAL BLOCK ROW (k > 0)
    // -------------------------------------------------------------------------
    else {
        const unsigned C_set_size = state_size*state_size + state_size*control_size;
        const unsigned G_set_size = state_size*state_size + control_size*control_size;

        // --- Additional shared memory allocations ----------------------------
        T *s_Ak        = s_end_main;                                             // s²
        T *s_Bk        = s_Ak       + state_size*state_size;                     // s*c
        T *s_Qk        = s_Bk       + state_size*control_size;                   // s²
        T *s_Qk_i      = s_Qk       + state_size*state_size;                     // s²
        T *s_Qkp1      = s_Qk_i     + state_size*state_size;                     // s²
        T *s_Qkp1_i    = s_Qkp1     + state_size*state_size;                     // s²
        T *s_Rk        = s_Qkp1_i   + state_size*state_size;                     // c²
        T *s_Rk_i      = s_Rk       + control_size*control_size;                 // c²
        T *s_qk        = s_Rk_i     + control_size*control_size;                 // s
        T *s_qkp1      = s_qk       + state_size;                                // s
        T *s_rk        = s_qkp1     + state_size;                                // c
        T *s_extra     = s_rk       + control_size;                              // scratch (≥ s²)

        __syncthreads();

        // Copy all required dense blocks.
        glass::copy<T>(state_size*state_size,       d_C + (blockrow-1)*C_set_size,                                   s_Ak);
        glass::copy<T>(state_size*control_size,     d_C + (blockrow-1)*C_set_size + state_size*state_size,            s_Bk);
        glass::copy<T>(state_size*state_size,       d_G + (blockrow-1)*G_set_size,                                   s_Qk);
        glass::copy<T>(state_size*state_size,       d_G +  blockrow   *G_set_size,                                   s_Qkp1);
        glass::copy<T>(control_size*control_size,   d_G + (blockrow-1)*G_set_size + state_size*state_size,            s_Rk);
        glass::copy<T>(state_size,                  d_g + (blockrow-1)*(state_size+control_size),                     s_qk);
        glass::copy<T>(state_size,                  d_g +  blockrow   *(state_size+control_size),                     s_qkp1);
        glass::copy<T>(control_size,                d_g + (blockrow-1)*(state_size+control_size) + state_size,        s_rk);

        __syncthreads();

        add_identity(s_Qk,    state_size,  rho);
        add_identity(s_Qkp1,  state_size,  rho);
        add_identity(s_Rk,    control_size, rho);

        // Invert (Q_k, Q_{k+1}, R_k) – kept for γ + for writing back to d_G.
        loadIdentity<T>(state_size, state_size, control_size, s_Qk_i, s_Qkp1_i, s_Rk_i);
        __syncthreads();
        invertMatrix<T>(state_size, state_size, control_size, state_size, s_Qk, s_Qkp1, s_Rk, s_extra);
        __syncthreads();

        // Persist Q_k^{-1} and R_k^{-1} back into d_G (used by later stages).
        glass::copy<T>(state_size*state_size,       s_Qk_i,     d_G + (blockrow-1)*G_set_size);
        glass::copy<T>(control_size*control_size,   s_Rk_i,     d_G + (blockrow-1)*G_set_size + state_size*state_size);
        if (blockrow == knot_points-1)
            glass::copy<T>(state_size*state_size, s_Qkp1_i, d_G + blockrow*G_set_size);

        // ------------------------------------------------------------------
        // Build ϕ_k  (−A_k Q_k^{-1})
        // ------------------------------------------------------------------
        glass::gemm<T>(state_size, state_size, state_size, static_cast<T>(1.0), s_Ak, s_Qk_i, s_phi_k);

        // ------------------------------------------------------------------
        // Build θ_k (A Q−¹ Aᵀ + B R−¹ Bᵀ + Q_{k+1}^{-1})
        // ------------------------------------------------------------------
        glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_phi_k, s_Ak, s_theta_k);
        __syncthreads();
        for (unsigned i = threadIdx.x; i < state_size*state_size; i += blockDim.x)
            s_theta_k[i] += s_Qkp1_i[i];                       // add Q_{k+1}^{-1}
        __syncthreads();
        glass::gemm<T, true>(state_size, control_size, state_size, static_cast<T>(1.0), s_Bk, s_Rk_i, s_Qkp1_i); // reuse s_Qkp1_i as temp
        __syncthreads();
        for (unsigned i = threadIdx.x; i < state_size*state_size; i += blockDim.x)
            s_theta_k[i] += s_Qkp1_i[i];                       // add B R^{-1} Bᵀ

        // ------------------------------------------------------------------
        // γ_k  --------------------------------------------------------------
        //           =  Q_{k+1}^{-1} q_{k+1}  -  A Q_k^{-1} q_k  -  B R_k^{-1} r_k
        //                - integrator_error
        // ------------------------------------------------------------------
        mat_vec_prod<T>(state_size, state_size, s_Qkp1_i, s_qkp1, s_gamma_k);                    // Q_{k+1}^{-1} q_{k+1}
        mat_vec_prod<T>(state_size, state_size, s_phi_k,   s_qk,    s_extra);                    // tmp0 = -A Q_k^{-1} q_k
        mat_vec_prod<T>(state_size, control_size, s_Bk,    s_Rk_i,  s_Qkp1);                     // s_Qkp1  re‑used for BR−¹
        mat_vec_prod<T>(state_size, control_size, s_Qkp1,  s_rk,    s_extra + state_size);       // tmp1 = -B R_k^{-1} r_k
        __syncthreads();

        for (unsigned i = threadIdx.x; i < state_size; i += blockDim.x) {
            T val =  s_gamma_k[i]           //   Q_{k+1}^{-1} q_{k+1}
                    + s_extra[i]            // − A Q_k^{-1} q_k
                    + s_extra[state_size+i] // − B R_k^{-1} r_k
                    - d_c[blockrow*state_size + i]; // integrator error
            d_gamma[blockrow*state_size + i] = -val; // store with minus sign
        }

        // ------------------------------------------------------------------
        // Write Schur blocks (ϕ_k, −θ_k, ϕ_kᵀ) to global S ------------------
        // ------------------------------------------------------------------
        store_block_bd<T>(state_size, knot_points, s_phi_k,    d_S, 0, blockrow, -1); // left off‑diag
        store_block_bd<T>(state_size, knot_points, s_theta_k,  d_S, 1, blockrow, -1); // main diag (negated)

        // ϕ_kᵀ -> right off‑diag (belongs to row k‑1, col=2) -----------------
        T *s_phi_k_T = s_Qkp1; // recycler (state²)
        glass::gemm<T, true>(state_size, state_size, state_size, static_cast<T>(1.0), s_Ak /*identity*/, s_phi_k, s_phi_k_T);
        __syncthreads();
        store_block_bd<T>(state_size, knot_points, s_phi_k_T, d_S, 2, blockrow-1, -1);
    }

    __syncthreads();
}

/*******************************************************************************
 *                              Kernel wrapper                                 *
 *******************************************************************************/

template <typename T>
__global__
void form_S_gamma_kernel(
    uint32_t state_size,
    uint32_t control_size,
    uint32_t knot_points,
    T *d_G,
    T *d_C,
    T *d_g,
    T *d_c,
    T *d_S,
    T *d_gamma,
    T rho)
{
    extern __shared__ T s_temp[];

    for (unsigned blockrow = blockIdx.x; blockrow < knot_points; blockrow += gridDim.x) {
        form_S_gamma_blockrow<T>(state_size, control_size, knot_points,
                                  d_G, d_C, d_g, d_c, d_S, d_gamma, rho,
                                  s_temp, blockrow);
    }
}

/*******************************************************************************
 *                            Host‑side launcher                               *
 *******************************************************************************/

template <typename T>
void form_schur_system(
    uint32_t state_size,
    uint32_t control_size,
    uint32_t knot_points,
    T *d_G_dense,
    T *d_C_dense,
    T *d_g,
    T *d_c,
    T *d_S,
    T *d_gamma,
    T rho)
{
    // Shared‑memory size — same compute path as original; could be tightened,
    // but keeping the original upper bound is safe and avoids recalculation.
    const uint32_t s_temp_size = sizeof(T)*(8 * state_size*state_size +
                                            7 * state_size +
                                            state_size * control_size +
                                            3 * control_size +
                                            2 * control_size * control_size +
                                            3);

    void *kernel = (void*)form_S_gamma_kernel<T>;
    void *args[] = {
        (void*)&state_size,
        (void*)&control_size,
        (void*)&knot_points,
        (void*)&d_G_dense,
        (void*)&d_C_dense,
        (void*)&d_g,
        (void*)&d_c,
        (void*)&d_S,
        (void*)&d_gamma,
        (void*)&rho
    };

    gpuErrchk(cudaLaunchCooperativeKernel(kernel, knot_points, SCHUR_THREADS, args, s_temp_size));
}
