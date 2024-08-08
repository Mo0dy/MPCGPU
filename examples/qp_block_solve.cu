#include <fstream>      // std::ofstream
#include <iostream>
#include <stdio.h>
#include "gpuassert.cuh"
#include "read_array.h"
#include <ctime>
#include "pcg/qp_block.cuh"
#include "pcg/qp.cuh"
#include <tuple>

#define tic      double tic_t = clock();
#define toc      std::cout << (clock() - tic_t)/CLOCKS_PER_SEC \
                           << " seconds" << std::endl;

int main() {

    const uint32_t state_size = STATE_SIZE;
    const uint32_t knot_points = KNOT_POINTS;
    const uint32_t control_size = CONTROL_SIZE;

    const uint32_t states_sq = state_size * state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;

    const uint32_t KKT_G_DENSE_SIZE = (states_sq + controls_sq) * knot_points - controls_sq;
    const uint32_t KKT_C_DENSE_SIZE = (states_sq + states_p_controls) * (knot_points - 1);
    const uint32_t KKT_g_SIZE = (state_size + control_size) * knot_points - control_size;
    const uint32_t KKT_c_SIZE = state_size * knot_points;
    const uint32_t DZ_SIZE = states_s_controls * knot_points - control_size;

    double h_G_dense[KKT_G_DENSE_SIZE];
    double h_C_dense[KKT_C_DENSE_SIZE];
    double h_g[KKT_g_SIZE];
    double h_c[KKT_c_SIZE];
    readArrayFromFile(KKT_G_DENSE_SIZE, "data/G_dense.txt", h_G_dense);
    readArrayFromFile(KKT_C_DENSE_SIZE, "data/C_dense.txt", h_C_dense);
    readArrayFromFile(KKT_g_SIZE, "data/g.txt", h_g);
    readArrayFromFile(KKT_c_SIZE, "data/c.txt", h_c);

    double h_dz_trans[DZ_SIZE];
    double h_dz_org[DZ_SIZE];

    struct pcg_config<double> config;
    config.pcg_org_trans = false;
    config.pcg_poly_order = 0;
    std::tuple<uint32_t, double, double> qp_trans_stats, qp_org_stats;
    qp_org_stats = qpSolvePcg<double>(state_size, control_size, knot_points,
                                      h_G_dense,
                                      h_C_dense,
                                      h_g,
                                      h_c,
                                      h_dz_org,
                                      config);

    config.pcg_org_trans = true;
    qp_trans_stats = qpBlockSolvePcg<double, CHOL_OR_LDL>(state_size, control_size, knot_points,
                                                          h_G_dense,
                                                          h_C_dense,
                                                          h_g,
                                                          h_c,
                                                          h_dz_trans,
                                                          config);
    uint32_t pcg_org_iters = std::get<0>(qp_org_stats);
    uint32_t pcg_trans_iters = std::get<0>(qp_trans_stats);

    std::cout << "Original PCG iteration number: " << pcg_org_iters << std::endl;
    std::cout << "Transformed PCG iteration number: " << pcg_trans_iters << std::endl;

    double norm_org = 0;
    double norm_trans = 0;
    double diff = 0;
    for (uint32_t i = 0; i < DZ_SIZE; i++) {
        norm_org += h_dz_org[i] * h_dz_org[i];
        norm_trans += h_dz_trans[i] * h_dz_trans[i];
        diff += (h_dz_org[i] - h_dz_trans[i]) * (h_dz_org[i] - h_dz_trans[i]);
    }
    std::cout << "Original dz norm: " << sqrt(norm_org) << std::endl;
    std::cout << "Transformed dz norm: " << sqrt(norm_trans) << std::endl;
    std::cout << "dz norm difference: " << sqrt(diff) << std::endl;


//    int iteration = 1000;
//    double linsys_time_total = 0;
//    double qp_solve_time_total = 0;
//    for (int i = 0; i < iteration; i++) {
//        qp_trans_stats = qpBlockSolvePcg<double, CHOL_OR_LDL>(state_size, control_size, knot_points,
//                                                             h_G_dense,
//                                                             h_C_dense,
//                                                             h_g,
//                                                             h_c,
//                                                             h_dz_trans,
//                                                             config);
//        double linsys_time = std::get<1>(qp_trans_stats);
//        double qp_solve_time = std::get<2>(qp_trans_stats);
//        linsys_time_total += linsys_time;
//        qp_solve_time_total += qp_solve_time;
//    }
//    std::cout << "PCG time avg in " << iteration << " iterations: " << linsys_time_total / iteration
//              << " us (1e-6) microseconds. " << std::endl;
//    std::cout << "QP time avg in " << iteration << " iterations: " << qp_solve_time_total / iteration
//              << " us (1e-6) microseconds. " << std::endl;
    return 0;
}

