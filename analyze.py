#!/usr/bin/env python3

import argparse
import os
import numpy as np
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt

algs = ['PCG', 'QDLDL']

@dataclass
class Result:
    alg: str
    knot_points: int
    eps: float
    linsys_times: np.ndarray
    sqp_times: np.ndarray

def load_data(result_dir) -> list[Result]:
    results = []
    pattern = re.compile(r'(\d+)_([A-Z]+)(?:_(\d\.\d+))?_0_sqp_times\.result')

    for fname in os.listdir(result_dir):
        if not fname.endswith('_sqp_times.result'):
            continue

        match = pattern.match(fname)
        if not match:
            assert False, f"Unexpected filename format: {fname}"

        knot_points = int(match.group(1))
        alg = match.group(2)
        eps = float(match.group(3)) if match.group(3) is not None else -1.0

        base_name = fname.replace('_sqp_times.result', '')

        linsys_path = os.path.join(result_dir, f"{base_name}_linsys_times.result")
        sqp_path = os.path.join(result_dir, f"{base_name}_sqp_times.result")

        # if not os.path.exists(linsys_path):
        #     linsys_times = np.array([])
        # else:
        #     linsys_times = np.loadtxt(linsys_path)
        linsys_times = None
        sqp_times = np.loadtxt(sqp_path)

        results.append(Result(
            alg=alg,
            knot_points=knot_points,
            eps=eps,
            linsys_times=linsys_times,
            sqp_times=sqp_times
        ))
        print(f"Loaded {alg} with {knot_points} knot points and eps {eps} from {base_name}")

    return results


def analyze(result_dir: str, exit_tol: float):
    results_list = load_data(result_dir)
    algs = set(r.alg for r in results_list)
    knot_points = set(r.knot_points for r in results_list)
    epsilons = set(r.eps for r in results_list)

    base_eps = 1e-4
    assert base_eps in epsilons

    # ===============================================
    # Basic Comparison
    # ===============================================

    # plot the algs against each other over the knot points for PCG use eps = 1e-4
    # plot mean with std of mean. Add a red cross for worst case time. In a subplot below plot standard deviation

    # setup the figure

    xs = sorted(knot_points)
    # top plot is mean with std of mean and worst case
    # bottom plot is variance
    fig, axs = plt.subplots(2, figsize=(12, 8))
    fig.suptitle(f"Results for {result_dir} with exit tol {exit_tol}")

    mean_ax = axs[0]
    var_ax = axs[1]

    mean_ax.set_title("Mean and Worst Case")
    mean_ax.set_xlabel("Knot Points")
    mean_ax.set_ylabel("Time (us)")

    var_ax.set_title("Variance")
    var_ax.set_xlabel("Knot Points")

    for i, (alg, c) in enumerate(zip(algs, ['blue', 'orange'])):
        if alg == 'PCG':
            results = {res.knot_points: res for res in results_list if res.alg == alg and res.eps == base_eps}
        elif alg == 'QDLDL':
            results = {res.knot_points: res for res in results_list if res.alg == alg}
        else:
            assert False, f"Unexpected algorithm: {alg}"

        means = []
        std_means = []
        stds = []
        variances = []
        worst_case = []

        for k in xs:
            assert k in results, f"Missing result for {alg} with {k} knot points"
            res = results[k]
            sqp_times = res.sqp_times
            means.append(np.mean(sqp_times))
            std_means.append(np.std(sqp_times) / np.sqrt(len(sqp_times)))
            stds.append(np.std(sqp_times))
            variances.append(np.var(sqp_times))
            worst_case.append(np.max(sqp_times))

        # error bar with I shaped errors
        mean_ax.errorbar(xs, means, color=c, yerr=stds, label=alg, fmt='-o', capsize=5)
        mean_ax.scatter(xs, worst_case, color=c, marker='x', label=f"{alg} worst case")

        var_ax.plot(xs, variances, label=alg, color=c)

    mean_ax.legend()
    var_ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    plt.savefig(os.path.join(result_dir, "results.png"))


    # ===============================================
    # For all EPS and QDLDL
    # ===============================================

    fig, axs = plt.subplots(2, figsize=(12, 8))
    fig.suptitle(f"Results for {result_dir} with exit tol {exit_tol}")
    mean_ax = axs[0]
    var_ax = axs[1]
    mean_ax.set_title("Mean and Worst Case")
    mean_ax.set_xlabel("Knot Points")
    mean_ax.set_ylabel("Time (us)")
    var_ax.set_title("Variance")
    var_ax.set_xlabel("Knot Points")

    # if the alg is PCG then plot the results for every eps

    for alg, eps in [('QDLDL', None)] + [('PCG', eps) for eps in epsilons if eps > 0]:
        if alg == 'PCG':
            results = {res.knot_points: res for res in results_list if res.alg == alg and res.eps == eps}
        elif alg == 'QDLDL':
            results = {res.knot_points: res for res in results_list if res.alg == alg}
        else:
            assert False, f"Unexpected algorithm: {alg}"

        means = []
        std_means = []
        stds = []
        variances = []
        worst_case = []

        for k in xs:
            assert k in results, f"Missing result for {alg} with {k} knot points"
            res = results[k]
            sqp_times = res.sqp_times
            means.append(np.mean(sqp_times))
            std_means.append(np.std(sqp_times) / np.sqrt(len(sqp_times)))
            stds.append(np.std(sqp_times))
            variances.append(np.var(sqp_times))
            worst_case.append(np.max(sqp_times))

        if alg == "QDLDL":
            mean_ax.errorbar(xs, means, label=f"{alg} eps {eps}", yerr=stds, fmt='--o', capsize=7, markersize=10)
        else:
            mean_ax.errorbar(xs, means, label=f"{alg} eps {eps}", yerr=stds, fmt='-o', capsize=5)
        mean_ax.scatter(xs, worst_case, marker='x', label=f"{alg} eps {eps} worst case")

        var_ax.plot(xs, variances, label=f"{alg} eps {eps}")

    mean_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    var_ax.legend()
    # plt.tight_layout()

    plt.subplots_adjust(top=0.85)
    plt.show()
    plt.savefig(os.path.join(result_dir, "results_eps.png"))


def main():
    parser = argparse.ArgumentParser(description="Analyzes experiment results from MPCGPU")
    parser.add_argument("result_dir", help="Directory containing experiment results")
    parser.add_argument("--exit-tol", help="Tolerance for exit condition", type=str, default="1e-4")

    args = parser.parse_args()
    result_dir = args.result_dir
    exit_tol = float(args.exit_tol)
    analyze(result_dir, exit_tol)

if __name__ == "__main__":
    main()
