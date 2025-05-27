#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import numpy as np
import re
from dataclasses import dataclass
from typing import Callable
from itertools import *
import sys

from pathlib import Path

import matplotlib.pyplot as plt

algs = ['PCG', 'QDLDL']

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
colors = cycle(colors)

class NoDataException(Exception):
    pass

@dataclass
class Settings:
    result_dir: Path
    plot_dir: Path
    base_eps: float

@dataclass
class Result:
    alg: str
    knot_points: int
    eps: float | None
    sqp_times: np.ndarray
    linsys_times: np.ndarray | None

@dataclass(frozen=True)
class Dataset:
    # TODO: make immutable
    results: list[Result]

    @property
    def epsilons(self) -> list[float]:
        return sorted(set(r.eps for r in self.results if r.eps is not None))

    @property
    def knot_points(self) -> list[int]:
        return sorted(set(r.knot_points for r in self.results))

    def filter_by_alg(self, alg: str) -> Dataset:
        return self.filter_by(lambda r: r.alg == alg)

    def filter_by_eps(self, eps: float | None, relative_tol=1e-4) -> Dataset:
        if eps is None:
            return self.filter_by(lambda r: r.eps is None)
        return self.filter_by(lambda r: (abs(r.eps - eps) / eps) < relative_tol if r.eps is not None else False)

    def filter_by_knot_points(self, knot_points: int) -> Dataset:
        return self.filter_by(lambda r: r.knot_points == knot_points)

    def filter_by(self, f: Callable[[Result], bool]) -> Dataset | OneDataset:
        dataset = Dataset(
            results=[r for r in self.results if f(r)]
        )
        if len(dataset.results) == 0:
            return Dataset([])
        if dataset.is_one_run():
            return OneDataset(sorted(dataset.results, key=lambda r: r.knot_points))
        return dataset

    def is_one_run(self) -> bool:
        """Checks if the dataset contains the results of only one run."""
        if len(self.results) != len(set(r.knot_points for r in self.results)):
            return False
        r0 = self.results[0]
        return all(r.alg == r0.alg and r.eps == r0.eps for r in self.results)

    def get_result(self, alg: str, knot_points: int, eps: float | None=None) -> Result:
        """Get the result for a specific algorithm and knot points."""
        res = None
        for r in self.results:
            if r.alg == alg and r.knot_points == knot_points and (eps is None or r.eps == eps):
                raise NoDataException(f"Multiple results found for {alg} with {knot_points} knot points and eps {eps}")
                res = r
        raise NoDataException(f"Result not found for {alg} with {knot_points} knot points and eps {eps}")
        return res


class OneDataset(Dataset):
    # TODO generalize to may vary at most in one dimension
    def __init__(self, results: list[Result]):
        super().__init__(results)
        assert self.is_one_run(), "OneDataset must contain results of only one run"
        self.by_knot_points = {r.knot_points: r for r in self.results}

    def __getitem__(self, item):
        return self.by_knot_points[item]


def make_title(
        title: str,
        alg: str | None = None,
        knot_points: int | None = None,
        eps: float | None = None,
        result_type: str = 'sqp_times',
) -> str:
    return "_".join(
        filter(None, [
            title,
            alg if alg else None,
            f"n={knot_points}" if knot_points is not None else None,
            f"eps={eps}" if eps is not None else None,
            result_type
        ])
        )


def make_title_from_data(
        title: str,
        data: Dataset | list[Result]
):
    if isinstance(data, list):
        data = Dataset(data)
    if len(epsilons := data.epsilons) == 1:
        eps = next(iter(epsilons))
    else:
        eps = None
    if len(knot_points := data.knot_points) == 1:
        knot_points = next(iter(knot_points))
    else:
        knot_points = None
    if len(algorithms := set(r.alg for r in data.results)) == 1:
        alg = next(iter(algorithms))
    else:
        alg = None
    return make_title(
        title=title,
        alg=alg,
        knot_points=knot_points,
        eps=eps
    )



def load_data(result_dir: Path) -> Dataset:
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
        eps = float(match.group(3)) if match.group(3) is not None else None

        base_name = fname.replace('_sqp_times.result', '')

        linsys_path = os.path.join(result_dir, f"{base_name}_linsys_times.result")
        sqp_path = os.path.join(result_dir, f"{base_name}_sqp_times.result")

        # TODO: load linsys_times if they exist
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
            sqp_times=sqp_times,
            linsys_times=linsys_times,
        ))
        print(f"Loaded {alg} with {knot_points} knot points and eps {eps} from {base_name}")

    return Dataset(results)

def plot_mean_and_var_over_knot_points(data: Dataset, epsilons: list[float] | float, settings: Settings, show=False, close=True):
    fig, axs = plt.subplots(2, figsize=(12, 8))
    if isinstance(epsilons, list) and len(epsilons) == 1:
        title = make_title("mean_and_variance", eps=epsilons[0])
    if isinstance(epsilons, list):
        title = make_title("mean_and_variance")
    else:
        title = make_title("mean_and_variance", eps=epsilons)
        epsilons = [epsilons]
    fig.suptitle(f"Results for {title}.")
    mean_ax = axs[0]
    var_ax = axs[1]

    mean_ax.set_title("Mean and Worst Case")
    mean_ax.set_xlabel("Knot Points")
    mean_ax.set_ylabel("Time (us)")
    var_ax.set_title("Variance")
    var_ax.set_xlabel("Knot Points")

    for (alg, eps), c in zip(
            [('QDLDL', None)] + [('PCG', eps) for eps in epsilons if eps > 0],
            colors
    ):
        results = data.filter_by_alg(alg).filter_by_eps(eps)
        assert isinstance(results, OneDataset), f"Expected OneDataset but got {type(results)}"

        xs = results.knot_points

        means = []
        std_means = []
        stds = []
        variances = []
        worst_case = []

        for k in xs:
            # assert k in results, f"Missing result for {alg} with {k} knot points"
            res = results[k]
            sqp_times = res.sqp_times
            means.append(np.mean(sqp_times))
            std_means.append(np.std(sqp_times) / np.sqrt(len(sqp_times)))
            stds.append(np.std(sqp_times))
            variances.append(np.var(sqp_times))
            worst_case.append(np.max(sqp_times))

        # error bar with I shaped errors
        if alg == "QDLDL":
            mean_ax.errorbar(xs, means, label=f"{alg}", yerr=stds, fmt='--o', capsize=7, markersize=10)
        else:
            mean_ax.errorbar(xs, means, color=c, yerr=stds, label=f"{alg} eps {eps}", fmt='-o', capsize=5)
        mean_ax.scatter(xs, worst_case, color=c, marker='x', label=f"{alg} worst case")

        if alg == "QDLDL":
            var_ax.plot(xs, variances, label=f"{alg}", linestyle='--', color=c)
        else:
            var_ax.plot(xs, variances, label=f"{alg} eps {eps}", color=c)

    mean_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    var_ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(settings.plot_dir / f"{title}.png")
    if show:
        plt.show()
    if close:
        plt.close(fig)


def plot_histograms_from_data(
        results: list[Result],
        title: str,
        settings: Settings,
        get_data: Callable[[Result], np.ndarray] = lambda r: r.sqp_times,
        scale_to_q1_q3=False,
        show=False,
        close=True
):
    title = make_title_from_data(title, results)

    # plots a series of histograms into a single figure. One for every result
    # align all x axes

    if scale_to_q1_q3:
        min_time = min(np.percentile(r.sqp_times, 25) for r in results)
        max_time = max(np.percentile(r.sqp_times, 75) for r in results)
        diff = max_time - min_time
        min_time = max(0, min_time - diff * 0.4)
        max_time = max_time + diff * 0.4
    else:
        min_time = min(get_data(r).min() for r in results)
        max_time = max(get_data(r).max() for r in results)
        diff = max_time - min_time
        min_time = max(0, min_time - diff * 0.1)
        max_time = max_time + diff * 0.1



    def plot_histogram(ax, result: Result):
        ax.hist(get_data(result), bins='auto')
        ax.set_title(f"{result.alg} with {result.knot_points} knot points and eps {result.eps}")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Frequency")
        ax.grid()
        # draw line at mean, q1, q3
        mean = np.mean(get_data(result))
        q1 = np.percentile(get_data(result), 25)
        q3 = np.percentile(get_data(result), 75)
        ax.axvline(mean, color='gray', linestyle='--', label='Mean')
        ax.axvline(q1, color='blue', linestyle='--', label='Q1, Q3')
        ax.axvline(q3, color='blue', linestyle='--')

        # mark min and max value of this plot
        ax.axvline(get_data(result).min(), color='green', linestyle='--', label='Min')
        ax.axvline(get_data(result).max(), color='red', linestyle='--', label='Max')

        ax.set_xlim(min_time, max_time)

        ax.legend()

    fig, axs = plt.subplots(len(results), figsize=(12, 8))
    fig.suptitle(f"Results for {title}.")
    for ax, result in zip(axs, results):
        plot_histogram(ax, result)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(settings.plot_dir / f"{title}.png")
    if show:
        plt.show()
    if close:
        plt.close(fig)


def analyze(settings: Settings):
    data = load_data(settings.result_dir)
    plot_mean_and_var_over_knot_points(data, settings.base_eps, settings, close=False)
    plot_mean_and_var_over_knot_points(data, data.epsilons, settings, close=False)
    for n in [16, 128, 512]:
        qdldl_result = data.filter_by_alg("QDLDL").filter_by_knot_points(n)
        assert qdldl_result.is_one_run(), "QDLDL result should be a single run"
        for eps in data.epsilons:
            pcg_result = data.filter_by_alg("PCG").filter_by_knot_points(n).filter_by_eps(eps)
            if len(pcg_result.results) == 0:
                print(f"No data for eps {eps} and {n} knot points", file=sys.stderr)
                continue
            assert pcg_result.is_one_run(), "PCG result should be a single run"
            plot_histograms_from_data(
                [qdldl_result.results[0], pcg_result.results[0]],
                "histogram",
                settings,
                close=False
            )
            plot_histograms_from_data(
                [qdldl_result.results[0], pcg_result.results[0]],
                "histogram_scaled",
                settings,
                scale_to_q1_q3=True,
                close=False
            )


def main():
    parser = argparse.ArgumentParser(description="Analyzes experiment results from MPCGPU")
    parser.add_argument("result_dir", help="Directory containing experiment results")
    parser.add_argument("--exit-tol", help="Tolerance for exit condition", type=str, default="1e-4")
    parser.add_argument("-c", "--clean", action="store_true", default=False,
                        help="Clean the plot directory before running the analysis")

    args = parser.parse_args()
    result_dir = args.result_dir
    plot_dir = Path(str(result_dir) + "_plots")
    if args.clean and plot_dir.exists():
        print(f"Cleaning plot directory: {plot_dir}")
        for file in plot_dir.glob("*"):
            file.unlink()
        plot_dir.rmdir()
    plot_dir.mkdir(parents=True, exist_ok=True)
    exit_tol = float(args.exit_tol)
    analyze(
        Settings(
            result_dir=Path(result_dir),
            plot_dir=plot_dir,
            base_eps=exit_tol
        )
    )

if __name__ == "__main__":
    main()
