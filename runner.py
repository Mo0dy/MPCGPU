#!/usr/bin/env python3

import os
from pathlib import Path
import re
import numpy as np

def compile():
    os.system("make clean && make examples")

def run():
    os.system("./run_examples.sh")

settings_file = Path(__file__).parent / "include/common/settings.cuh"

def set_knot_points(n: int):
    with open(settings_file, 'r') as f:
        content = f.read()

    content = re.sub(r'#define\s+KNOT_POINTS\s+\d+', f'#define KNOT_POINTS {n}', content)

    with open(settings_file, 'w') as f:
        f.write(content)

def run_expr(ns: list[int]):
    for n in ns:
        set_knot_points(n)
        compile()
        run()

if __name__ == "__main__":
    # knot_points = np.linspace(2, 512, 10, dtype=int)
    knot_points = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    print("Running experiment for knot_points: " + str(knot_points))
    run_expr(knot_points)
