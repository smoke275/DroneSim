"""Sensitivity sweeps for the paper (Pillar 4): fleet scaling, swap-latency
sensitivity, and map-topology comparison. Writes one CSV per sweep to
results/sweeps/. Plot with plot_sweeps.py.

    python sweep.py --seeds 10
"""

import argparse
import csv
import os

from dronesim.config import Strategy
from benchmark import run_one

MAX_ITER = 300000

FLEET_TRUCKS = [2, 4, 6, 8, 12]
FLEET_DPS = [1, 2, 4]
LATENCY_FRAMES = [150, 600, 1800, 3600, 7200]  # 1.25 s .. 60 s at 120 fps
DENSITY_MAZES = ['maze.csv',
                 'maps/maze_35x35_loop20_s1.csv',
                 'maps/maze_35x35_loop80_s1.csv',
                 'maps/maze_50x50_loop60_s1.csv']


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{path}: {len(rows)} rows')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--out', default='results/sweeps')
    args = parser.parse_args()
    seeds = range(1, args.seeds + 1)

    # 1. Fleet scaling: all strategies across truck fleet sizes; the proposed
    # method additionally across drone counts (saturation analysis).
    rows = []
    for trucks in FLEET_TRUCKS:
        for seed in seeds:
            for strategy in Strategy:
                rows.append(run_one(strategy, seed, MAX_ITER, 'maze.csv',
                                    num_trucks=trucks))
            for dps in (2, 4):  # dps=1 is the default run above
                rows.append(run_one(Strategy.PROACTIVE_FUEL, seed, MAX_ITER, 'maze.csv',
                                    num_trucks=trucks, drones_per_station=dps))
        print(f'fleet: trucks={trucks} done')
    write_csv(os.path.join(args.out, 'fleet_scaling.csv'), rows)

    # 2. Swap-latency sensitivity: how each swapping strategy degrades as the
    # physical swap gets slower (depot-return is unaffected by definition).
    rows = []
    for frames in LATENCY_FRAMES:
        for strategy in (Strategy.FIXED_STATION_EVRP, Strategy.REACTIVE_DRONE,
                         Strategy.PROACTIVE_FUEL):
            for seed in seeds:
                rows.append(run_one(strategy, seed, MAX_ITER, 'maze.csv',
                                    service_frames=frames))
        print(f'latency: {frames} frames done')
    for seed in seeds:  # depot reference (latency-independent)
        rows.append(run_one(Strategy.DEPOT_ONLY, seed, MAX_ITER, 'maze.csv'))
    write_csv(os.path.join(args.out, 'swap_latency.csv'), rows)

    # 3. Map topology: all strategies across maze densities/sizes.
    rows = []
    for maze in DENSITY_MAZES:
        for strategy in Strategy:
            for seed in seeds:
                rows.append(run_one(strategy, seed, MAX_ITER, maze))
        print(f'density: {maze} done')
    write_csv(os.path.join(args.out, 'map_density.csv'), rows)


if __name__ == '__main__':
    main()
