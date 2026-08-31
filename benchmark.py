"""Headless benchmark: run every strategy on identical scenarios (paired by
seed) and report per-strategy metrics for the E-VRP-BSS comparison.

Usage (inside the container, no display needed):
    python benchmark.py --seeds 10
    python benchmark.py --seeds 20 --max-iter 100000 --out results
"""

import argparse
import csv
import os
import statistics
import time

from dronesim.config import Strategy
from dronesim.simulation import Simulation

SUMMARY_FIELDS = [
    'makespan', 'total_truck_distance', 'active_distance', 'detour_distance',
    'detour_percentage', 'uptime_ratio', 'total_swaps',
    'drone_flight_distance', 'drone_energy', 'drone_efficiency_percentage',
]


def run_one(strategy, seed, max_iter, maze, num_trucks=None, drones_per_station=None,
            service_frames=None):
    from dronesim import config
    kwargs = dict(window=None, strategy=strategy, seed=seed, maze_path=maze)
    if num_trucks is not None:
        kwargs['num_trucks'] = num_trucks
    if drones_per_station is not None:
        kwargs['drones_per_station'] = drones_per_station
    if service_frames is not None:
        kwargs['service_frames'] = service_frames
    sim = Simulation(**kwargs)
    start = time.perf_counter()
    sim.run(max_iterations=max_iter)
    metrics = sim.get_metrics()
    metrics['seed'] = seed
    metrics['maze'] = os.path.basename(maze)
    metrics['num_trucks'] = sim.num_trucks
    metrics['drones_per_station'] = sim.drones_per_station
    metrics['service_frames'] = sim.drone_service_frames
    metrics['wall_time_s'] = round(time.perf_counter() - start, 2)
    return metrics


def summarize(rows):
    """mean ± std for each summary field over a strategy's runs."""
    out = {}
    for f in SUMMARY_FIELDS:
        values = [float(r[f]) for r in rows]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        out[f] = (mean, std)
    out['completed'] = sum(1 for r in rows if r['completed'])
    out['fuel_violations'] = sum(int(r['fuel_violations']) for r in rows)
    return out


def main():
    parser = argparse.ArgumentParser(description='E-VRP-BSS strategy benchmark')
    parser.add_argument('--seeds', type=int, default=10, help='number of scenarios per strategy')
    parser.add_argument('--seed-start', type=int, default=1, help='first seed value')
    parser.add_argument('--max-iter', type=int, default=100000, help='frame cap per run')
    parser.add_argument('--out', default='results', help='output directory')
    parser.add_argument('--strategies', nargs='*', default=[s.value for s in Strategy],
                        help='subset of strategies to run')
    parser.add_argument('--mazes', nargs='*', default=['maze.csv'],
                        help='maze CSVs to benchmark on (default: maze.csv)')
    parser.add_argument('--trucks', type=int, default=None, help='override fleet size')
    parser.add_argument('--drones-per-station', type=int, default=None,
                        help='override drones per base station')
    parser.add_argument('--service-frames', type=int, default=None,
                        help='override swap service time (frames; 120 frames = 1 s)')
    args = parser.parse_args()

    strategies = [Strategy(s) for s in args.strategies]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    for maze in args.mazes:
        for strategy in strategies:
            for seed in seeds:
                row = run_one(strategy, seed, args.max_iter, maze,
                              num_trucks=args.trucks,
                              drones_per_station=args.drones_per_station,
                              service_frames=args.service_frames)
                all_rows.append(row)
                status = 'ok' if row['completed'] else 'INCOMPLETE'
                viol = f"  FUEL<0 x{row['fuel_violations']}" if row['fuel_violations'] else ''
                print(f"{row['maze']:<24} {strategy.value:<20} seed {seed:>3}  "
                      f"makespan {row['makespan']:>7}  uptime {row['uptime_ratio']:.3f}  "
                      f"detour {row['detour_percentage']:5.1f}%  [{status}]{viol}  ({row['wall_time_s']}s)")

    csv_path = os.path.join(args.out, 'benchmark_results.csv')
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print('\n' + '=' * 100)
    print(f"{'metric':<28}" + ''.join(f'{s.value:>18}' for s in strategies))
    print('-' * 100)
    by_strategy = {s: summarize([r for r in all_rows if r['strategy'] == s.value])
                   for s in strategies}
    for f in SUMMARY_FIELDS:
        cells = ''.join(f'{by_strategy[s][f][0]:>11.1f}±{by_strategy[s][f][1]:<6.1f}'
                        for s in strategies)
        print(f'{f:<28}{cells}')
    for f in ('completed', 'fuel_violations'):
        cells = ''.join(f'{by_strategy[s][f]:>18}' for s in strategies)
        print(f'{f + f"/{len(seeds)}" if f == "completed" else f:<28}{cells}')
    print('=' * 100)
    print(f'\nPer-run rows written to {csv_path}')


if __name__ == '__main__':
    main()
