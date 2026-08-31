"""Render the paper's sensitivity figures from results/sweeps/*.csv.

    python3 plot_sweeps.py            # writes PNGs to results/sweeps/
"""

import csv
import os
from collections import defaultdict
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/sweeps'

# Categorical palette (CVD-validated); color follows the strategy everywhere.
STRAT_COLOR = {
    'proactive_fuel': '#2a78d6',
    'depot_only': '#eb6834',
    'fixed_station_evrp': '#1baf7a',
    'reactive_drone': '#eda100',
}
STRAT_LABEL = {
    'proactive_fuel': 'FUEL (proposed)',
    'depot_only': 'Depot-Return',
    'fixed_station_evrp': 'E-VRP-BSS',
    'reactive_drone': 'Reactive Drone',
}
# Ordered series (drones/station) use one hue, light -> dark.
DPS_COLOR = {1: '#a8cbee', 2: '#4a90dd', 4: '#1b4f94'}

TEXT = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def agg(rows, key_fields, value='makespan'):
    """-> {key_tuple: (mean, stderr-ish std)}"""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in key_fields)].append(float(r[value]))
    return {k: (mean(v), stdev(v) if len(v) > 1 else 0.0) for k, v in groups.items()}


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.grid(True, axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def fig_fleet():
    rows = load('fleet_scaling.csv')
    # Strategy comparison at the default 1 drone/station (drone count saturates
    # at 1/station for this task load; reported in the text).
    rows = [r for r in rows if r['drones_per_station'] == '1']
    trucks = sorted({int(r['num_trucks']) for r in rows})
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)

    by = agg(rows, ['strategy', 'num_trucks'])
    for s in ('depot_only', 'fixed_station_evrp', 'reactive_drone', 'proactive_fuel'):
        ys = [by[(s, str(t))][0] for t in trucks]
        # Reactive nearly coincides with FUEL: dash it so both stay readable
        ax.plot(trucks, ys, color=STRAT_COLOR[s], linewidth=2, marker='o',
                markersize=4.5, label=STRAT_LABEL[s],
                linestyle=(0, (3, 1.5)) if s == 'reactive_drone' else '-',
                zorder=3 if s == 'proactive_fuel' else 2)

    ax.set_xticks(trucks)
    style(ax, 'Number of trucks', 'Makespan (frames)')
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fleet_scaling.png')
    print('fleet_scaling.png')


def fig_latency():
    rows = load('swap_latency.csv')
    frames = sorted({int(r['service_frames']) for r in rows
                     if r['strategy'] != 'depot_only'})
    secs = [f / 120 for f in frames]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)

    depot_mean = mean(float(r['makespan']) for r in rows if r['strategy'] == 'depot_only')
    ax.axhline(depot_mean, color=MUTED, linewidth=1.4, linestyle='--')
    ax.annotate('Depot-Return (swap-time independent)', (secs[1], depot_mean),
                textcoords='offset points', xytext=(0, -13), fontsize=7.5, color=MUTED)

    by = agg(rows, ['strategy', 'service_frames'])
    for s in ('fixed_station_evrp', 'reactive_drone', 'proactive_fuel'):
        ys = [by[(s, str(f))][0] for f in frames]
        ax.plot(secs, ys, color=STRAT_COLOR[s], linewidth=2, marker='o',
                markersize=4.5, label=STRAT_LABEL[s])

    ax.set_xscale('log')
    ax.set_xticks(secs)
    ax.set_xticklabels([f'{s:g}' for s in secs])
    ax.minorticks_off()
    style(ax, 'Swap service time (s)', 'Makespan (frames)')
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{OUT}/swap_latency.png')
    print('swap_latency.png')


def fig_density():
    rows = load('map_density.csv')
    mazes = ['maze.csv', 'maze_35x35_loop20_s1.csv',
             'maze_35x35_loop80_s1.csv', 'maze_50x50_loop60_s1.csv']
    maze_label = {'maze.csv': 'Original\n(35×35)',
                  'maze_35x35_loop20_s1.csv': 'Sparse\n(loop 20)',
                  'maze_35x35_loop80_s1.csv': 'Dense\n(loop 80)',
                  'maze_50x50_loop60_s1.csv': 'Large\n(50×50)'}
    strategies = ['depot_only', 'fixed_station_evrp', 'reactive_drone', 'proactive_fuel']

    by = agg(rows, ['maze', 'strategy'])
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
    width = 0.19
    for j, s in enumerate(strategies):
        xs = [i + (j - 1.5) * (width + 0.012) for i in range(len(mazes))]
        means = [by[(m, s)][0] for m in mazes]
        errs = [by[(m, s)][1] for m in mazes]
        ax.bar(xs, means, width, color=STRAT_COLOR[s], label=STRAT_LABEL[s],
               yerr=errs, error_kw=dict(ecolor=MUTED, capsize=2, linewidth=0.9))

    ax.set_xticks(range(len(mazes)))
    ax.set_xticklabels([maze_label[m] for m in mazes], fontsize=8)
    style(ax, '', 'Makespan (frames)')
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT, ncol=2,
              loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(f'{OUT}/map_density.png')
    print('map_density.png')


if __name__ == '__main__':
    fig_fleet()
    fig_latency()
    fig_density()
