"""Maze/map generator producing CSVs in the same format as maze.csv
(pyamaze saveMaze format: '  cell  ,E,W,N,S', 1 = open passage).

Algorithm: recursive-backtracker perfect maze, then a fraction of the
remaining internal walls is opened to create loops (road-network feel).
loop_percent=0 gives a perfect maze (single path between any two cells);
loop_percent=80 opens 80% of the leftover walls, giving a dense grid with
many alternative routes, similar to the original map.

Usage:
    python -m dronesim.mazegen --rows 35 --cols 35 --loop 80 --seed 1 -o maps/maze_a.csv
"""

import argparse
import csv
import os
import random

# Direction -> (dr, dc, opposite)
DIRS = {'E': (0, 1, 'W'), 'W': (0, -1, 'E'), 'N': (-1, 0, 'S'), 'S': (1, 0, 'N')}


def generate(rows, cols, loop_percent=80, seed=None):
    """Return {(r, c): {'E':0/1, 'W':0/1, 'N':0/1, 'S':0/1}} with 1-based cells."""
    rng = random.Random(seed)
    cells = {(r, c): {'E': 0, 'W': 0, 'N': 0, 'S': 0}
             for r in range(1, rows + 1) for c in range(1, cols + 1)}

    def neighbors(cell):
        r, c = cell
        for d, (dr, dc, _) in DIRS.items():
            n = (r + dr, c + dc)
            if n in cells:
                yield d, n

    def open_wall(cell, d, n):
        cells[cell][d] = 1
        cells[n][DIRS[d][2]] = 1

    # Perfect maze via iterative recursive backtracker (guarantees connectivity)
    start = (1, 1)
    visited = {start}
    stack = [start]
    while stack:
        cell = stack[-1]
        unvisited = [(d, n) for d, n in neighbors(cell) if n not in visited]
        if not unvisited:
            stack.pop()
            continue
        d, n = rng.choice(unvisited)
        open_wall(cell, d, n)
        visited.add(n)
        stack.append(n)

    # Open a fraction of the remaining internal walls to create loops
    closed = [(cell, d, n) for cell in cells for d, n in neighbors(cell)
              if cells[cell][d] == 0 and (d in ('E', 'S'))]  # each wall once
    rng.shuffle(closed)
    for cell, d, n in closed[:round(len(closed) * loop_percent / 100)]:
        open_wall(cell, d, n)

    _assert_connected(cells, rows, cols)
    return cells


def _assert_connected(cells, rows, cols):
    seen = {(1, 1)}
    frontier = [(1, 1)]
    while frontier:
        r, c = frontier.pop()
        for d, (dr, dc, _) in DIRS.items():
            n = (r + dr, c + dc)
            if cells[(r, c)][d] == 1 and n in cells and n not in seen:
                seen.add(n)
                frontier.append(n)
    assert len(seen) == rows * cols, 'generated maze is not fully connected'


def save(cells, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['  cell  ', 'E', 'W', 'N', 'S'])
        for (r, c), walls in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            writer.writerow([f'({r}, {c})', walls['E'], walls['W'], walls['N'], walls['S']])


def main():
    parser = argparse.ArgumentParser(description='Generate a maze map CSV')
    parser.add_argument('--rows', type=int, default=35)
    parser.add_argument('--cols', type=int, default=35)
    parser.add_argument('--loop', type=float, default=80,
                        help='percent of leftover walls to open as loops (0-100)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-o', '--out', required=True, help='output CSV path')
    args = parser.parse_args()

    cells = generate(args.rows, args.cols, loop_percent=args.loop, seed=args.seed)
    save(cells, args.out)
    print(f'wrote {args.rows}x{args.cols} maze (loop {args.loop}%, seed {args.seed}) to {args.out}')


if __name__ == '__main__':
    main()
