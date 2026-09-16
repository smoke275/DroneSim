# DroneSim — CHARGE: Drone-Delivered Battery Swaps for Last-Mile Fleets

Two runnable systems live in this repo:

1. **Fleet simulator** (`dronesim/`, PyQt5) — fast 2D simulation of 6 delivery
   trucks + battery-swap drones in a maze road network. Used for all benchmark
   numbers in the paper.
2. **ROS 2 / Gazebo emulation** (`gazebo_emulation/`) — high-fidelity physics
   recreation: every truck and drone runs its own controller node, coordinated
   by a central BMS dispatcher.

The paper lives in `ICRA-27-Battery-Exchange/`.

---

## 1. Fleet simulator

### Start / stop

```bash
./start.sh                      # builds the image, starts an idle container, attaches a shell
python main.py                  # inside the container: launch the GUI simulation
```

The repo is live-mounted at `/app`, so code edits apply without rebuilding.

```bash
docker rm -f dronesim-run       # stop the container (from the host)
```

### main.py flags

| Flag | Meaning | Default |
|---|---|---|
| `--maze PATH` | maze CSV to load | `maze.csv` |
| `--strategy S` | `depot_only` \| `fixed_station_evrp` \| `reactive_drone` \| `proactive_fuel` | `proactive_fuel` |
| `--seed N` | scenario seed (tasks + station placement) | random |
| `--trucks N` | fleet size override | `config.NUM_TRUCKS` |
| `--truck-range R` | battery range override | `config.TRUCK_RANGE` |

### In-window controls

| Input | Action |
|---|---|
| Scroll wheel | zoom (about the cursor, 0.5×–10×) |
| Left-drag | pan |
| `+` / `-` | zoom in / out |
| `R` | reset view |
| `Esc` | quit |

The simulation runs until every task is delivered and all trucks are back at
the warehouse.

### Planning pipeline (shared by every strategy)

Every strategy runs on the **same** seeded scenario and the **same** plan:

1. **Routing** (`dronesim/planning.py`): road-graph shortest-path distances
   between depot, tasks and base stations, then a min-max multi-vehicle TSP
   (minimise the longest truck route, total distance as tie-break). The
   default solver is a deterministic pure-Python construction + local search
   (`config.ROUTING_SOLVER = 'local_search'`); set it to `'ortools'` to use
   Google OR-Tools when installed (`pip install ortools`). Note: on hosts
   where scikit-learn/scipy come from Anaconda, importing OR-Tools after them
   segfaults, which is why it is opt-in.
2. **Replenishment planning**: an exact resource-constrained shortest path
   over each truck's remaining route decides where to refill so that no
   segment exceeds the usable range (`TRUCK_RANGE - SAFE_RETURN_MARGIN`).
   Strategies differ only in the options the planner may use:
   depot detour (Depot-Return), station or depot detour (E-VRP-BSS),
   aerial swap or depot detour (CHARGE). The reactive baseline plans nothing
   and dispatches on a battery threshold.
3. **Dispatch** (CHARGE): a swap request activates once the truck's time to
   its planned rendezvous node is within a drone's ETA (+ lead); active
   requests are matched to idle drones with the Hungarian algorithm on
   expected waiting time. The drone flies to the node; the truck parks
   there only if the drone is late.

Routes are cached per (maze, seed, fleet size) inside a process, so the
paired strategies of one benchmark seed share one routing solve.

**Energy enforcement** (`config.STRAND_ON_EMPTY`, default on): a truck whose
battery reaches zero is stranded and cannot move. Drone strategies rescue it
with a drone swap at the truck's position; ground strategies wait for a
recovery vehicle (a road round trip from the depot plus the swap time). The
`strand_events` / `stranded_frames` metrics count these rescues, and
`fuel_violations` counts trucks that ran out at least once.

**Important:** with real routing, the default scenario (6 trucks, 35–40
tasks, range 3000) never needs a swap — every strategy ties. Energy only
binds with fewer trucks (`--trucks 2` or `3`) or a shorter range
(`--truck-range`); a range below twice the farthest task's depot distance
makes Depot-Return infeasible for that task (reported as an energy violation).

### Benchmark (headless, no display needed)

```bash
python benchmark.py --seeds 30                        # all 4 strategies, paired scenarios
python benchmark.py --seeds 10 --mazes maze.csv maps/*.csv
python benchmark.py --strategies proactive_fuel depot_only --trucks 8
```

| Flag | Meaning |
|---|---|
| `--seeds N` / `--seed-start N` | number of scenarios / first seed |
| `--strategies ...` | subset of strategies |
| `--mazes ...` | maze CSVs to run on |
| `--trucks N` | fleet size override |
| `--drones-per-station N` | drone count override |
| `--service-frames N` | swap service time (120 frames = 1 s) |
| `--truck-range R` | truck battery range override (canvas units) |
| `--tasks N` | task-count override (default random in `[MIN_TASKS, MAX_TASKS]`) |
| `--max-iter N` | frame cap per run |
| `--out DIR` | output directory (default `results/`) |

Per-run rows land in `<out>/benchmark_results.csv`; a mean±std summary table
prints at the end.

### Sensitivity sweeps and plots

```bash
python sweep.py --seeds 10                 # fleet scaling, swap latency, map topology -> results/sweeps/*.csv
python3 plot_sweeps.py                     # renders the three paper figures -> results/sweeps/*.png
python3 plot_sweeps.py results/sweeps_v2   # same, from another sweep directory
```

### Maze generator

```bash
python -m dronesim.mazegen --rows 35 --cols 35 --loop 80 --seed 1 -o maps/my_maze.csv
```

`--loop` is the percent of leftover walls opened as loops: 0 = perfect maze
(single path between any two cells), 80 = dense road network. Output is the
same CSV format as `maze.csv` and works everywhere a maze path is accepted.

---

## 2. ROS 2 / Gazebo emulation

ROS 2 Jazzy + Gazebo Harmonic in Docker. Six trucks start at the central
warehouse, four drones dock at base stations in the quadrants; the BMS
dispatches drones to low-battery trucks continuously. See
[gazebo_emulation/README.md](gazebo_emulation/README.md) for full details.

```bash
cd gazebo_emulation
./start.sh                                                        # build + idle container + shell
ros2 launch fuel_rendezvous rendezvous.launch.py                  # inside: launch with GUI
ros2 launch fuel_rendezvous rendezvous.launch.py headless:=true   # inside: no GUI
./stop.sh                                                         # host: stop mission, keep container
./stop.sh --all                                                   # host: also remove the container
```

Regenerate the world (different maze or fleet size), then rebuild:

```bash
python3 gen_world.py --maze ../maze.csv --trucks 6 --drones 4
./start.sh
```

Per-drone rendezvous metrics accumulate in the container at
`/tmp/rendezvous_metrics_<drone>.csv`:

```bash
docker cp fuel-gz-run:/tmp/rendezvous_metrics_x3_0.csv .
```

### Gazebo GUI tips

- Scroll to zoom, left-drag to rotate, right-drag to pan.
- The map spans ~105 m: zoom out first. Landmarks: dark cylinder = warehouse,
  blue discs = drone pads.
- Entity Tree → right-click a robot (`x3_0`, `ugv_3`, ...) → *Move To* /
  *Follow* to track it.
- Orange pause/play buttons (bottom left) pause the physics.

---

## 3. Paper

```bash
cd ICRA-27-Battery-Exchange
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  sh -c "pdflatex root.tex && bibtex root && pdflatex root.tex && pdflatex root.tex"
```

Benchmark figures are copied from `results/sweeps/` into `Images/`.
