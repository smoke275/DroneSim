# DroneSim — FUEL: Drone-Delivered Battery Swaps for Last-Mile Fleets

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
| `--max-iter N` | frame cap per run |
| `--out DIR` | output directory (default `results/`) |

Per-run rows land in `<out>/benchmark_results.csv`; a mean±std summary table
prints at the end.

### Sensitivity sweeps and plots

```bash
python sweep.py --seeds 10      # fleet scaling, swap latency, map topology -> results/sweeps/*.csv
python3 plot_sweeps.py          # renders the three paper figures -> results/sweeps/*.png
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
