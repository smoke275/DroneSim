# FUEL Fleet Emulation (ROS 2 Jazzy + Gazebo Harmonic)

Docker-based high-fidelity recreation of the FUEL experiment: **6 delivery
trucks** start at the central warehouse and patrol routes through the full
maze; **4 quadcopters** (X3 model from the Gazebo Fuel library, full
multicopter rotor dynamics) dock at base stations in the four quadrants. Every
vehicle runs its **own controller node**, and a central **BMS dispatcher**
monitors each truck's state of charge and assigns the nearest idle drone when
one drops below the predictive threshold — the ROS counterpart of the fleet
simulator's Algorithm 2.

A rendezvous cycle: dispatch → drone flies over the maze walls → truck holds →
drone hovers over the deck for `T_swap` (the mechanical exchange is
characterized on the bench prototype, not visually modeled) → SOC resets →
drone returns, lands, recharges → available again. Trucks left waiting queue
until a drone frees up.

## Controls

### Start

```bash
./start.sh          # builds the image, starts an idle container, attaches a shell
```

Then, inside the container:

```bash
ros2 launch fuel_rendezvous rendezvous.launch.py                  # with the Gazebo window
ros2 launch fuel_rendezvous rendezvous.launch.py headless:=true   # server only, logs + metrics
```

### Stop

- `Ctrl+C` in the launch shell, or from the host:

```bash
./stop.sh           # kill mission + Gazebo, keep the container idle for a fast relaunch
./stop.sh --all     # also remove the container
```

(Gazebo's server runs under a ruby wrapper that sometimes survives Ctrl+C;
`stop.sh` handles it.)

### Change the world / fleet

```bash
python3 gen_world.py --maze ../maze.csv --trucks 6 --drones 4 --cell 3.0
./start.sh          # rebuild the image with the new world
```

`gen_world.py` emits the maze walls, warehouse, pads, robots, the ros_gz
bridge config, and `config/fleet.yaml` (routes/pads consumed by the launch
file). Any maze CSV from the fleet simulator works (`../maps/*.csv`).

### Disturbances (wind, GNSS noise, telemetry loss)

The launch file takes disturbance arguments; all default to the nominal
(undisturbed) experiment:

| Argument | Meaning | Default |
|---|---|---|
| `wind_speed`, `wind_dir_deg` | mean horizontal wind [m/s], direction [deg] | 0, 0 |
| `wind_gust`, `wind_gust_period` | Ornstein-Uhlenbeck gust std [m/s] and correlation time [s] | 0, 5 |
| `pos_noise` | Gaussian GNSS noise on the drone's own and truck position estimates [m] | 0 |
| `drop_prob` | probability that an odometry/telemetry message is lost (estimate goes stale) | 0 |
| `seed`, `run_tag` | RNG seed; tag written into the metrics file names and rows | 0, '' |

```bash
ros2 launch fuel_rendezvous rendezvous.launch.py headless:=true wind_speed:=6 wind_gust:=2 pos_noise:=0.5
```

Wind is physical: the world carries Gazebo's `WindEffects` system and the
drones use a local, wind-enabled copy of the X3 model (`models/x3_wind`). The
`wind_field` node sets the wind vector at runtime on `/world/swap_world/wind`
and logs the realised wind to `/tmp/wind_<tag>.csv`. The plugin applies
`F = m * k * (v_wind - v_link)` with `k = 0.1` (`gen_world.py --wind-drag`),
about 0.75 N on the 1.5 kg X3 at 5 m/s, comparable to its bluff-body drag.
Control uses the noisy/stale estimates; metrics use ground truth. A truck
whose SOC reaches zero is immobilised (`STRANDED` in the log) until a drone
rescues it.

The sweep script runs every condition headless for a fixed wall time and
collects the results, then the summary script tabulates them:

```bash
./run_disturbance_sweep.sh                    # 7 conditions x 300 s -> ../results/gazebo/<condition>/
DURATION=120 ./run_disturbance_sweep.sh calm wind6_gust2
python3 summarize_gazebo.py                   # -> results/gazebo/summary.csv, summary_table.tex
```

### Metrics

Each drone appends one row per completed rendezvous to
`/tmp/rendezvous_metrics_[<tag>_]<name>.csv` inside the container (takeoff,
approach time, hover mean / p95 / max error, fraction of hover samples within
the 10 cm mechanism tolerance and within the capture cone, capture losses,
return time, cycle total, realised message-drop fraction, condition):

```bash
docker cp fuel-gz-run:/tmp/rendezvous_metrics_x3_0.csv .
```

Dispatch decisions and swaps are in the launch log
(`docker exec fuel-gz-run grep -E "DISPATCH|swapped|METRICS" /tmp/launch.log`).

### Gazebo GUI

Scroll = zoom, left-drag = rotate, right-drag = pan. The site spans ~105 m —
zoom out first. Landmarks: dark cylinder = warehouse (trucks start there),
blue discs = drone pads. Entity Tree → right-click a robot → *Move To* or
*Follow* to track it. The orange buttons (bottom left) pause/step physics.

## Architecture

| Piece | File | Role |
|---|---|---|
| world generator | `gen_world.py` | maze CSV → SDF world + bridge + fleet config |
| drone controller (×N) | `fuel_rendezvous/drone_agent.py` | flight FSM: IDLE → ARM → TAKEOFF → INTERCEPT → HOVER_SWAP → RETURN → LAND → RECHARGE; GNSS noise / message loss on estimates; per-cycle metrics |
| wind field (×1) | `fuel_rendezvous/wind_field.py` | mean wind + OU gusts published to Gazebo's WindEffects at runtime |
| truck controller (×M) | `fuel_rendezvous/ugv_agent.py` | waypoint route follower (ping-pong patrol), distance-based battery model, hold/swap handling |
| BMS dispatcher (×1) | `fuel_rendezvous/bms_dispatcher.py` | SOC monitoring, nearest-idle-drone assignment, lost-message-safe re-publish |
| bridge | `config/bridge.yaml` (generated) | per-robot ROS ↔ Gazebo topics |
| wind-enabled X3 | `models/x3_wind/` | OpenRobotics X3 v4 with `enable_wind`, local meshes |
| sweep / summary | `run_disturbance_sweep.sh`, `summarize_gazebo.py` | disturbance conditions → `results/gazebo/` |
| launch | `launch/rendezvous.launch.py` | gz sim + bridge + one node per robot + BMS |

### Topics (per robot `i`/`j`)

| Topic | Type | Purpose |
|---|---|---|
| `/x3_i/cmd_vel`, `/x3_i/enable` | Twist, Bool | drone velocity command / motor arm (bridged to gz) |
| `/x3_i/odometry`, `/ugv_j/odometry` | Odometry | world-frame ground truth (bridged from gz) |
| `/x3_i/assign` | String | BMS → drone: truck name to service |
| `/x3_i/status` | String | drone → BMS: FSM phase (IDLE = available) |
| `/ugv_j/cmd_vel` | Twist | truck drive command (bridged to gz) |
| `/ugv_j/soc` | Float32 | truck → BMS: state of charge |
| `/ugv_j/hold` | Bool | drone → truck: halt for rendezvous |
| `/ugv_j/swap` | Bool | drone → truck: swap complete, SOC resets |

### Notes

- Gazebo runs in real time — fleet-scale benchmark numbers come from the
  headless `dronesim` simulator; this emulation is the architecture
  demonstration and flight-phase characterization layer.
- `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` is baked into the image: without
  it, ROS nodes in sibling containers on the docker bridge discover each other
  and cross-talk.
- Truck spawns are staggered a cell or two along each route's exit corridor —
  the fleet simulator can overlap vehicles at the warehouse; a physics engine
  cannot.
