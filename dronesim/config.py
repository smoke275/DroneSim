"""Central configuration for the drone logistics simulation."""

from enum import Enum


class Strategy(Enum):
    DEPOT_ONLY = 'depot_only'                  # Baseline 1: Return to central depot to refuel
    FIXED_STATION_EVRP = 'fixed_station_evrp'  # Baseline 2: Detour to nearest static base station
    REACTIVE_DRONE = 'reactive_drone'          # Baseline 3: Drone swap dispatched at low battery threshold
    PROACTIVE_FUEL = 'proactive_fuel'          # Proposed: Predictive BMS aerial drone swap (Algorithm 2)


# Simulation Strategy
DEFAULT_STRATEGY = Strategy.PROACTIVE_FUEL

# Canvas / world
WINDOW_SIZE = 700
BOUNDARY_X = 500
BOUNDARY_Y = 500
ITERATIONS = 10000
FPS = 120

# View controls
MIN_ZOOM = 0.5
MAX_ZOOM = 10.0
ZOOM_STEP = 1.15  # multiplier per wheel notch

# Rendezvous & Swapping behaviour
DRONE_STOP_RADIUS = 80       # trucks halt when an inbound drone is within this radius
DRONE_SERVICE_DIST = 10      # distance at which a drone begins servicing a truck
DRONE_SERVICE_FRAMES = 150   # ~1.25 s of servicing at 120 fps (T_swap)
STATION_SERVICE_FRAMES = 150 # ~1.25 s holding time at fixed station

# Vehicles
TRUCK_RANGE = 3000
DRONE_RANGE = 2000
NUM_TRUCKS = 6
NUM_BASE_STATIONS = 4
DRONES_PER_STATION = 1       # 4 total drones (1 per station) to match paper setup
TRUCK_SPEED = 1
DRONE_SPEED = 3
SAFE_RETURN_MARGIN = 50      # B_safe buffer for reachability guarantees
REACTIVE_THRESHOLD = 0.25    # 25% battery threshold for reactive drone dispatch

# Energy coefficients
ENERGY_TRUCK_PER_DIST = 1.0  # alpha_T
ENERGY_DRONE_PER_DIST = 1.2  # alpha_D (drone flight energy per unit distance)
SWAP_BOOST_CAPACITY = 3000   # Delta B_T (full capacity restoration)

# Tasks
MIN_TASKS = 35
MAX_TASKS = 40
