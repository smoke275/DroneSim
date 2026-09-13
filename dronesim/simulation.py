"""Core multi-strategy simulation engine for autonomous delivery vehicles and
aerial battery swapping drones.

Every strategy shares the same scenario (seeded), the same min-max mTSP routes
and the same exact replenishment planner (dronesim.planning); they differ only
in which replenishment options the planner may use:

1. DEPOT_ONLY:          {depot detour}                      (classical depot E-VRP)
2. FIXED_STATION_EVRP:  {station detour, depot detour}      (classical E-VRP-BSS)
3. REACTIVE_DRONE:      threshold-triggered aerial swap, no planning (baseline)
4. PROACTIVE_FUEL:      {aerial swap, depot detour} + predictive ETA dispatch
                        with Hungarian drone assignment (proposed, Algorithm 2)
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from . import config, planning
from .config import Strategy
from .render import OPERATION
from .sprites import SpriteStore
from .world import World

TRUCK_COLORS = [Qt.darkCyan, Qt.darkMagenta, Qt.darkRed, Qt.darkGreen,
                Qt.darkBlue, Qt.darkYellow, Qt.darkGray, Qt.black]
STATION_COLORS = [Qt.gray, Qt.cyan, Qt.magenta, Qt.darkYellow]

# Caches shared across runs in one process: the world is immutable, and the
# routes depend only on (maze, seed, fleet size, solver), so the paired
# strategies of a benchmark seed reuse them instead of re-solving.
_WORLD_CACHE: Dict[str, World] = {}
_ROUTE_CACHE: Dict[tuple, List[List[int]]] = {}


@dataclass
class Truck:
    pos: tuple
    tasks: list  # (row, col) tuples, in route order
    color: object
    path_color: QColor
    fuel: float = config.TRUCK_RANGE
    path: list = None  # cell path currently being driven
    path_canvas: list = None
    path_index: int = 0
    task_index: int = 0
    completed: set = field(default_factory=set)
    at_warehouse: bool = True
    needs_path_update: bool = True
    holding: bool = False  # true while waiting for an inbound drone or servicing
    station_service_timer: int = 0
    detouring_to_station: bool = False
    detouring_to_warehouse: bool = False
    active_distance: float = 0.0
    detour_distance: float = 0.0
    active_frames: int = 0
    hold_frames: int = 0
    swaps_received: int = 0
    min_fuel: float = config.TRUCK_RANGE  # lowest fuel seen; < 0 flags an energy violation
    # Predictive aerial swap bookkeeping (planned strategies only)
    swap_cell: Optional[tuple] = None    # route cell where the planned swap must be done by
    waiting_for_swap: bool = False       # parked at swap_cell until the swap completes
    route_length: float = 0.0            # planned depot-to-depot route length
    # Energy enforcement: stranded trucks cannot move until rescued
    stranded: bool = False
    strand_events: int = 0
    stranded_frames: int = 0
    rescue_timer: int = 0                # ground recovery countdown (frames)


@dataclass
class Drone:
    pos: list  # [x, y]
    home: int  # base station index
    range_left: float = config.DRONE_RANGE
    target: int = None  # index of the truck being serviced, or None
    target_point: tuple = None  # planned rendezvous point (FUEL); None = chase the truck
    returning: bool = False
    service_timer: int = 0
    total_flight_dist: float = 0.0
    swaps_delivered: int = 0

    @property
    def on_mission(self):
        return self.target is not None or self.returning


class Simulation:

    def __init__(self, window=None, strategy: Strategy = config.DEFAULT_STRATEGY,
                 num_trucks: int = config.NUM_TRUCKS,
                 num_stations: int = config.NUM_BASE_STATIONS,
                 drones_per_station: int = config.DRONES_PER_STATION,
                 seed: Optional[int] = None,
                 maze_path: str = 'maze.csv',
                 service_frames: Optional[int] = None,
                 truck_range: Optional[float] = None,
                 solver: Optional[str] = None,
                 num_tasks: Optional[int] = None):
        self.window = window
        self.strategy = strategy
        self.num_trucks = num_trucks
        self.num_stations = num_stations
        self.drones_per_station = drones_per_station
        self.maze_path = maze_path
        self.truck_range = float(truck_range or config.TRUCK_RANGE)
        self.solver = solver or config.ROUTING_SOLVER
        # Swap service time (applies to both aerial and fixed-station swaps so
        # latency comparisons stay fair); default from config.
        self.drone_service_frames = service_frames or config.DRONE_SERVICE_FRAMES
        self.station_service_frames = service_frames or config.STATION_SERVICE_FRAMES
        if maze_path not in _WORLD_CACHE:
            _WORLD_CACHE[maze_path] = World(maze_path)
        self.world = _WORLD_CACHE[maze_path]
        self.sprites = None
        self.frame_count = 0

        if seed is not None:
            random.seed(seed)
        # Derive numpy random states from the (seeded) python RNG so that the
        # scenario (stations + tasks) is identical for every strategy run with
        # the same seed — df.sample() ignores random.seed() otherwise.
        station_rs = random.randint(0, 2 ** 32 - 1)
        task_rs = random.randint(0, 2 ** 32 - 1)

        # Base stations on random cells (tuples, so arrival checks against
        # path end-cells compare correctly)
        self.station_cells = [tuple(c) for c in
                              self.world.sample_cells(self.num_stations, random_state=station_rs)]
        self.base_stations = [self.world.cell_to_canvas(r, c) for r, c in self.station_cells]

        # Random tasks, routed with a min-max multi-vehicle TSP on road
        # distances (identical for every strategy on the same seed).
        drawn = random.randint(config.MIN_TASKS, config.MAX_TASKS)  # always drawn: keeps seeds stable
        num_tasks = num_tasks or drawn
        all_tasks = [tuple(c) for c in self.world.sample_cells(num_tasks, random_state=task_rs)]
        self._build_planning_graph(all_tasks)
        routes = self._plan_routes(seed)

        self.trucks: List[Truck] = []
        for i in range(self.num_trucks):
            color = TRUCK_COLORS[i % len(TRUCK_COLORS)]
            path_color = QColor(color)
            path_color.setAlpha(130)
            truck = Truck(
                pos=self.world.warehouse_pos,
                tasks=[self.cells[n] for n in routes[i]],
                color=color,
                path_color=path_color,
                fuel=self.truck_range,
                min_fuel=self.truck_range,
            )
            truck.route_length = planning.route_length(self.dist, self.depot_node, routes[i])
            self.trucks.append(truck)

        self.drones: List[Drone] = []
        if self.strategy in (Strategy.REACTIVE_DRONE, Strategy.PROACTIVE_FUEL):
            for station in range(self.num_stations):
                for _ in range(self.drones_per_station):
                    x, y = self.base_stations[station]
                    self.drones.append(Drone(
                        pos=[x + random.uniform(-5, 5), y + random.uniform(-5, 5)],
                        home=station
                    ))

        self.targeted_trucks: Set[int] = set()
        self.replenishment_opts = self._make_options()

    # ------------------------------------------------------------- planning
    def _build_planning_graph(self, tasks):
        """Depot + tasks + stations as planning nodes with road distances."""
        self.cells: List[tuple] = [self.world.warehouse_cell] + list(tasks) + list(self.station_cells)
        self.depot_node = 0
        self.task_nodes = list(range(1, 1 + len(tasks)))
        self.station_nodes = list(range(1 + len(tasks), len(self.cells)))
        self.node_index: Dict[tuple, int] = {}
        for i, c in enumerate(self.cells):
            self.node_index.setdefault(c, i)
        self.dist = planning.distance_matrix(self.world, self.cells)
        # Aerial swap feasibility/cost per node: nearest base by straight-line
        # flight; a full drone must be able to fly there and back.
        self._swap_cost_cache: Dict[Optional[int], Optional[float]] = {}

    def _plan_routes(self, seed):
        key = (self.maze_path, seed, self.num_trucks, self.solver, tuple(self.cells[n] for n in self.task_nodes))
        if seed is not None and key in _ROUTE_CACHE:
            return _ROUTE_CACHE[key]
        routes = planning.plan_routes(self.dist, self.depot_node, self.task_nodes,
                                      self.num_trucks, seed=seed or 0, solver=self.solver)
        if seed is not None:
            _ROUTE_CACHE[key] = routes
        return routes

    def _swap_cost_at(self, pos_xy) -> Optional[float]:
        """Planning cost (frames) of an aerial swap at a canvas position."""
        best = min(math.dist(pos_xy, base) for base in self.base_stations)
        if 2 * best + config.DRONE_RETURN_MARGIN > config.DRONE_RANGE:
            return None
        return self.drone_service_frames + config.SWAP_ENERGY_WEIGHT * 2 * best

    def _make_options(self) -> Optional[planning.ReplenishmentOptions]:
        if self.strategy == Strategy.DEPOT_ONLY:
            return planning.ReplenishmentOptions(allow_depot=True, truck_speed=config.TRUCK_SPEED)
        if self.strategy == Strategy.FIXED_STATION_EVRP:
            return planning.ReplenishmentOptions(
                allow_depot=True, allow_station=True, station_nodes=self.station_nodes,
                service_frames=self.station_service_frames, truck_speed=config.TRUCK_SPEED)
        if self.strategy == Strategy.PROACTIVE_FUEL:
            def swap_cost(node):
                if node is None:  # current truck location, resolved by the caller
                    return self._swap_cost_here
                if node not in self._swap_cost_cache:
                    self._swap_cost_cache[node] = self._swap_cost_at(self.world.cell_to_canvas(*self.cells[node]))
                return self._swap_cost_cache[node]
            return planning.ReplenishmentOptions(
                allow_depot=True, swap_cost=swap_cost, truck_speed=config.TRUCK_SPEED)
        return None  # reactive baseline plans nothing

    # ----------------------------------------------------------------- status
    def is_done(self) -> bool:
        """True when all tasks are delivered, every truck is back at the
        warehouse, and every drone is docked at its base station."""
        tasks_complete = all(len(t.completed) == len(t.tasks) for t in self.trucks)
        trucks_home = all(t.at_warehouse for t in self.trucks)
        drones_idle = all(not d.on_mission for d in self.drones)
        return tasks_complete and trucks_home and drones_idle

    def get_metrics(self) -> Dict:
        """Compute performance metrics for benchmarking."""
        total_active_dist = sum(t.active_distance for t in self.trucks)
        total_detour_dist = sum(t.detour_distance for t in self.trucks)
        total_truck_dist = total_active_dist + total_detour_dist
        total_swaps = sum(t.swaps_received for t in self.trucks)
        total_drone_flight = sum(d.total_flight_dist for d in self.drones)
        total_drone_energy = total_drone_flight * config.ENERGY_DRONE_PER_DIST
        energy_delivered = total_swaps * config.SWAP_BOOST_CAPACITY

        total_active_frames = sum(t.active_frames for t in self.trucks)
        total_possible_frames = max(1, len(self.trucks) * max(1, self.frame_count))
        uptime_ratio = total_active_frames / total_possible_frames

        efficiency = (energy_delivered / (total_drone_energy + 1e-6)) if total_drone_energy > 0 else 0.0

        tasks_total = sum(len(t.tasks) for t in self.trucks)
        tasks_done = sum(len(t.completed) for t in self.trucks)

        return {
            'strategy': self.strategy.value,
            'makespan': self.frame_count,
            'completed': self.is_done(),
            'tasks_done': tasks_done,
            'tasks_total': tasks_total,
            'fuel_violations': sum(1 for t in self.trucks if t.min_fuel < 0 or t.strand_events > 0),
            'strand_events': sum(t.strand_events for t in self.trucks),
            'stranded_frames': sum(t.stranded_frames for t in self.trucks),
            'min_fuel': round(min((t.min_fuel for t in self.trucks), default=0.0), 2),
            'total_truck_distance': round(total_truck_dist, 2),
            'active_distance': round(total_active_dist, 2),
            'detour_distance': round(total_detour_dist, 2),
            'detour_percentage': round((total_detour_dist / max(1.0, total_truck_dist)) * 100, 2),
            'uptime_ratio': round(uptime_ratio, 4),
            'hold_frames': sum(t.hold_frames for t in self.trucks),
            'total_swaps': total_swaps,
            'drone_flight_distance': round(total_drone_flight, 2),
            'drone_energy': round(total_drone_energy, 2),
            'drone_efficiency_percentage': round(efficiency * 100, 2),
            'max_route_length': round(max((t.route_length for t in self.trucks), default=0.0), 2),
            'truck_range': self.truck_range,
            'num_tasks': tasks_total,
        }

    # ----------------------------------------------------------------- main loop
    def step(self):
        """Execute one simulation tick."""
        self.frame_count += 1
        self._update_holding()
        if self.strategy == Strategy.PROACTIVE_FUEL:
            self._dispatch_predictive()
        self._update_drones()
        self._update_trucks()

    def run(self, max_iterations: int = config.ITERATIONS):
        """Run the simulation either with Qt GUI or headless."""
        if self.window is not None:
            self.sprites = SpriteStore()
            self.window.set_static_ops(self._maze_ops())
            for frame in range(max_iterations):
                self.step()
                self._draw_tasks()
                self._draw_landmarks()
                self._draw_routes()
                self._draw_trucks(frame)
                self._draw_drones()
                self._draw_hud()
                self.window.execute()
                time.sleep(1 / config.FPS)
                if self.is_done():
                    break
        else:
            # Headless execution
            for _ in range(max_iterations):
                self.step()
                if self.is_done():
                    break

    # -------------------------------------------------------------- helpers
    def _update_holding(self):
        """Trucks hold while parked for a planned swap, while a fixed station
        services them, or while an inbound drone is within the rendezvous radius."""
        for truck in self.trucks:
            if truck.detouring_to_station and truck.station_service_timer > 0:
                continue
            truck.holding = truck.waiting_for_swap or truck.stranded

        for drone in self.drones:
            if drone.target is not None and not drone.returning and drone.target_point is None:
                truck = self.trucks[drone.target]
                d = math.hypot(drone.pos[0] - truck.pos[0], drone.pos[1] - truck.pos[1])
                if d <= config.DRONE_STOP_RADIUS:
                    truck.holding = True

        for truck in self.trucks:
            if truck.holding:
                truck.hold_frames += 1

    def _drone_can_serve(self, drone: Drone, point_xy) -> Optional[float]:
        """Flight distance to ``point_xy`` if the drone can get there and
        still return home (constraint 7d), else None."""
        flight = math.dist(drone.pos, point_xy)
        back = math.dist(point_xy, self.base_stations[drone.home])
        if flight + back + config.DRONE_RETURN_MARGIN > drone.range_left:
            return None
        return flight

    def _dispatch_drone(self, truck_index: int) -> bool:
        """Reactive baseline: send the nearest idle drone that can reach the
        truck and return home."""
        if self.strategy not in (Strategy.REACTIVE_DRONE, Strategy.PROACTIVE_FUEL):
            return False
        if truck_index in self.targeted_trucks:
            return False

        truck = self.trucks[truck_index]
        best_drone = None
        best_dist = float('inf')
        for drone in self.drones:
            if drone.target is None and not drone.returning:
                d = self._drone_can_serve(drone, truck.pos)
                if d is not None and d < best_dist:
                    best_drone = drone
                    best_dist = d

        if best_drone is not None:
            best_drone.target = truck_index
            self.targeted_trucks.add(truck_index)
            return True
        return False

    def _frames_to_cell(self, truck: Truck, cell: tuple) -> float:
        """Truck travel time (frames) along its route to ``cell``."""
        if truck.waiting_for_swap or not truck.path_canvas:
            return 0.0
        path = truck.path_canvas
        dist = 0.0
        if truck.path_index + 1 < len(path):
            dist += math.dist(truck.pos, path[truck.path_index + 1])
            for a, b in zip(path[truck.path_index + 1:], path[truck.path_index + 2:]):
                dist += math.dist(a, b)
        end_cell = tuple(truck.path[-1])
        if end_cell != cell:
            remaining = [t for t in truck.tasks if t not in truck.completed]
            prev = self.node_index.get(end_cell)
            for t in remaining:
                if prev is None:
                    break
                if t == end_cell:
                    continue
                cur = self.node_index[t]
                dist += self.dist[prev][cur]
                prev = cur
                if t == cell:
                    break
        return dist / config.TRUCK_SPEED

    def _dispatch_predictive(self):
        """FUEL: ETA-triggered dispatch with Hungarian drone assignment.

        A pending request (truck with a planned swap cell, no drone inbound)
        activates once the truck's time-to-node is within the closest feasible
        drone's ETA (+ lead), or immediately if the truck is already parked.
        Active requests are assigned to drones minimising total expected
        truck waiting time (lateness), ETA as tie-break.
        """
        pending = []
        for k, truck in enumerate(self.trucks):
            if truck.swap_cell is None or k in self.targeted_trucks:
                continue
            pending.append((k, self.world.cell_to_canvas(*truck.swap_cell),
                            self._frames_to_cell(truck, truck.swap_cell)))
        if not pending:
            return
        idle = [d for d in self.drones if d.target is None and not d.returning]
        if not idle:
            return

        cost = []
        rows = []
        for k, point, t_truck in pending:
            row = []
            for drone in idle:
                flight = self._drone_can_serve(drone, point)
                if flight is None:
                    row.append(planning.INF)
                    continue
                eta = flight / config.DRONE_SPEED
                row.append((max(0.0, eta - t_truck), eta))
            feasible = [c for c in row if c != planning.INF]
            if not feasible:
                continue
            best_eta = min(c[1] for c in feasible)
            active = self.trucks[k].waiting_for_swap or t_truck <= best_eta + config.DISPATCH_LEAD_FRAMES
            if not active:
                continue
            cost.append([c if c == planning.INF else c[0] + 1e-3 * c[1] for c in row])
            rows.append(k)
        if not rows:
            return
        for r, c in planning.assign_drones(cost):
            truck_index = rows[r]
            idle[c].target = truck_index
            idle[c].target_point = self.world.cell_to_canvas(*self.trucks[truck_index].swap_cell)
            self.targeted_trucks.add(truck_index)

    def _release_drone(self, drone: Drone):
        if drone.target is not None:
            # The truck may be idling while it waits for this drone (reactive
            # dispatch leaves it without a path): make sure it replans, both
            # after a completed swap and after a mid-flight abort.
            self.trucks[drone.target].needs_path_update = True
        self.targeted_trucks.discard(drone.target)
        drone.target = None
        drone.target_point = None
        drone.service_timer = 0
        drone.returning = True

    def _path_length(self, path: Optional[List]) -> float:
        return (len(path) - 1) * self.world.cell_size if path else 0.0

    def _find_nearest_station(self, from_cell: Tuple[int, int]) -> Tuple[Tuple[int, int], List, float]:
        """Find the nearest base station and shortest path for fixed-station EVRP."""
        best_cell = None
        best_path = None
        best_dist = float('inf')
        for sc in self.station_cells:
            p = self.world.find_shortest_path(from_cell, sc)
            if p:
                d = self._path_length(p)
                if d < best_dist:
                    best_dist = d
                    best_path = p
                    best_cell = sc
        return best_cell, best_path, best_dist

    # ------------------------------------------------------------- updates
    def _update_drones(self):
        for drone in self.drones:
            x, y = drone.pos
            dx = dy = 0

            if drone.target is not None and not drone.returning:
                truck = self.trucks[drone.target]
                # Planned rendezvous: fly to the node; once the truck is parked
                # there, close on the truck itself so the service radius is met.
                goal = truck.pos if (drone.target_point is None or truck.waiting_for_swap) else drone.target_point
                dx, dy = goal[0] - x, goal[1] - y
                dist = math.hypot(dx, dy)
                truck_dist = math.hypot(truck.pos[0] - x, truck.pos[1] - y)

                if truck_dist <= config.DRONE_SERVICE_DIST:
                    # Hover over holding truck while mobile swap executes
                    drone.service_timer += 1
                    if drone.service_timer >= self.drone_service_frames:
                        # REPLENISH TRUCK FUEL
                        truck.fuel = self.truck_range
                        truck.swaps_received += 1
                        truck.swap_cell = None
                        truck.waiting_for_swap = False
                        truck.stranded = False
                        drone.swaps_delivered += 1
                        self._release_drone(drone)
                elif dist <= config.DRONE_SERVICE_DIST:
                    pass  # hovering at the rendezvous point, waiting for the truck
                else:
                    step = min(config.DRONE_SPEED, dist)
                    nx_, ny_ = x + step * dx / dist, y + step * dy / dist
                    home_after = math.dist((nx_, ny_), self.base_stations[drone.home])
                    if drone.range_left - step < home_after + config.DRONE_RETURN_MARGIN:
                        # Continuing would strand the drone: abort and go home
                        self._release_drone(drone)
                        continue
                    drone.pos[0], drone.pos[1] = nx_, ny_
                    drone.range_left -= step
                    drone.total_flight_dist += step

            elif drone.returning:
                hx, hy = self.base_stations[drone.home]
                dx, dy = hx - x, hy - y
                dist = math.hypot(dx, dy)
                if dist < 3:
                    drone.returning = False
                    drone.range_left = config.DRONE_RANGE  # recharge at home station
                else:
                    step = min(config.DRONE_SPEED, dist)
                    drone.pos[0] += step * dx / dist
                    drone.pos[1] += step * dy / dist
                    drone.range_left -= step
                    drone.total_flight_dist += step

    def _update_trucks(self):
        for i, truck in enumerate(self.trucks):
            if truck.stranded:
                truck.stranded_frames += 1
                if truck.rescue_timer > 0:
                    # Ground recovery: a service vehicle drives out from the
                    # depot with a battery and back (2 x road distance) and
                    # performs the exchange.
                    truck.rescue_timer -= 1
                    if truck.rescue_timer <= 0:
                        truck.fuel = self.truck_range
                        truck.swaps_received += 1
                        truck.stranded = False
                        truck.needs_path_update = True
                elif self.strategy == Strategy.REACTIVE_DRONE:
                    self._dispatch_drone(i)  # keep asking until a drone accepts
                continue

            # Fixed station servicing hold
            if truck.detouring_to_station and truck.station_service_timer > 0:
                truck.station_service_timer -= 1
                if truck.station_service_timer <= 0:
                    truck.fuel = self.truck_range
                    truck.swaps_received += 1
                    truck.detouring_to_station = False
                    truck.holding = False
                    truck.needs_path_update = True
                continue

            if truck.needs_path_update:
                self._plan_truck_path(truck, i)
                truck.needs_path_update = False

            if truck.path_canvas and not truck.holding:
                self._move_truck(i, truck)

    def _plan_truck_path(self, truck: Truck, truck_idx: int):
        w = self.world
        # Plan from the cell the truck is actually in — the previous path
        # may already be cleared (task arrival) or stale (mid-route replan
        # after a drone swap), and using anything else sends the truck in
        # a straight line through walls.
        start = w.warehouse_cell if truck.at_warehouse else w.canvas_to_cell(*truck.pos)
        remaining = [t for t in truck.tasks if t not in truck.completed]

        if not remaining:
            if not truck.at_warehouse:
                # All assigned tasks completed: return to warehouse
                self._set_truck_path(truck, w.find_shortest_path(start, w.warehouse_cell))
                truck.detouring_to_warehouse = True
            return

        if self.strategy == Strategy.REACTIVE_DRONE:
            self._plan_truck_path_reactive(truck, truck_idx, start, remaining)
            return

        # A drone is already inbound for a committed swap: keep driving the
        # route (the plan guarantees the swap cell is reachable); the truck
        # parks there if the drone has not met it en route.
        if truck.swap_cell is not None and truck_idx in self.targeted_trucks:
            self._drive_to(truck, start, remaining[0])
            return

        route_nodes = [self.node_index[t] for t in remaining] + [self.depot_node]
        start_dists = planning.cell_distances(w, start, self.cells)
        fuel = self.truck_range if truck.at_warehouse else truck.fuel
        if self.strategy == Strategy.PROACTIVE_FUEL:
            self._swap_cost_here = self._swap_cost_at(truck.pos)
        plan = planning.plan_replenishment(
            self.dist, route_nodes, self.depot_node, start_dists, fuel,
            self.truck_range, config.SAFE_RETURN_MARGIN, self.replenishment_opts,
            allow_start_refill=not truck.at_warehouse)

        truck.swap_cell = None
        truck.waiting_for_swap = False
        if plan is None:
            self._plan_truck_path_fallback(truck, truck_idx, start, remaining, start_dists)
            return

        first = plan[0] if plan else None
        if first is not None and first[0] == 0:
            pos, kind, station = first
            if kind == 'depot':
                self._set_truck_path(truck, w.find_shortest_path(start, w.warehouse_cell))
                truck.at_warehouse = False
                truck.detouring_to_warehouse = True
                return
            if kind == 'station':
                self._set_truck_path(truck, w.find_shortest_path(start, self.cells[station]))
                truck.at_warehouse = False
                truck.detouring_to_station = True
                return
            # swap now: park here until a drone completes the exchange
            truck.swap_cell = start
            truck.waiting_for_swap = True
            self._drive_to(truck, start, remaining[0])
            return

        self._drive_to(truck, start, remaining[0])
        if first is not None and first[1] == 'swap':
            truck.swap_cell = remaining[first[0] - 1]

    def _drive_to(self, truck: Truck, start, cell):
        self._set_truck_path(truck, self.world.find_shortest_path(start, cell))
        truck.at_warehouse = False
        truck.detouring_to_warehouse = False
        truck.detouring_to_station = False

    def _plan_truck_path_fallback(self, truck, truck_idx, start, remaining, start_dists):
        """No feasible plan exists (some task cannot be reached and left again
        within the usable range under this strategy's options). Fall back to
        the myopic per-leg rule of Algorithm 2: proceed if the leg plus the
        return to the depot is affordable, otherwise swap (FUEL) or go home."""
        w = self.world
        next_task = remaining[0]
        fuel = self.truck_range if truck.at_warehouse else truck.fuel
        need = (start_dists[self.node_index[next_task]]
                + self.dist[self.node_index[next_task]][self.depot_node]
                + config.SAFE_RETURN_MARGIN)
        if truck.at_warehouse or fuel >= need:
            self._drive_to(truck, start, next_task)
        elif self.strategy == Strategy.PROACTIVE_FUEL and self._swap_cost_at(truck.pos) is not None:
            truck.swap_cell = start
            truck.waiting_for_swap = True
            self._drive_to(truck, start, next_task)
        else:
            self._set_truck_path(truck, w.find_shortest_path(start, w.warehouse_cell))
            truck.at_warehouse = False
            truck.detouring_to_warehouse = True

    def _plan_truck_path_reactive(self, truck, truck_idx, start, remaining):
        """Baseline 3: no prediction. Drive if the next leg is reachable and
        call a drone only once the battery is below the critical threshold."""
        w = self.world
        next_task = remaining[0]
        path_to_task = w.find_shortest_path(start, next_task)
        if not path_to_task:
            return
        dist_to_task = self._path_length(path_to_task)
        dist_task_to_depot = self._path_length(w.find_shortest_path(next_task, w.warehouse_cell))
        safe_margin = config.SAFE_RETURN_MARGIN

        if truck.fuel >= dist_to_task + dist_task_to_depot + safe_margin or truck.at_warehouse:
            self._set_truck_path(truck, path_to_task)
            truck.at_warehouse = False
        elif truck.fuel >= dist_to_task + safe_margin:
            self._set_truck_path(truck, path_to_task)
            truck.at_warehouse = False
            if truck.fuel / self.truck_range <= config.REACTIVE_THRESHOLD:
                self._dispatch_drone(truck_idx)
        else:
            # Critical: try requesting drone on the spot or return to warehouse
            dispatched = self._dispatch_drone(truck_idx)
            if not dispatched:
                self._set_truck_path(truck, w.find_shortest_path(start, w.warehouse_cell))
                truck.detouring_to_warehouse = True

    def _strand(self, index: int, truck: Truck):
        """Battery exhausted: the truck stops where it is until rescued."""
        truck.fuel = 0.0
        truck.min_fuel = min(truck.min_fuel, 0.0)
        truck.stranded = True
        truck.strand_events += 1
        cell = self.world.canvas_to_cell(*truck.pos)
        aerial = (self.strategy in (Strategy.REACTIVE_DRONE, Strategy.PROACTIVE_FUEL)
                  and self._swap_cost_at(truck.pos) is not None)
        if aerial:
            if self.strategy == Strategy.PROACTIVE_FUEL:
                truck.swap_cell = cell
                truck.waiting_for_swap = True
            else:
                self._dispatch_drone(index)
        else:
            road = planning.cell_distances(self.world, cell, [self.world.warehouse_cell])[0]
            truck.rescue_timer = int(2 * road / config.TRUCK_SPEED) + self.station_service_frames

    def _set_truck_path(self, truck: Truck, path: Optional[List]):
        if not path:
            return
        truck.path = path
        truck.path_canvas = [self.world.cell_to_canvas(r, c) for r, c in path]
        truck.path_index = 0

    def _move_truck(self, index: int, truck: Truck):
        path = truck.path_canvas
        nxt = path[(truck.path_index + 1) % len(path)]
        dx, dy = nxt[0] - truck.pos[0], nxt[1] - truck.pos[1]
        dist = math.hypot(dx, dy)

        if dist > 1:
            step = config.TRUCK_SPEED
            if config.STRAND_ON_EMPTY and truck.fuel < step:
                self._strand(index, truck)
                return
            truck.pos = (truck.pos[0] + step * dx / dist, truck.pos[1] + step * dy / dist)
            truck.fuel -= step
            truck.min_fuel = min(truck.min_fuel, truck.fuel)

            if truck.detouring_to_warehouse or truck.detouring_to_station:
                truck.detour_distance += step
            else:
                truck.active_distance += step
                truck.active_frames += 1
            return

        truck.path_index = (truck.path_index + 1) % len(path)
        if truck.path_index != len(path) - 1:
            return

        # Arrived at end of path
        end_cell = tuple(truck.path[-1])
        if end_cell == self.world.warehouse_cell and truck.detouring_to_warehouse:
            truck.fuel = self.truck_range
            truck.at_warehouse = True
            truck.detouring_to_warehouse = False
        elif truck.detouring_to_station and end_cell in self.station_cells:
            # Arrived at fixed charging station
            truck.station_service_timer = self.station_service_frames
            truck.holding = True
        else:
            # Arrived at task
            truck.completed.add(end_cell)
            truck.task_index += 1
            truck.at_warehouse = False
            if truck.swap_cell == end_cell:
                # Planned swap point reached before the drone met us: park.
                truck.waiting_for_swap = True

        truck.path = None
        truck.path_canvas = None
        truck.path_index = 0
        truck.needs_path_update = True

    # ------------------------------------------------------------- drawing
    def _maze_ops(self):
        ops = []
        w = self.world
        grid = QColor(96, 125, 139, 35)
        left, _ = w.cell_to_canvas(1, 1)
        left -= w.cell_size / 2
        top = config.BOUNDARY_Y
        for c in range(w.max_col + 1):
            x = left + c * w.cell_size
            ops.append([OPERATION.line, x, top, x, top - w.max_row * w.cell_size, 1, grid])
        for r in range(w.max_row + 1):
            y = top - r * w.cell_size
            ops.append([OPERATION.line, left, y, left + w.max_col * w.cell_size, y, 1, grid])
        for x1, y1, x2, y2 in w.wall_segments:
            ops.append([OPERATION.wall, x1, y1, x2, y2])
        return ops

    def _draw_tasks(self):
        size = 5
        for truck in self.trucks:
            for cell in truck.tasks:
                if cell in truck.completed:
                    continue
                x, y = self.world.cell_to_canvas(*cell)
                self.window.draw([OPERATION.filled_polygon,
                                  [x - size / 2, x + size / 2, x + size / 2, x - size / 2],
                                  [y - size / 2, y - size / 2, y + size / 2, y + size / 2],
                                  1, truck.color])

    def _draw_landmarks(self):
        draw = self.window.draw
        wx, wy = self.world.warehouse_pos
        draw([OPERATION.filled_circle, wx, wy, 16, 1, QColor('#34495e')])
        draw([OPERATION.filled_circle, wx, wy, 6, 1, QColor('#ecf0f1')])
        draw([OPERATION.text, wx - 38, -(wy - 30), 1, QColor('#34495e'), 'WAREHOUSE'])
        for i, (x, y) in enumerate(self.base_stations):
            size = 20
            draw([OPERATION.border_polygon,
                  [x - size / 2, x + size / 2, x + size / 2, x - size / 2],
                  [y - size / 2, y - size / 2, y + size / 2, y + size / 2],
                  1, STATION_COLORS[i % len(STATION_COLORS)]])
            draw([OPERATION.text, x - 18, -(y - 26), 1, QColor('#546e7a'), f'BASE {i + 1}'])

    def _draw_routes(self):
        for truck in self.trucks:
            if truck.path_canvas:
                for (x1, y1), (x2, y2) in zip(truck.path_canvas, truck.path_canvas[1:]):
                    self.window.draw([OPERATION.dotted_line, x1, y1, x2, y2, 1, truck.path_color])
            if truck.swap_cell is not None:
                sx, sy = self.world.cell_to_canvas(*truck.swap_cell)
                ring = QColor('#2980b9')
                ring.setAlpha(160)
                self.window.draw([OPERATION.circle, sx, sy, 12, 2, ring])

    def _draw_trucks(self, frame):
        draw = self.window.draw
        for i, truck in enumerate(self.trucks):
            x, y = truck.pos
            angle = 0
            if truck.path_canvas and len(truck.path_canvas) > 1:
                cur = truck.path_canvas[truck.path_index]
                nxt = truck.path_canvas[(truck.path_index + 1) % len(truck.path_canvas)]
                angle = math.degrees(math.atan2(nxt[1] - cur[1], nxt[0] - cur[0]))

            image = self.sprites.rotated('truck', angle)
            draw([OPERATION.image, x - image.width() // 2, y - image.height() // 2, image])
            self._draw_gauge(x, y + 30, 30, 5, truck.fuel / self.truck_range)

            if i in self.targeted_trucks:
                pulse = config.DRONE_STOP_RADIUS * (0.92 + 0.08 * math.sin(frame * 0.15))
                ring = QColor('#2980b9')
                ring.setAlpha(110)
                draw([OPERATION.circle, x, y, pulse, 2, ring])
                if truck.holding:
                    draw([OPERATION.text, x - 28, -(y + 40), 1, QColor('#c0392b'), 'HOLDING'])
            elif truck.stranded:
                draw([OPERATION.text, x - 28, -(y + 40), 1, QColor('#c0392b'), 'STRANDED'])
            elif truck.waiting_for_swap:
                draw([OPERATION.text, x - 28, -(y + 40), 1, QColor('#c0392b'), 'WAITING'])
            elif truck.detouring_to_station and truck.station_service_timer > 0:
                draw([OPERATION.text, x - 28, -(y + 40), 1, QColor('#27ae60'), 'SWAPPING'])

    def _draw_drones(self):
        draw = self.window.draw
        for drone in self.drones:
            x, y = drone.pos
            dx = dy = 0
            if drone.target is not None and not drone.returning:
                truck = self.trucks[drone.target]
                goal = drone.target_point if drone.target_point is not None else truck.pos
                dx, dy = goal[0] - x, goal[1] - y
                link = QColor('#2980b9')
                link.setAlpha(150)
                draw([OPERATION.dotted_line, x, y, goal[0], goal[1], 1, link])
            elif drone.returning:
                hx, hy = self.base_stations[drone.home]
                dx, dy = hx - x, hy - y

            angle = math.degrees(math.atan2(dy, dx))
            image = self.sprites.rotated('drone', angle)
            draw([OPERATION.image, x - image.width() // 2, y - image.height() // 2, image])
            if drone.on_mission:
                self._draw_gauge(x, y + 20, 24, 3, drone.range_left / config.DRONE_RANGE)

    def _draw_gauge(self, cx, cy, width, height, frac):
        frac = max(0.0, min(1.0, frac))
        color = QColor('#2ecc71') if frac > 0.5 else (
            QColor('#f5a623') if frac > 0.25 else QColor('#e74c3c'))
        x = cx - width / 2
        xs = [x, x + width, x + width, x]
        ys = [cy, cy, cy + height, cy + height]
        self.window.draw([OPERATION.filled_polygon, xs, ys, 1, QColor('#cfd8dc')])
        self.window.draw([OPERATION.filled_polygon,
                          [x, x + frac * width, x + frac * width, x], ys, 1, color])
        self.window.draw([OPERATION.polygon, xs, ys, 1, QColor('#607d8b')])

    def _draw_hud(self):
        draw = self.window.draw
        en_route = sum(1 for d in self.drones if d.target is not None)
        returning = sum(1 for d in self.drones if d.returning)
        docked = len(self.drones) - en_route - returning
        holding = sum(1 for t in self.trucks if t.holding)
        tasks_left = sum(len(t.tasks) - len(t.completed) for t in self.trucks)

        draw([OPERATION.hud_panel, 10, 10, 370, 72])
        draw([OPERATION.hud_text, 20, 28, '#263238', f'STRATEGY: {self.strategy.value.upper()}'])
        draw([OPERATION.hud_text, 20, 48, '#263238',
              f'DRONES  en route {en_route}  returning {returning}  docked {docked}'])
        draw([OPERATION.hud_text, 20, 68, '#263238',
              f'TRUCKS  holding {holding}  tasks remaining {tasks_left}'])
