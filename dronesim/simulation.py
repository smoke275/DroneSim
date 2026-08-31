"""Core multi-strategy simulation engine for autonomous delivery vehicles and
aerial battery swapping drones.

Supports 4 comparative logistics strategies:
1. DEPOT_ONLY: Central Depot E-VRP (return to warehouse to refuel)
2. FIXED_STATION_EVRP: Classical E-VRP-BSS (detour to nearest static base station)
3. REACTIVE_DRONE: Drone dispatched reactively when truck SOC is critically low
4. PROACTIVE_FUEL: Predictive BMS aerial drone swap (Algorithm 2)
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from sklearn.cluster import KMeans

from . import config
from .config import Strategy
from .render import OPERATION
from .sprites import SpriteStore
from .world import World

TRUCK_COLORS = [Qt.darkCyan, Qt.darkMagenta, Qt.darkRed, Qt.darkGreen,
                Qt.darkBlue, Qt.darkYellow, Qt.darkGray, Qt.black]
STATION_COLORS = [Qt.gray, Qt.cyan, Qt.magenta, Qt.darkYellow]


@dataclass
class Truck:
    pos: tuple
    tasks: list  # (row, col) tuples, in patrol order
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
    swaps_received: int = 0
    min_fuel: float = config.TRUCK_RANGE  # lowest fuel seen; < 0 flags an energy violation


@dataclass
class Drone:
    pos: list  # [x, y]
    home: int  # base station index
    range_left: float = config.DRONE_RANGE
    target: int = None  # index of the truck being serviced, or None
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
                 service_frames: Optional[int] = None):
        self.window = window
        self.strategy = strategy
        self.num_trucks = num_trucks
        self.num_stations = num_stations
        self.drones_per_station = drones_per_station
        self.maze_path = maze_path
        # Swap service time (applies to both aerial and fixed-station swaps so
        # latency comparisons stay fair); default from config.
        self.drone_service_frames = service_frames or config.DRONE_SERVICE_FRAMES
        self.station_service_frames = service_frames or config.STATION_SERVICE_FRAMES
        self.world = World(maze_path)
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

        # Random tasks, clustered into one group per truck
        num_tasks = random.randint(config.MIN_TASKS, config.MAX_TASKS)
        all_tasks = self.world.sample_cells(num_tasks, random_state=task_rs)
        kmeans_seed = seed if seed is not None else 0
        labels = KMeans(n_clusters=self.num_trucks, random_state=kmeans_seed).fit(all_tasks).labels_
        clustered = [[] for _ in range(self.num_trucks)]
        for task, label in zip(all_tasks, labels):
            clustered[label].append(tuple(task))

        self.trucks: List[Truck] = []
        for i in range(self.num_trucks):
            color = TRUCK_COLORS[i % len(TRUCK_COLORS)]
            path_color = QColor(color)
            path_color.setAlpha(130)
            self.trucks.append(Truck(
                pos=self.world.warehouse_pos,
                tasks=clustered[i],
                color=color,
                path_color=path_color
            ))

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
            'fuel_violations': sum(1 for t in self.trucks if t.min_fuel < 0),
            'min_fuel': round(min((t.min_fuel for t in self.trucks), default=0.0), 2),
            'total_truck_distance': round(total_truck_dist, 2),
            'active_distance': round(total_active_dist, 2),
            'detour_distance': round(total_detour_dist, 2),
            'detour_percentage': round((total_detour_dist / max(1.0, total_truck_dist)) * 100, 2),
            'uptime_ratio': round(uptime_ratio, 4),
            'total_swaps': total_swaps,
            'drone_flight_distance': round(total_drone_flight, 2),
            'drone_energy': round(total_drone_energy, 2),
            'drone_efficiency_percentage': round(efficiency * 100, 2),
        }

    # ----------------------------------------------------------------- main loop
    def step(self):
        """Execute one simulation tick."""
        self.frame_count += 1
        self._update_holding()
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
        """Trucks hold position while an inbound drone is within the rendezvous radius."""
        for truck in self.trucks:
            if not truck.detouring_to_station or truck.station_service_timer <= 0:
                truck.holding = False

        for drone in self.drones:
            if drone.target is not None and not drone.returning:
                truck = self.trucks[drone.target]
                d = math.hypot(drone.pos[0] - truck.pos[0], drone.pos[1] - truck.pos[1])
                if d <= config.DRONE_STOP_RADIUS:
                    truck.holding = True

    def _dispatch_drone(self, truck_index: int) -> bool:
        """Send the nearest available drone from any base station to intercept the truck."""
        if self.strategy not in (Strategy.REACTIVE_DRONE, Strategy.PROACTIVE_FUEL):
            return False
        if truck_index in self.targeted_trucks:
            return False

        truck = self.trucks[truck_index]
        best_drone = None
        best_dist = float('inf')

        for drone in self.drones:
            if drone.target is None and not drone.returning and \
                    drone.range_left >= config.DRONE_RANGE / 2:
                d = math.hypot(drone.pos[0] - truck.pos[0], drone.pos[1] - truck.pos[1])
                if d < best_dist:
                    best_drone = drone
                    best_dist = d

        if best_drone is not None:
            best_drone.target = truck_index
            self.targeted_trucks.add(truck_index)
            return True
        return False

    def _release_drone(self, drone: Drone):
        if drone.target is not None:
            # The truck may be idling while it waits for this drone (reactive
            # dispatch leaves it without a path): make sure it replans, both
            # after a completed swap and after a mid-flight abort.
            self.trucks[drone.target].needs_path_update = True
        self.targeted_trucks.discard(drone.target)
        drone.target = None
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
                dx, dy = truck.pos[0] - x, truck.pos[1] - y
                dist = math.hypot(dx, dy)

                if dist <= config.DRONE_SERVICE_DIST:
                    # Hover over holding truck while mobile swap executes
                    drone.service_timer += 1
                    if drone.service_timer >= self.drone_service_frames:
                        # REPLENISH TRUCK FUEL
                        truck.fuel = config.TRUCK_RANGE
                        truck.swaps_received += 1
                        drone.swaps_delivered += 1
                        self._release_drone(drone)
                else:
                    step = min(config.DRONE_SPEED, dist)
                    drone.pos[0] += step * dx / dist
                    drone.pos[1] += step * dy / dist
                    drone.range_left -= step
                    drone.total_flight_dist += step
                    if drone.range_left < config.DRONE_RANGE / 2:
                        self._release_drone(drone)

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
            # Fixed station servicing hold
            if truck.detouring_to_station and truck.station_service_timer > 0:
                truck.station_service_timer -= 1
                if truck.station_service_timer <= 0:
                    truck.fuel = config.TRUCK_RANGE
                    truck.swaps_received += 1
                    truck.detouring_to_station = False
                    truck.needs_path_update = True
                continue

            if truck.needs_path_update:
                self._plan_truck_path(truck, i)
                truck.needs_path_update = False

            if truck.path_canvas and not truck.holding:
                self._move_truck(i, truck)

    def _plan_truck_path(self, truck: Truck, truck_idx: int):
        w = self.world

        if truck.at_warehouse:
            while truck.task_index < len(truck.tasks) and \
                    truck.tasks[truck.task_index] in truck.completed:
                truck.task_index += 1
            if truck.task_index >= len(truck.tasks):
                return
            start = w.warehouse_cell
            next_task = truck.tasks[truck.task_index]
        else:
            # Plan from the cell the truck is actually in — the previous path
            # may already be cleared (task arrival) or stale (mid-route replan
            # after a drone swap), and using anything else sends the truck in
            # a straight line through walls.
            start = w.canvas_to_cell(*truck.pos)
            remaining = [t for t in truck.tasks if t not in truck.completed]
            if not remaining:
                # All assigned tasks completed: return to warehouse
                self._set_truck_path(truck, w.find_shortest_path(start, w.warehouse_cell))
                truck.detouring_to_warehouse = True
                return
            next_task = remaining[0]

        current_cell = start
        path_to_task = w.find_shortest_path(current_cell, next_task)
        if not path_to_task:
            return

        dist_to_task = self._path_length(path_to_task)
        path_task_to_depot = w.find_shortest_path(next_task, w.warehouse_cell)
        dist_task_to_depot = self._path_length(path_task_to_depot)
        safe_margin = config.SAFE_RETURN_MARGIN

        # Strategy Decisions
        if self.strategy == Strategy.DEPOT_ONLY:
            # Baseline 1: Standard Depot Return
            if truck.fuel >= dist_to_task + dist_task_to_depot + safe_margin or truck.at_warehouse:
                self._set_truck_path(truck, path_to_task)
                truck.at_warehouse = False
                truck.detouring_to_warehouse = False
            elif not truck.at_warehouse:
                self._set_truck_path(truck, w.find_shortest_path(current_cell, w.warehouse_cell))
                truck.detouring_to_warehouse = True

        elif self.strategy == Strategy.FIXED_STATION_EVRP:
            # Baseline 2: Classical E-VRP-BSS (Detour to closest static base station)
            if truck.fuel >= dist_to_task + dist_task_to_depot + safe_margin or truck.at_warehouse:
                self._set_truck_path(truck, path_to_task)
                truck.at_warehouse = False
                truck.detouring_to_station = False
                truck.detouring_to_warehouse = False
            elif not truck.at_warehouse:
                _, station_path, station_dist = self._find_nearest_station(current_cell)
                if station_path and truck.fuel >= station_dist:
                    self._set_truck_path(truck, station_path)
                    truck.detouring_to_station = True
                else:
                    self._set_truck_path(truck, w.find_shortest_path(current_cell, w.warehouse_cell))
                    truck.detouring_to_warehouse = True

        elif self.strategy == Strategy.REACTIVE_DRONE:
            # Baseline 3: Reactive Drone Swap (dispatched when fuel is critical)
            if truck.fuel >= dist_to_task + dist_task_to_depot + safe_margin or truck.at_warehouse:
                self._set_truck_path(truck, path_to_task)
                truck.at_warehouse = False
            elif truck.fuel >= dist_to_task + safe_margin:
                self._set_truck_path(truck, path_to_task)
                truck.at_warehouse = False
                if truck.fuel / config.TRUCK_RANGE <= config.REACTIVE_THRESHOLD:
                    self._dispatch_drone(truck_idx)
            elif not truck.at_warehouse:
                # Critical: try requesting drone on the spot or return to warehouse
                dispatched = self._dispatch_drone(truck_idx)
                if not dispatched:
                    self._set_truck_path(truck, w.find_shortest_path(current_cell, w.warehouse_cell))
                    truck.detouring_to_warehouse = True

        elif self.strategy == Strategy.PROACTIVE_FUEL:
            # Proposed: Predictive FUEL (Algorithm 2)
            if truck.fuel >= dist_to_task + dist_task_to_depot + safe_margin or truck.at_warehouse:
                self._set_truck_path(truck, path_to_task)
                truck.at_warehouse = False
                truck.detouring_to_warehouse = False
            elif truck.fuel + config.SWAP_BOOST_CAPACITY >= dist_to_task + dist_task_to_depot + safe_margin and \
                    truck.fuel >= dist_to_task:
                # Algorithm 2: Preemptively dispatch drone to swap while vehicle executes task
                self._dispatch_drone(truck_idx)
                self._set_truck_path(truck, path_to_task)
                truck.at_warehouse = False
            elif not truck.at_warehouse:
                self._set_truck_path(truck, w.find_shortest_path(current_cell, w.warehouse_cell))
                truck.detouring_to_warehouse = True

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
        if end_cell == self.world.warehouse_cell:
            truck.fuel = config.TRUCK_RANGE
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

            if self.strategy == Strategy.PROACTIVE_FUEL and index not in self.targeted_trucks:
                # Check if next leg requires proactive swap
                remaining = [t for t in truck.tasks if t not in truck.completed]
                if remaining:
                    p_next = self.world.find_shortest_path(end_cell, remaining[0])
                    p_home = self.world.find_shortest_path(remaining[0], self.world.warehouse_cell)
                    req = self._path_length(p_next) + self._path_length(p_home) + config.SAFE_RETURN_MARGIN
                    if truck.fuel < req:
                        self._dispatch_drone(index)

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
            self._draw_gauge(x, y + 30, 30, 5, truck.fuel / config.TRUCK_RANGE)

            if i in self.targeted_trucks:
                pulse = config.DRONE_STOP_RADIUS * (0.92 + 0.08 * math.sin(frame * 0.15))
                ring = QColor('#2980b9')
                ring.setAlpha(110)
                draw([OPERATION.circle, x, y, pulse, 2, ring])
                if truck.holding:
                    draw([OPERATION.text, x - 28, -(y + 40), 1, QColor('#c0392b'), 'HOLDING'])
            elif truck.detouring_to_station and truck.station_service_timer > 0:
                draw([OPERATION.text, x - 28, -(y + 40), 1, QColor('#27ae60'), 'SWAPPING'])

    def _draw_drones(self):
        draw = self.window.draw
        for drone in self.drones:
            x, y = drone.pos
            dx = dy = 0
            if drone.target is not None and not drone.returning:
                truck = self.trucks[drone.target]
                dx, dy = truck.pos[0] - x, truck.pos[1] - y
                link = QColor('#2980b9')
                link.setAlpha(150)
                draw([OPERATION.dotted_line, x, y, truck.pos[0], truck.pos[1], 1, link])
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
