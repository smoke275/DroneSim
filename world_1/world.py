import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import random
import math
from pyamaze import maze

# Helper distance function
def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_df_maze(rows=20, cols=20, lp=80):
    # Create a Pyamaze maze
    m = maze(rows, cols)
    m.CreateMaze(loopPercent=lp)  # No loops to match a grid-like maze

    # List to store cell data
    maze_data = []

    # Loop through each cell in the maze
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            cell = (x, y)
            cell_walls = m.maze_map[cell]  # Get wall info for each cell

            # Convert to E, W, N, S format (0 for wall, 1 for no wall)
            E = 1 if cell_walls['E'] else 0
            W = 1 if cell_walls['W'] else 0
            N = 1 if cell_walls['N'] else 0
            S = 1 if cell_walls['S'] else 0

            # Store cell information
            maze_data.append([f"({x},{y})", E, W, N, S])

    # Create DataFrame
    df = pd.DataFrame(maze_data, columns=["  cell  ", "E", "W", "N", "S"])
    return df

class World:
    def __init__(self, config):
        """
        Initializes the World with a maze loaded from CSV and basic parameters.
        This replicates the core EV (truck) logic from the old code, but omits drone logic.
        """

        # Load the maze CSV as a DataFrame
        self.df_maze = generate_df_maze(config["world"]["maze_size"], config["world"]["maze_size"], config["world"]["maze_loop_percentage"])
        # Ensure row/col are parsed from the 'cell' column if needed
        if '  cell  ' in self.df_maze.columns:
            self.df_maze[['row', 'col']] = (
                self.df_maze['  cell  '].astype(str)
                .str.replace(r'[()]', '', regex=True)
                .str.split(',', expand=True)
                .astype(int)
            )
        # Maze dimensions
        self.max_row = self.df_maze['row'].max()
        self.max_col = self.df_maze['col'].max()
        # Build a graph from the maze for pathfinding
        self.G = self._build_graph()

        # Generate tasks and cluster them
        self.num_tasks = config["world"]["num_tasks"]
        self.num_patrols = config['world']['num_evs']
        # Fuel / range parameters
        self.R = config["ev"]["range"] # default 3000 if not specified
        self.R_D = [self.R] * self.num_patrols  # each EV's current fuel/battery
        self.R_P = [i/self.R for i in self.R_D]
        self.num_uavs = config["world"]["num_uavs_per_bs"]
        self.num_base_stations = config["world"]["num_base_stations"]
        self.B = config["uav"]["range"]
        self.drone_speed = config["uav"]["speed"]

        bs_df = self.df_maze.sample(self.num_base_stations, random_state=47)[['row', 'col']]
        bs_list = []
        for t in bs_df.values.tolist():  # e.g. t = [np.int64(15), np.int64(13)]
            bs_list.append( (int(t[0]), int(t[1])) )  # cast to (int, int)
        self.base_stations = bs_list

        # Center cell for warehouse
        self.center_row = self.max_row // 2
        self.center_col = self.max_col // 2
        self.warehouse_pos = (int(self.center_row), int(self.center_col))

        self.default_colors = ["red", "blue", "green", "yellow", "magenta", "cyan", "orange", "black"]

        self.world_state = {}
    
    def _build_graph(self):
        """
        Builds a graph from the maze DataFrame. Each cell is a node, and edges
        exist if E/W/N/S = 1 indicates connectivity in that direction.
        """
        G = nx.Graph()
        for _, row in self.df_maze.iterrows():
            r, c = int(row['row']), int(row['col'])
            G.add_node((r, c))
            # East
            if row['E'] == 1:
                G.add_edge((r, c), (r, c + 1))
            # West
            if row['W'] == 1:
                G.add_edge((r, c), (r, c - 1))
            # North
            if row['N'] == 1:
                G.add_edge((r, c), (r - 1, c))
            # South
            if row['S'] == 1:
                G.add_edge((r, c), (r + 1, c))
        return G

    def _update_world_state(self):
        # This dictionary is updated each time so the console can read it
        self.world_state = {
            "num_patrols": self.num_patrols,
            "patrol_positions": self.patrol_positions,
            "patrol_colors": self.patrol_colors,
            "patrol_paths": self.patrol_paths,
            "R_P": self.R_P,
            "active_tasks": self.active_tasks,
            "drone_positions": self.drone_positions,
            "drone_status": self.drone_status,
            "completed_tasks": self.completed_tasks,
            "tasks_completed_flag": self.tasks_completed_flag
        }

    def initialize(self, edge_weight):
        """
        1. Creates random tasks (between self.num_tasks and self.max_random_tasks).
        2. Clusters them with k-means into self.num_patrols groups.
        3. Assigns tasks to each EV and sets all EV positions to the warehouse center.
        4. Initializes EV path states.
        """

        self.edge_weight = edge_weight
        
        # # Determine how many tasks to pick
        # # tasks_df = self.df_maze.sample(self.num_tasks, random_state=47)[['row', 'col']]
        # # task_list = []
        # # for t in tasks_df.values.tolist():  # e.g. t = [np.int64(15), np.int64(13)]
        # #     task_list.append( (int(t[0]), int(t[1])) )  # cast to (int, int)
        # # self.all_tasks = task_list
        # # self.all_tasks = [
        # #     (10, 8), (10, 6), (10, 4),  # North
        # #     (10, 12), (10, 14), (10, 16),  # South
        # #     (8, 10), (6, 10), (4, 10),  # West
        # #     (12, 10), (14, 10), (16, 10)  # East
        # # ]
        # self.all_tasks = [
        #     (15,10),(15,15),
        # ]

        # # Cluster tasks
        # # kmeans = KMeans(n_clusters=self.num_patrols, random_state=47).fit(self.all_tasks)
        # # task_clusters = kmeans.labels_
        # # task_clusters = [
        # #     0,0,0,
        # #     1,1,1,
        # #     2,2,2,
        # #     3,3,3
        # # ]
        # task_clusters = [0,0]
        # print("All tasks", self.all_tasks)
        # print("-----------------------------------------\n\n")

        # Determine how many tasks to pick
        tasks_df = self.df_maze.sample(self.num_tasks)[['row', 'col']]
        task_list = []
        for t in tasks_df.values.tolist():  # e.g. t = [np.int64(15), np.int64(13)]
            task_list.append( (int(t[0]), int(t[1])) )  # cast to (int, int)
        self.all_tasks = task_list
        # Cluster tasks
        kmeans = KMeans(n_clusters=self.num_patrols).fit(self.all_tasks)
        task_clusters = kmeans.labels_
        # print("All tasks", self.all_tasks)
        # print("-----------------------------------------\n\n")

        # Clear and assign tasks to each patrol
        self.completed_tasks = [set() for _ in range(self.num_patrols)]
 
        self.patrol_positions = [self.warehouse_pos for _ in range(self.num_patrols)]
        # self.patrol_colors = self.default_colors[:self.num_patrols]
        self.patrol_colors = [i for i in range(self.num_patrols)]
        self.patrol_tasks = [[] for _ in range(self.num_patrols)]
        for i, t in enumerate(self.all_tasks):
            cluster_idx = task_clusters[i]
            self.patrol_tasks[cluster_idx].append(t)  # store in row,col form
        self.active_tasks = []
        self.R_D = [self.R] * self.num_patrols  # reset fuel/battery
        self.R_P = [i/self.R for i in self.R_D]
        self.patrol_paths = [[] for _ in range(self.num_patrols)]  # row,col form
        self.backcost=[[] for _ in range(self.num_patrols)]
        for i in range(self.num_patrols):
            current_pos = self.patrol_positions[i]  # (row, col) of the EV's current location
            
            closest_dist = float('inf')
            closest_task = None
            
            # Find the closest task by comparing the shortest path length in the graph
            for task in self.patrol_tasks[i]:
                # Convert the task (row, col) into a node the graph recognizes, e.g. (row, col)
                # Then get the path length from current_pos to task using BFS or shortest_path_length
                try:
                    dist = nx.shortest_path_length(self.G, source=current_pos, target=(task[0], task[1]))
                    self.backcost[i].append(dist)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_task = (task[0], task[1])
                except nx.NetworkXNoPath:
                    # If there's no path to that task, skip it
                    continue
            self.active_tasks.append(closest_task)

            # Now check if the distance * edge_weight is within this EV's range
            if closest_task is not None and 2*(closest_dist * edge_weight) <= self.R:
                # Retrieve the actual path
                self.patrol_paths[i] = nx.shortest_path(self.G, source=current_pos, target=closest_task)[1:]
            else:
                # No reachable task, or out of range, so keep it empty
                self.patrol_paths[i] = []
        self.drone_positions = []
        self.drone_bases = []
        self.drone_status=[]
        self.drone_battery = []
        self.drone_targets = []
        for i in range(self.num_base_stations):
            for _ in range(self.num_uavs):
                x,y = self.base_stations[i]
                self.drone_positions.append([x, y])  # Use a list instead of a tuple
                self.drone_bases.append(i)
                self.drone_status.append(0)
                self.drone_battery.append(self.B)
                self.drone_targets.append(-1)
        self.recharge_request = [0 for _ in range(self.num_patrols)]

        self.tasks_completed_flag = False
        self.completed_tasks = 0
        self.world_state["tasks_completed_flag"] = self.tasks_completed_flag

        # Update the initial world_state
        self._update_world_state()
        print("-----------------------------------------\n\n")
        print("Clusters", self.patrol_tasks)
        self.print_world_state()
        print("-----------------------------------------\n\n")

    def simulate(self, timesteps=1):
        for _ in range(timesteps):
            self.lmd_simulate()
            # self.bms_simulate()
            if not self.tasks_completed_flag:
                check = True
                for i in self.patrol_paths:
                    if len(i) !=0:
                        check = False
                        break
                
                if check:
                    self.tasks_completed_flag = True
            self.world_state["tasks_completed_flag"] = self.tasks_completed_flag

            print("-----------------------------------------\n\n")
            print("Clusters", self.patrol_tasks)
            self.print_world_state()
            print("-----------------------------------------\n\n")

    def bms_simulate(self):
        """
        Extended battery management for drones:
        1) Assign idle drones to EVs that request recharge (recharge_request[i] == 1).
        - Check if drone has enough battery for a full round trip (drone->EV->droneBase).
            If not, skip it.
        2) While traveling to EV (status=1), keep checking if the drone
        still has enough battery for the remainder of the trip + return.
        If it no longer does, switch to returning (status=2) mid-flight.
        3) When a drone arrives at the EV, that EV is fully recharged,
        and the drone goes into returning state.
        4) When a drone arrives at base, it fully recharges and becomes idle.
        """

        DRONE_SPEED = self.drone_speed  # Speed in "cells" (or units) per sim step

        # 1) Assign idle drones to EVs that request recharge
        for ev_idx in range(self.num_patrols):
            if self.recharge_request[ev_idx] == 1 and (not ev_idx in self.drone_targets):
                ev_pos = self.patrol_positions[ev_idx]

                chosen_drone = None
                best_dist = float('inf')

                for d_idx in range(len(self.drone_positions)):
                    if self.drone_status[d_idx] == 0 and self.drone_battery[d_idx] > 0:
                        drone_pos = self.drone_positions[d_idx]
                        base_idx = self.drone_bases[d_idx]
                        base_pos = self.base_stations[base_idx]

                        # Distances for round-trip check
                        dist_to_ev = _distance(drone_pos, ev_pos)
                        dist_ev_to_base = _distance(ev_pos, base_pos)
                        needed_battery = dist_to_ev + dist_ev_to_base

                        # Only consider drones with enough battery for a complete round trip
                        if self.drone_battery[d_idx] >= needed_battery:
                            # Among all valid drones, pick the closest to the EV
                            if dist_to_ev < best_dist:
                                best_dist = dist_to_ev
                                chosen_drone = d_idx

                # If we found a drone that can do round-trip, assign it
                if chosen_drone is not None:
                    self.drone_status[chosen_drone] = 1  # traveling to EV
                    self.drone_targets[chosen_drone] = ev_idx

                # Reset the EV's request flag
                self.recharge_request[ev_idx] = 0

        # 2) Move each drone according to its state
        for d_idx in range(len(self.drone_positions)):
            status = self.drone_status[d_idx]
            if status == 0:  
                # Idle at base, do nothing
                continue

            elif status == 1:
                # Traveling to EV
                ev_idx = self.drone_targets[d_idx]
                if ev_idx < 0 or ev_idx >= len(self.patrol_positions):
                    # Invalid EV => reset to idle
                    self.drone_status[d_idx] = 0
                    self.drone_targets[d_idx] = -1
                    continue

                ev_pos = self.patrol_positions[ev_idx]
                drone_pos = self.drone_positions[d_idx]
                base_idx = self.drone_bases[d_idx]
                base_pos = self.base_stations[base_idx]

                dist_to_ev = _distance(drone_pos, ev_pos)
                dist_ev_to_base = _distance(ev_pos, base_pos)

                # (A) Check mid-flight if drone can still complete the trip + return
                #     from its CURRENT position. So the battery needed is:
                #     distance to EV (dist_to_ev) + distance from EV to base (dist_ev_to_base).
                #     If not enough battery, switch to returning.
                if self.drone_battery[d_idx] < (dist_to_ev + dist_ev_to_base):
                    # Not enough battery for a full round trip => bail out
                    self.drone_status[d_idx] = 2
                    self.drone_targets[d_idx] = -1
                    continue
                
                # (B) If we are very close, we consider ourselves "arrived"
                if dist_to_ev <= 1:
                    # Arrived at EV => recharge EV, start returning
                    self.R_D[ev_idx] = self.R
                    self.R_P[ev_idx] = 1
                    self.drone_status[d_idx] = 2  # go back to base
                    self.drone_targets[d_idx] = -1
                    self.recharge_request[ev_idx] = 0
                else:
                    # Move closer to EV by DRONE_SPEED
                    step = min(dist_to_ev, DRONE_SPEED)
                    dx = ev_pos[0] - drone_pos[0]
                    dy = ev_pos[1] - drone_pos[1]
                    norm = math.sqrt(dx * dx + dy * dy)
                    if norm > 0:
                        self.drone_positions[d_idx][0] += (dx / norm) * step
                        self.drone_positions[d_idx][1] += (dy / norm) * step
                        # Subtract battery usage
                        self.drone_battery[d_idx] -= step
                        if self.drone_battery[d_idx] < 0:
                            self.drone_battery[d_idx] = 0
                            # Force return
                            self.drone_status[d_idx] = 2
                            self.drone_targets[d_idx] = -1

            elif status == 2:
                # Returning to base
                base_idx = self.drone_bases[d_idx]
                base_pos = self.base_stations[base_idx]
                drone_pos = self.drone_positions[d_idx]
                dist_to_base = _distance(drone_pos, base_pos)

                if dist_to_base < 1e-3:
                    # Arrived => fully recharge, go idle
                    self.drone_positions[d_idx] = [base_pos[0], base_pos[1]]
                    self.drone_battery[d_idx] = self.B
                    self.drone_status[d_idx] = 0
                    self.drone_targets[d_idx] = -1
                else:
                    step = min(dist_to_base, DRONE_SPEED)
                    dx = base_pos[0] - drone_pos[0]
                    dy = base_pos[1] - drone_pos[1]
                    norm = math.sqrt(dx*dx + dy*dy)
                    if norm > 0:
                        self.drone_positions[d_idx][0] += (dx / norm) * step
                        self.drone_positions[d_idx][1] += (dy / norm) * step
                        self.drone_battery[d_idx] -= step
                        if self.drone_battery[d_idx] < 0:
                            self.drone_battery[d_idx] = 0

        # After updating all drones, reflect new statuses/positions
        self._update_world_state()



    def lmd_simulate(self):
        """
        Steps the simulation for 'timesteps' frames. Each frame:
        - For each EV: if it has a path and battery remaining, move one BFS step.
        - Reduce that EV's battery by `edge_weight`.
        - If the path is exhausted, we consider that task reached:
            * Remove it from patrol_tasks[i].
            * Attempt to find a new task. If none or out-of-range, try to go to warehouse.
        - If both new tasks and warehouse are unreachable, the EV stands idle.

        :param timesteps: Number of simulation steps to run
        :param edge_weight: Cost per BFS hop (e.g., cell-to-cell movement cost)
        """
        edge_weight = self.edge_weight
        # for _ in range(timesteps):
        if True:
            for i in range(self.num_patrols):
                # If there's no path or the EV is out of battery, skip
                if not self.patrol_paths[i] or self.R_D[i] <= 0:
                    continue

                # Move one BFS step (pop the next cell from the path)
                next_node = self.patrol_paths[i].pop(0)  # e.g. (row, col)
                self.patrol_positions[i] = next_node

                # Reduce battery by 'edge_weight' per BFS hop
                self.R_D[i] -= edge_weight
                if self.R_D[i] < 0:
                    self.R_D[i] = 0  # clamp battery to 0

                # Update battery fraction
                self.R_P[i] = self.R_D[i] / self.R

                # If the path is now empty, we've reached the target
                if len(self.patrol_paths[i]) == 0:
                    final_task = self.active_tasks[i]

                    if final_task is not None:
                        # Remove final_task from patrol_tasks
                        if final_task in self.patrol_tasks[i]:
                            self.completed_tasks += 1
                            final_task_idx = self.patrol_tasks[i].index(final_task)
                            self.patrol_tasks[i].remove(final_task)
                            self.backcost[i].pop(final_task_idx)
                            if random.random()<0.9:
                            # if True:
                                self.recharge_request[i] = 1
                        
                        elif final_task==self.warehouse_pos:
                            self.R_D[i] = self.R
                            self.R_P[i] = 1

                        # Mark it as completed
                        self.active_tasks[i] = None

                        # Attempt to find a NEW nearest task
                        closest_dist = float('inf')
                        closest_task = None
                        current_pos = self.patrol_positions[i]
                        closest_task_idx = None

                        for idx, task in enumerate(self.patrol_tasks[i]):
                            try:
                                dist = nx.shortest_path_length(
                                    self.G, source=current_pos, target=(task[0], task[1])
                                )
                                if dist < closest_dist:
                                    closest_dist = dist
                                    closest_task = (task[0], task[1])
                                    closest_task_idx = idx
                            except nx.NetworkXNoPath:
                                continue

                        # If we found a new reachable task
                        if closest_task is not None and (closest_dist * edge_weight)+(self.backcost[i][closest_task_idx]*edge_weight) <= self.R_D[i]:
                            self.patrol_paths[i] = nx.shortest_path(
                                self.G, source=current_pos, target=closest_task
                            )[1:]
                            self.active_tasks[i] = closest_task
                            continue

                        # Otherwise, try to route to the warehouse
                        try:
                            dist_to_wh = nx.shortest_path_length(
                                self.G, source=current_pos, target=self.warehouse_pos
                            )
                            if (dist_to_wh * edge_weight) <= self.R_D[i]:
                                self.patrol_paths[i] = nx.shortest_path(
                                    self.G, source=current_pos, target=self.warehouse_pos
                                )[1:]
                                self.active_tasks[i] = self.warehouse_pos
                            else:
                                # No tasks or warehouse in range
                                self.patrol_paths[i] = []
                        except nx.NetworkXNoPath:
                            # Can't even reach the warehouse
                            self.patrol_paths[i] = []

            # After all EVs move once, update the world_state for visualization
            self._update_world_state()

    def get_world_state(self):
        """
        Returns a dictionary describing the current world state (positions, tasks, fuel levels, etc.).
        This is used by the console code to draw the simulation.
        """
        return self.world_state
    
    def print_world_state(self):
        """
        Prints the current world state for debugging purposes, including patrol (EV) and drone information,
        plus info about recharge requests, assigned EVs for each drone, distance to assigned EV,
        and the number of tasks completed vs. tasks remaining.
        """

        def _distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        print("\n===== WORLD STATE =====")
        print(f"Number of Patrols (EVs): {self.world_state['num_patrols']}")
        print(f"Tasks completed flag: {self.world_state["tasks_completed_flag"]}")

        # Display tasks completed and tasks left
        # Completed tasks count is stored in self.completed_tasks.
        # Tasks left is computed by summing the remaining tasks in each patrol's list.
        remaining_tasks = sum(len(tasks) for tasks in self.patrol_tasks)
        print(f"Total Tasks Completed: {self.completed_tasks}")
        print(f"Total Tasks Left: {remaining_tasks}")

        # 1. Show which EVs are requesting recharge
        requesting_evs = [i+1 for i, req in enumerate(self.recharge_request) if req == 1]
        if requesting_evs:
            print(f"Recharge Requests: EV(s) {requesting_evs}")
        else:
            print("No active recharge requests.")

        # 2. Patrol EV information
        print("\n-- Patrol Positions and Status --")
        for i, pos in enumerate(self.world_state['patrol_positions']):
            battery_percentage = self.world_state['R_P'][i] * 100
            active_task = self.world_state['active_tasks'][i] if self.world_state['active_tasks'][i] else "None"
            is_requesting = " (Requesting Recharge)" if self.recharge_request[i] == 1 else ""
            print(f"  EV {i+1}: Position = {pos}, Battery = {battery_percentage:.2f}%, "
                f"Active Task = {active_task}{is_requesting}")

        print("\n-- Patrol Paths --")
        for i, path in enumerate(self.world_state['patrol_paths']):
            path_str = " -> ".join([f"({x[0]},{x[1]})" for x in path]) if path else "No Path"
            print(f"  EV {i+1}: Path = {path_str}")

        # 3. Drone information
        print("\n-- Drone Positions and Status --")
        for d_idx, (pos, status, battery) in enumerate(zip(
            self.world_state['drone_positions'],
            self.world_state['drone_status'],
            self.drone_battery
        )):
            # Determine status string
            if status == 0:
                status_str = "Idle"
            elif status == 1:
                status_str = "Traveling to EV"
            else:
                status_str = "Returning to Base"

            # Check if assigned to an EV
            assigned_ev_idx = self.drone_targets[d_idx]
            assigned_ev_str = "None"
            dist_str = ""
            if assigned_ev_idx != -1:
                assigned_ev_str = f"EV {assigned_ev_idx+1}"
                # If traveling to EV, compute distance
                if status == 1:
                    ev_pos = self.world_state['patrol_positions'][assigned_ev_idx]
                    dist_to_ev = _distance(pos, ev_pos)
                    dist_str = f", Dist to EV: {dist_to_ev:.2f}"
            print(f"  Drone {d_idx+1}: Position = {pos}, Status = {status_str}, "
                f"Assigned EV = {assigned_ev_str}, Battery = {battery:.2f}/{self.B}{dist_str}")

        print("======================\n")




