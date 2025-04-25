import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import random
import math
from pyamaze import maze
import tkinter as tk
from matplotlib import pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from envs.vehicles.UGV import UGV
# from agents.UAV import UAVAgent

def generate_df_maze(rows=20, cols=20, lp=80):
    # Create a Pyamaze maze
    m = maze(rows, cols)
    m.CreateMaze(loopPercent=lp)  # No loops to match a grid-like maze

    # List to store cell data
    maze_data = []
    for y in range(1, rows + 1):
        for x in range(1, cols + 1):
            cell = (x, y)
            cell_walls = m.maze_map[cell]  # Get wall info for each cell

            E = 1 if cell_walls['E'] else 0
            W = 1 if cell_walls['W'] else 0
            N = 1 if cell_walls['N'] else 0
            S = 1 if cell_walls['S'] else 0

            maze_data.append([f"({x},{y})", E, W, N, S])

    df = pd.DataFrame(maze_data, columns=["  cell  ", "E", "W", "N", "S"])
    if '  cell  ' in df.columns:
        df[['row', 'col']] = (
            df['  cell  '].astype(str)
            .str.replace(r'[()]', '', regex=True)
            .str.split(',', expand=True)
            .astype(int)
        )
    return df


class LMDEnv(gym.Env):
    def __init__(self, config, render_mode="human"):
        metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 4}
        super(LMDEnv, self).__init__()

        '''GUI CONFIGURATION'''
        self.render_mode = render_mode
               
        '''LOADING THE CONFIGURATION VARIABLES'''
        # Maze vars
        # self.df_maze = generate_df_maze(
        #     config["world"]["maze_size"],
        #     config["world"]["maze_size"],
        #     config["world"]["maze_loop_percentage"]
        # )
        self.df_maze = pd.read_csv(config["world"]["maze_file"])
        self.max_row = self.df_maze['row'].max()
        self.max_col = self.df_maze['col'].max()
        self.cell_size = config["world"]["cell_size"]
        self.G = self._build_graph()

        # Task vars
        self.num_tasks = config["world"]["num_tasks"]

        # Warehouse vars
        self.warehouse_opt = config["world"]["warehouse"]
        if self.warehouse_opt == "center":
            center_row = self.max_row // 2
            center_col = self.max_col // 2
            self.warehouse_pos = (int(center_row), int(center_col))
        elif self.warehouse_opt == "random":
            self.warehouse_pos = (
                random.randint(1, self.max_row),
                random.randint(1, self.max_col)
            )

        # Base station vars
        self.num_base_stations = config["world"]["num_base_stations"]
        bs_df = self.df_maze.sample(self.num_base_stations, random_state=47)[['row', 'col']]
        self.base_stations = []
        for t in bs_df.values.tolist():
            self.base_stations.append( (int(t[0]), int(t[1])) )

        # Agent vars
        self.num_patrols = config['world']['num_ugvs']
        self.num_uavs_bs = config["world"]["num_uavs_per_bs"]
        self.max_ugv_range = config['ugv']['range']
        self.drain_rate = config['ugv']['drain_rate']
        self.max_load = config['ugv']['max_load']
        self.ugv_speed = config['ugv']['speed']
        
        # Simulation vars
        self.bms = config['world']['bms']
        self.max_timesteps = config['world']['max_timesteps']
        self.traffic_b = config['world']['traffic']
        self.load_b = config['world']['load']

        # Traffic vars
        self.traffic_std_dev = config['traffic']['std_dev']  # Standard deviation for Gaussian distribution
        self.traffic_reset_dur = config['traffic']['reset_dur'] 
        self.traffic_prob = config['traffic']['prob']
        self.traffic_num_centroids = config['traffic']['num_centroids']

        '''METRICS DEFINITIONS'''
        # NEW: Track total distance traveled by EVs and drones.
        self.total_ev_distance = 0.0
        self.num_tasks_completed = 0
        self.current_timestep = 0
        self.time_elapsed = 0.0
        self.total_enegy_consumed = 0.0

        '''State vars'''
        self.ugv_states = []
        self.task_list = []
        self.task_loads = []
        self.traffic_centeroids = []
        self.red_roads = []
        self.yellow_roads = []
        self.info = {
            "maze": self.df_maze,
            "graph": self.G,
            "base_stations": self.base_stations,
            "num_patrols": self.num_patrols,
            "patrol_positions": [],
            "patrol_colors": [],
            "R_P": [],
            "ev_distance_traveled": self.total_ev_distance,
            "num_tasks_completed": self.num_tasks_completed,
            "time_elapsed":self.time_elapsed,
            "total_energy_consumed": self.total_enegy_consumed,
            "active_tasks": [],
            "red_roads": [],
            "yellow_roads": [],
        }

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([config["ugv"]["num_primitives"]]*self.num_patrols)
        
        # Fix the observation space definition:
        task_space = spaces.MultiDiscrete(
            np.array([self.max_row+1, self.max_col+1] * self.num_tasks)
        )
        ugv_pos_space = spaces.MultiDiscrete(
            np.array([self.max_row+1, self.max_col+1] * self.num_patrols)
        )
        nb_traffic_space = spaces.MultiDiscrete(
            np.array([3,3,3,3]*self.num_patrols)
        )
        battery_space = spaces.Box(
            low=0,
            high=self.max_ugv_range,
            shape=(self.num_patrols,),
            dtype=np.int32
        )
        task_load_space = spaces.Box(
            low=0,
            high=self.max_load,
            shape=(self.num_tasks,),
            dtype=np.int32)
        ugv_load_space = spaces.Box(
            low=0,
            high=self.max_load,
            shape=(self.num_patrols,),
            dtype=np.int32)
        
        self.observation_space = spaces.Dict({
            'task_positions': task_space,
            'ugv_positions': ugv_pos_space,
            'battery_levels': battery_space,
            'task_loads': task_load_space,
            'ugv_loads': ugv_load_space,
            'nb_traffic': nb_traffic_space
        })
        
        self.reset()

    def _build_graph(self):
        G = nx.Graph()
        for _, row in self.df_maze.iterrows():
            r, c = int(row['row']), int(row['col'])
            G.add_node((r, c))
            if row['E'] == 1:
                G.add_edge((r, c), (r, c+1), traffic=0)
            if row['W'] == 1:
                G.add_edge((r, c), (r, c - 1), traffic=0)
            if row['N'] == 1:
                G.add_edge((r, c), (r - 1, c), traffic=0)
            if row['S'] == 1:
                G.add_edge((r, c), (r + 1,c), traffic=0)
        return G

    def reset(self, seed=None, options=None):
        # Reset the environment to a starting condition.
        if seed:
            random.seed(seed)
        self.task_list = []
        self.task_loads = []
        self.red_roads = []
        self.yellow_roads = []
        self.traffic_centeroids = []
        for u, v in self.G.edges():
            self.G[u][v]['traffic'] = 0
        self.fill_task_list()
        if not self.load_b:
            self.task_loads = [0]*self.num_tasks
        if self.traffic_b:
            self.fill_traffic_centroids()
        self.ugv_states = [UGV(ugv_id=i, base_position=self.warehouse_pos, cell_dist=self.cell_size, max_range=self.max_ugv_range, max_load=self.max_load, drain_rate=self.drain_rate, speed=self.ugv_speed, G=self.G) for i in range(self.num_patrols)]

        self.action_response = [None]*self.num_patrols
        self.prev_task_distances = [100]*self.num_patrols
        self.last_miles = [0]*self.num_patrols
        self.prev_positions = [[] for _ in range(self.num_patrols)]
        # self.oscillation_counter = [0 for _ in range(self.num_patrols)]
        self.last_move_time = 0
        
        self.total_ev_distance = 0.0
        self.num_tasks_completed = 0
        self.current_timestep = 0
        self.time_elapsed = 0.0
        self.total_enegy_consumed = 0.0
            
        initial_obs = self._get_observation()
        self.update_info()
        return initial_obs, self.info
    
    def fill_task_list(self):
        # Fill the task list with random positions.
        new_tasks_count = self.num_tasks - len(self.task_list)
        new_tasks = random.sample(list(self.G.nodes),min(new_tasks_count, len(self.G.nodes)))
        self.task_list.extend(new_tasks)
        if self.load_b:
            new_task_loads = [random.randint(0,self.max_load) for _ in range(new_tasks_count)]
            self.task_loads.extend(new_task_loads)
    
    def fill_traffic_centroids(self):
        if random.random() < self.traffic_prob:
            self.traffic_centeroids = random.sample(list(self.G.nodes), self.traffic_num_centroids)
            # Convert traffic centroids to numpy array for easier computation
            centroid_array = np.array(self.traffic_centeroids)
            # Initialize lists for colored roads
            self.red_roads = []
            self.yellow_roads = []

            # Update traffic for each edge in the graph
            for u, v in self.G.edges():
                # Calculate midpoint of edge
                midpoint = np.array([(u[0] + v[0])/2, (u[1] + v[1])/2])
                
                # Calculate total Gaussian contribution from all centroids
                total_contrib = 0
                for centroid in centroid_array:
                    dist_sq = np.sum((midpoint - centroid)**2)
                    contrib = np.exp(-dist_sq / (2.0 * self.traffic_std_dev**2))
                    total_contrib += contrib
                
                # Normalize and set traffic level
                if total_contrib < 0.33:
                    self.G[u][v]['traffic'] = 0
                elif total_contrib < 0.66:
                    self.G[u][v]['traffic'] = 1
                    self.yellow_roads.append((u, v))
                else:
                    self.G[u][v]['traffic'] = 2
                    self.red_roads.append((u, v))
            
        else:
            self.traffic_centeroids = []
            self.red_roads = []
            self.yellow_roads = []
            for u, v in self.G.edges():
                self.G[u][v]['traffic'] = 0

    def step(self, action):
        self._apply_action(action)
        obs = self._get_observation()
        reward = self._get_reward()
        self.update_info()
        done = self._check_termination_condition()
        self.current_timestep += 1
        self.time_elapsed += self.last_move_time

        self.fill_task_list()
        if self.traffic_b:
            if self.current_timestep%self.traffic_reset_dur == 0:
                self.fill_traffic_centroids()
        self.action_response = [None]*self.num_patrols

        return obs, reward, done, False, self.info

    def _get_nb_traffic(self, position):
        """
        Get the traffic density in the North, East, South, and West directions
        from the given UGV's position using the traffic_centeroids as Gaussian centers.
        Returns a list of four discrete traffic levels (0, 1, 2) corresponding to [N, E, S, W].
        """
        # Define neighbor cell positions relative to current position.
        neighbor_positions = (
            (position[0] - 1, position[1]),
            (position[0], position[1] + 1),
            (position[0] + 1, position[1]),
            (position[0], position[1] - 1)
        )
        
        traffic_levels = [self.G.edges.get((position, neighbor), {}).get('traffic', 0) for neighbor in neighbor_positions]
        return traffic_levels


    def _get_observation(self):
        task_positions_flat = np.array(self.task_list, dtype=np.int32).flatten()
        ugv_positions_flat = np.array([ugv.position for ugv in self.ugv_states], dtype=np.int32).flatten()
        battery_flat = np.array([ugv.current_range for ugv in self.ugv_states], dtype=np.int32).flatten()
        task_loads_flat = np.array(self.task_loads, dtype=np.int32).flatten()
        ugv_loads_flat = np.array([ugv.load for ugv in self.ugv_states], dtype=np.int32).flatten()
        nb_traffic = []
        for ugv in self.ugv_states:
            traffic = self._get_nb_traffic(ugv.position)
            nb_traffic.extend(traffic)
        nb_traffic_flat = np.array(nb_traffic, dtype=np.int32).flatten()

        return {
        'task_positions': task_positions_flat,
        'ugv_positions': ugv_positions_flat,
        'battery_levels': battery_flat,
        'task_loads': task_loads_flat,
        'ugv_loads': ugv_loads_flat,
        'nb_traffic': nb_traffic_flat,
        }
            
    
    def _get_info(self):
        return self.info

    def update_info(self):
        self.info["patrol_positions"] = [ugv.position for ugv in self.ugv_states]
        self.info["patrol_colors"] = [ugv.agent_id for ugv in self.ugv_states]
        self.info["R_P"] = [ugv.current_range_percent for ugv in self.ugv_states]
        self.info["active_tasks"] = self.task_list
        self.info['red_roads'] = self.red_roads
        self.info['yellow_roads'] = self.yellow_roads

        self.info["time_elapsed"] = self.time_elapsed
        self.info["num_tasks_completed"] = self.num_tasks_completed
        self.total_ev_distance = 0.0
        self.total_energy_consumed = 0.0
        for ugv in self.ugv_states:
            self.total_ev_distance += ugv.distance_traveled
            self.total_energy_consumed += ugv.energy_consumed
        self.info["ev_distance_traveled"] = self.total_ev_distance
        self.info["total_energy_consumed"] = self.total_energy_consumed
    
    def _apply_action(self, action):
        # Define how each agent’s action modifies the simulation.
        for i, act in enumerate(action):
            if act < self.action_space.nvec[i]:
                ugv_i = self.ugv_states[i]
                self.action_response[i], self.last_move_time = ugv_i.move(act, self.G)

                if ugv_i.position == self.warehouse_pos:
                    ugv_i.recharge()

                # if len(self.prev_positions[i]) < 2:
                #     self.prev_positions[i].append(ugv_i.position)
                # else:
                #     osc_idx = self.oscillation_counter[i]%2
                #     if self.prev_positions[i][osc_idx] == ugv_i.position:
                #         self.oscillation_counter[i] += 1
                #     else:
                #         self.prev_positions[i] = []
                #         self.oscillation_counter[i] = 0
        
    def _get_reward(self):
        reward = 0.0
        
        # Task completion: reward for completing a task.
        completed_tasks = set()
        for t_i, task in enumerate(self.task_list):
            for r_i, ugv in enumerate(self.ugv_states):
                if task == ugv.position:
                    reward += 50.0
                    completed_tasks.add(task)
                    self.num_tasks_completed += 1

                    self.task_list.pop(t_i)
                    if self.load_b:
                        ugv.load = self.task_loads[t_i]
                        self.task_loads.pop(t_i)
        
        # Small time step penalty.
        reward -= self.last_move_time/self.cell_size

        # Penalize invalid moves.
        for ar in self.action_response:
            if ar == 0:
                reward -= 40.0

        # --- Progress-Based Shaping Reward ---
        bonus_factor = 5.0
        for i, ugv in enumerate(self.ugv_states):
            if self.task_list:
                distances = [abs(ugv.position[0] - task[0]) + abs(ugv.position[1] - task[1]) for task in self.task_list]
                # distances = [nx.shortest_path_length(self.G, source=ugv.position, target=task) for task in self.task_list]
                current_min_distance = min(distances)
                if current_min_distance < self.prev_task_distances[i]:
                    reward += bonus_factor * (self.prev_task_distances[i] - current_min_distance)
                    if current_min_distance == 0:
                        self.prev_task_distances[i] = 100  # Reset if task is completed.
                    else:
                        self.prev_task_distances[i] = current_min_distance

        # distance_factor = 1.0
        # for i, ugv in enumerate(self.ugv_states):
        #     d = abs(ugv.distance_traveled-self.last_miles[i])
        #     self.last_miles[i] = ugv.distance_traveled
        #     reward -= distance_factor*d

        # # --- Oscillation Penalty ---
        # # Penalize if a UGV oscillates between two positions.
        # oscillation_penalty = 0.0
        # osc_penalty_value = 10.0  # Penalty per oscillatory instance.
        # for i, ugv in enumerate(self.ugv_states):
        #     oscillation_penalty += self.oscillation_counter[i] * osc_penalty_value
        # reward -= oscillation_penalty

        return reward

    def _check_termination_condition(self):
        # Terminate if maximum timesteps are reached.
        if self.current_timestep >= self.max_timesteps:
            return True
        return False
