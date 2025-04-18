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
        
        # Simulation vars
        self.bms = config['world']['bms']
        self.max_timesteps = config['world']['max_timesteps']
        self.traffic_b = config['world']['traffic']

        # Traffic vars
        self.traffic_std_dev = config['traffic']['std_dev']  # Standard deviation for Gaussian distribution
        self.traffic_reset_dur = config['traffic']['reset_dur'] 
        self.traffic_prob = config['traffic']['prob']
        self.traffic_num_centroids = config['traffic']['num_centroids']

        # Misc vars
        self.action_response = [None]*self.num_patrols
        self.prev_task_distances = [float('inf')]*self.num_patrols
        # NEW: Initialize position history for oscillation detection.
        self.prev_positions = [[] for _ in range(self.num_patrols)]
        self.oscillation_counter = [0 for _ in range(self.num_patrols)]


        '''METRICS DEFINITIONS'''
        # NEW: Track total distance traveled by EVs and drones.
        self.total_ev_distance = 0.0
        self.num_tasks_completed = 0
        self.current_timestep = 0

        '''State vars'''
        self.ugv_states = []
        self.task_list = []
        self.traffic_centeroids = []
        self.red_cells = []
        self.yellow_cells = []
        self.info = {
            "maze": self.df_maze,
            "graph": self.G,
            "base_stations": self.base_stations,
            "num_patrols": self.num_patrols,
            "patrol_positions": [],
            "patrol_colors": [],
            "patrol_paths": [],
            "R_P": [],
            "ev_distance_traveled": self.total_ev_distance,
            "num_tasks_completed": self.num_tasks_completed,
            "active_tasks": [],
            "red_cells": [],
            "yellow_cells": [],
        }

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([config["ugv"]["num_primitives"]+1]*self.num_patrols)
        
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
        if self.traffic_b:        
            self.observation_space = spaces.Dict({
                'task_positions': task_space,
                'ugv_positions': ugv_pos_space,
                'nb_traffic': nb_traffic_space
            })
        else:
            self.observation_space = spaces.Dict({
                'task_positions': task_space,
                'ugv_positions': ugv_pos_space,
            })
        
        self.reset()

    def _build_graph(self):
        G = nx.Graph()
        for _, row in self.df_maze.iterrows():
            r, c = int(row['row']), int(row['col'])
            G.add_node((r, c))
            if row['E'] == 1:
                G.add_edge((r, c), (r, c+1))
            if row['W'] == 1:
                G.add_edge((r, c), (r, c - 1))
            if row['N'] == 1:
                G.add_edge((r, c), (r - 1, c))
            if row['S'] == 1:
                G.add_edge((r, c), (r + 1,c))
        return G

    def reset(self, seed=None, options=None):
        # Reset the environment to a starting condition.
        self.fill_task_list(seed=seed)
        if self.traffic_b:
            self.fill_traffic_centroids(seed=seed)
        self.ugv_states = [UGV(i, self.warehouse_pos, self.max_ugv_range) for i in range(self.num_patrols)]
        self.action_response = [None]*self.num_patrols
        self.prev_task_distances = [100]*self.num_patrols
        self.prev_positions = [[] for _ in range(self.num_patrols)]
        self.oscillation_counter = [0 for _ in range(self.num_patrols)]
        
        self.total_ev_distance = 0.0
        self.num_tasks_completed = 0
        self.current_timestep = 0
            
        initial_obs = self._get_observation()
        self.update_info()
        return initial_obs, self.info
    
    def fill_task_list(self, seed=None):
        # Fill the task list with random positions.
        new_tasks_count = self.num_tasks - len(self.task_list)
        new_tasks_df = self.df_maze[~((self.df_maze['row']==self.warehouse_pos[1])&(self.df_maze['col']==self.warehouse_pos[0]))].sample(new_tasks_count, random_state=seed)
        new_tasks = [tuple(row) for row in new_tasks_df[['row', 'col']].values.tolist()]
        self.task_list.extend(new_tasks)
    
    def fill_traffic_centroids(self, seed=None):
        # randomly select two cells from the maze
        if random.random() < self.traffic_prob:
            self.traffic_centeroids = self.df_maze.sample(self.traffic_num_centroids, random_state=seed)[['row', 'col']]
            self.red_cells = []
            self.yellow_cells = []         
            # Calculate traffic levels for all cells
            for _, row in self.df_maze.iterrows():
                pos = (row['row'], row['col'])
                traffic = self._get_cell_traffic(pos)
                if traffic == 2:
                    self.red_cells.append(pos)
                elif traffic == 1:
                    self.yellow_cells.append(pos)
        else:
            self.traffic_centeroids = []
            self.red_cells = []
            self.yellow_cells = []  

    def step(self, action):
        # Apply the chosen action.
        self._apply_action(action)
        self.current_timestep += 1

        obs = self._get_observation()
        reward = self._get_reward()
        self.update_info()
        done = self._check_termination_condition()
        self.fill_task_list()
        if self.traffic_b:
            if self.current_timestep%self.traffic_reset_dur == 0:
                self.fill_traffic_centroids()
        self.action_response = [None]*self.num_patrols

        return obs, reward, done, False, self.info

    def render(self, mode='human'):
        # (Optional) Custom display.
        pass

    def _get_cell_traffic(self, position):
        """
        Get the traffic density at the given UGV's position using the traffic_centeroids as Gaussian centers.
        Returns a list of four discrete traffic levels (0, 1, 2) corresponding to [N, E, S, W].
        """
        # Convert the DataFrame to a list of (row, col) tuples.
        centroids = [(row['row'], row['col']) for _, row in self.traffic_centeroids.iterrows()]
        
        total_contrib = 0.0
        for (cx, cy) in centroids:
            dist_sq = (position[0] - cx)**2 + (position[1] - cy)**2
            contrib = math.exp(-dist_sq / (2.0 * (self.std_dev**2)))
            total_contrib += contrib
        
        # Find the traffic level based on the total contribution.
        if total_contrib < 0.33:
            level = 0
        elif total_contrib < 0.66:
            level = 1
        else:
            level = 2
        
        return level

    def _get_nb_traffic(self, position):
        """
        Get the traffic density in the North, East, South, and West directions
        from the given UGV's position using the traffic_centeroids as Gaussian centers.
        Returns a list of four discrete traffic levels (0, 1, 2) corresponding to [N, E, S, W].
        """
        # Define neighbor cell positions relative to current position.
        neighbor_positions = {
            'N': (position[0] - 1, position[1]),
            'E': (position[0], position[1] + 1),
            'S': (position[0] + 1, position[1]),
            'W': (position[0], position[1] - 1)
        }
        
        # If no traffic centroids have been defined, return zero traffic in all directions.
        if (not isinstance(self.traffic_centeroids, pd.DataFrame)) or self.traffic_centeroids.empty:
            return [0, 0, 0, 0]
        
        # Convert the DataFrame to a list of (row, col) tuples.
        centroids = [(row['row'], row['col']) for _, row in self.traffic_centeroids.iterrows()]
        
        traffic_levels = []
        for direction in ['N', 'E', 'S', 'W']:
            nb_pos = neighbor_positions[direction]
            # If the neighbor position is outside the maze boundaries, treat it as having zero traffic.
            if nb_pos[0] < 1 or nb_pos[0] > self.max_row or nb_pos[1] < 1 or nb_pos[1] > self.max_col:
                traffic_levels.append(0)
                continue
            
            total_contrib = 0.0
            # Compute the Gaussian contribution from each centroid for the neighbor cell.
            for (cx, cy) in centroids:
                dist_sq = (nb_pos[0] - cx)**2 + (nb_pos[1] - cy)**2
                contrib = math.exp(-dist_sq / (2.0 * (self.std_dev**2)))
                total_contrib += contrib
            
            # Average the contribution over all centroids.
            avg_contrib = total_contrib / len(centroids)
            
            # Convert the continuous value into a discrete level.
            if total_contrib < 0.33:
                level = 0
            elif total_contrib < 0.66:
                level = 1
            else:
                level = 2
            
            traffic_levels.append(level)
        
        return traffic_levels


    def _get_observation(self):
        task_positions_flat = np.array(self.task_list, dtype=np.int32).flatten()
        ugv_positions_flat = np.array([ugv.position for ugv in self.ugv_states], dtype=np.int32).flatten()

        if self.traffic_b:
            nb_traffic = []
            for ugv in self.ugv_states:
                traffic = self._get_nb_traffic(ugv.position)
                nb_traffic.extend(traffic)
            nb_traffic_flat = np.array(nb_traffic, dtype=np.int32).flatten()

            return {
                'task_positions': task_positions_flat,
                'ugv_positions': ugv_positions_flat,
                'nb_traffic': nb_traffic_flat,
            }
        else:
            return {
            'task_positions': task_positions_flat,
            'ugv_positions': ugv_positions_flat,
            }
            
    
    def _get_info(self):
        return self.info

    def update_info(self):
        self.info["patrol_positions"] = [ugv.position for ugv in self.ugv_states]
        self.info["patrol_colors"] = [ugv.agent_id for ugv in self.ugv_states]
        self.info["R_P"] = [ugv.current_range_percent for ugv in self.ugv_states]
        self.info["active_tasks"] = self.task_list
        self.info["num_tasks_completed"] = self.num_tasks_completed
        self.info['red_cells'] = self.red_cells
        self.info['yellow_cells'] = self.yellow_cells

        self.total_ev_distance = 0.0
        for ugv in self.ugv_states:
            self.total_ev_distance += ugv.distance_traveled

        self.info["ev_distance_traveled"] = self.total_ev_distance
    
    def _apply_action(self, action):
        # Define how each agent’s action modifies the simulation.
        for i, act in enumerate(action):
            if act < self.action_space.nvec[i]:
                ugv_i = self.ugv_states[i]
                check_move = ugv_i.move(act, self.cell_size)
                if check_move:
                    if act == 4 or (ugv_i.tmp_position in self.G.nodes and (ugv_i.tmp_position, ugv_i.position) in self.G.edges):
                        ugv_i.move_approved = True
                        check_move = True
                    else:
                        ugv_i.move_approved = False
                        check_move = False
                ugv_i.update_move(self.cell_size)
                self.action_response[i] = check_move

                if len(self.prev_positions[i]) < 2:
                    self.prev_positions[i].append(ugv_i.position)
                else:
                    osc_idx = self.oscillation_counter[i]%2
                    if self.prev_positions[i][osc_idx] == ugv_i.position:
                        self.oscillation_counter[i] += 1
                    else:
                        self.prev_positions[i] = []
                        self.oscillation_counter[i] = 0
        
    def _get_reward(self):
        reward = 0.0
        
        # Task completion: reward for completing a task.
        completed_tasks = set()
        for task in self.task_list:
            for ugv in self.ugv_states:
                if task == ugv.position:
                    reward += 50.0
                    completed_tasks.add(task)
                    self.num_tasks_completed += 1                       
        for task in completed_tasks:
            self.task_list.remove(task)
        
        # Small step penalty.
        reward -= 1.0

        # Penalize invalid moves.
        for ar in self.action_response:
            if not ar:
                reward -= 40.0

        # --- Progress-Based Shaping Reward ---
        bonus_factor = 2.0
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
