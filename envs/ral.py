import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import random
import math
from pyamaze import maze
import tkinter as tk

import gymnasium as gym
from gymnasium import spaces

from envs.agents.UGV import UGVAgent
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
        self.df_maze = generate_df_maze(
            config["world"]["maze_size"],
            config["world"]["maze_size"],
            config["world"]["maze_loop_percentage"]
        )
        self.max_row = self.df_maze['row'].max()
        self.max_col = self.df_maze['col'].max()
        self.G = self._build_graph()
        self.cell_size = config["world"]["cell_size"]

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

        '''CREATING STATE VARIABLES'''
        # Creating Agents
        self.ugv_states = []
        # Creating Task state variables
        self.task_list = []

        '''METRICS DEFINITIONS'''
        # NEW: Track total distance traveled by EVs and drones
        self.total_ev_distance = 0.0
        self.num_tasks_completed = 0
        self.current_timestep = 0
        self.info = {
            "maze": self.df_maze,
            "base_stations": self.base_stations,
            "num_patrols": self.num_patrols,
            "patrol_positions": [],
            "patrol_colors": [],
            "patrol_paths": [],
            "R_P": [],
            "ev_distance_traveled": self.total_ev_distance,
            "latest_tasks_completed": self.num_tasks_completed
        }

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.MultiDiscrete([config["ugv"]["num_primitives"]+1]*self.num_patrols)
        # Observation space consists of:
        # - Task positions (x,y for each task, max num_tasks)
        # - UGV positions (x,y for each UGV)
        # - UGV battery levels (one value per UGV)
        # - Max range of UGVs (constant)
        
        task_space = spaces.MultiDiscrete(
            [[self.max_row+1, self.max_col+1]] * self.num_tasks
        )
        
        ugv_pos_space = spaces.MultiDiscrete(
            [[self.max_row+1, self.max_col+1]] * self.num_patrols
        )
        
        battery_space = spaces.Box(
            low=np.zeros(self.num_patrols),
            high=np.array([config['ugv']['range']]*self.num_patrols),  # assuming battery level 0-100
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'task_positions': task_space,
            'ugv_positions': ugv_pos_space,
            'battery_levels': battery_space,
        })
        
        self.reset()

    def _build_graph(self):
        G = nx.Graph()
        for _, row in self.df_maze.iterrows():
            r, c = int(row['row']), int(row['col'])
            G.add_node((r, c))
            if row['E'] == 1:
                G.add_edge((r, c), (r, c + 1))
            if row['W'] == 1:
                G.add_edge((r, c), (r, c - 1))
            if row['N'] == 1:
                G.add_edge((r, c), (r - 1, c))
            if row['S'] == 1:
                G.add_edge((r, c), (r + 1, c))
        return G

    def reset(self, seed=None, options=None):
        # 2) Reset your environment
        # - Re-initialize or reset your world to a starting condition
        # - Return the initial observation
        self.fill_task_list()
        self.ugv_states = [UGVAgent(i, self.warehouse_pos, self.max_ugv_range) for i in range(self.num_patrols)]
        self.total_ev_distance = 0.0
        self.num_tasks_completed = 0
            
        initial_obs = self._get_observation()
        self.update_info()

        return initial_obs, self.info
    
    def fill_task_list(self):
        # Fill the task list with random positions
        # 1. Spawn new tasks
        new_tasks_count = self.num_tasks - len(self.task_list)
        new_tasks_df = self.df_maze[~((self.df_maze['row']==self.warehouse_pos[1])&(self.df_maze['col']==self.warehouse_pos[0]))].sample(new_tasks_count)
        new_tasks = [tuple(row) for row in new_tasks_df[['row', 'col']].values.tolist()]
        self.current_timestep = 0
        # 2. Add new tasks to the task list
        self.task_list.extend(new_tasks)

    def step(self, action):
        # 3) Apply the chosen action
        # - Decide how the agent’s action affects the environment
        # - Step the simulation by one timeslice (or timesteps)
        self._apply_action(action)
        self.fill_task_list()
        self.current_timestep += 1

        # 4) Compute next observation, reward, done, info
        obs = self._get_observation()
        self.update_info()
        reward = self._get_reward()
        done = self._check_termination_condition()

        return obs, reward, done, False, self.info

    def render(self, mode='human'):
        # 5) (Optional) If you want a custom display or separate window
        pass

    # Helper methods below
    def _get_observation(self):
        return {
            'task_positions': np.array(self.task_list),
            'ugv_positions': np.array([ugv.position for ugv in self.ugv_states]),
            'battery_levels': np.array([ugv.current_range for ugv in self.ugv_states])
        }
    
    def update_info(self):
        self.info["patrol_positions"] = [ugv.position for ugv in self.ugv_states]
        self.info["patrol_colors"] = [ugv.agent_id for ugv in self.ugv_states]
        self.info["R_P"] = [ugv.current_range_percent for ugv in self.ugv_states]
        self.info["ev_distance_traveled"] = self.total_ev_distance
        self.info["latest_tasks_completed"] = self.num_tasks_completed
    
    def _apply_action(self, action):
        # Define how agent’s action modifies the simulation
        for i, act in enumerate(action):
            if act < self.action_space.nvec[i]:
                self.ugv_states[i].move(act, self.cell_size)
                # Update distance traveled
                self.ugv_states[i].distance_traveled += 1
                self.total_ev_distance += 1
        
    def _get_reward(self):
        # Return the immediate scalar reward
        return ...
    def _check_termination_condition(self):
        # Return True if the episode is finished (e.g., tasks done, time exceeded, etc.)
        if self.current_timestep >= self.max_timesteps:
            return True
        return False
        
