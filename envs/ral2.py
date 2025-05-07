import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import random
from pyamaze import maze
import tkinter as tk
tk.Tk.state = lambda self, s=None: self.wm_state('normal' if s == 'zoomed' else s)
import math
import pygame # Added pygame
import os # Added os for path joining
import time

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


# Define colors for pygame
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
DARK_GREEN = (0, 100, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Map agent IDs to colors (example)
AGENT_COLORS = [
    (50, 50, 255),    # Blue-ish
    (255, 50, 50),    # Red-ish
    (50, 255, 50),    # Green-ish
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
]

def get_agent_color(agent_id):
    """Returns a distinct color for a given agent ID."""
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


class LMDEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 30} # Adjusted FPS

    def __init__(self, config, render_mode=None): # Default render_mode is None
        super(LMDEnv, self).__init__()

        '''GUI CONFIGURATION'''
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.cell_size_px = 150 #config["world"].get("cell_size", 30) # Pixel size for rendering cells
        self.screen_width = None
        self.screen_height = None
        self.truck_image = None
        self.warehouse_image = None # Optional: image for warehouse
        self.task_image = None      # Optional: image for tasks
        self.bs_image = None        # Optional: image for base stations
        self.escape_pressed = False # Flag to track if escape was pressed

        '''LOADING THE CONFIGURATION VARIABLES'''
        self.max_row = config["world"]["maze_size"]
        self.max_col = config["world"]["maze_size"]
        self.maze_lp = config["world"]["maze_loop_percentage"]
        self.cell_size = config["world"]["cell_size"] # Physical cell size

        # Task vars
        self.num_tasks = config["world"]["num_tasks"]
        self.task_prob = config["world"]["task_prob"]

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

        # # Base station vars
        # self.num_base_stations = config["world"]["num_base_stations"]
        # bs_df = self.df_maze.sample(self.num_base_stations, random_state=47)[['row', 'col']]
        # self.base_stations = []
        # for t in bs_df.values.tolist():
        #     self.base_stations.append( (int(t[0]), int(t[1])) )

        # Agent vars
        self.num_uavs_bs = config["world"]["num_uavs_per_bs"]
        self.max_ugv_range = config['ugv']['range']
        self.drain_rate = config['ugv']['drain_rate']
        self.ugv_speed = config['ugv']['speed']
        
        # Simulation vars
        self.bms = config['world']['bms']
        self.max_time = config['world']['max_time']
        self.traffic_b = config['world']['traffic']

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
        self.total_energy_consumed = 0.0 # Corrected typo

        '''State vars'''
        self.ugv = None
        self.task_list = []
        self.active_task = []
        self.traffic_centeroids = []
        self.red_roads = []
        self.yellow_roads = []
        self.info = {}

        '''SETUP YOUR OBSERVATION SPACE, ACTION SPACE, ENVIRONMENT-SPECIFIC VARIABLES'''
        self.action_space = spaces.Discrete(config["ugv"]["num_primitives"]-1)
        
        active_task_space = spaces.MultiDiscrete(
            np.array([self.max_row+1, self.max_col+1])
        )
        ugv_pos_space = spaces.MultiDiscrete(
            np.array([self.max_row+1, self.max_col+1])
        )
        dir_wall_space = spaces.MultiDiscrete(
            np.array([3,3,3,3])
        )
        wall_occupancy_space = spaces.MultiDiscrete(np.array([2]*60))
        task_dir_space = spaces.Discrete(9)
        steps2dest_space = spaces.MultiDiscrete(np.array([10,10,10,10,10]))

        nb_traffic_space = spaces.MultiDiscrete(np.array([3,3,3,3]))
        battery_space = spaces.Box(
            low=0,
            high=self.max_ugv_range,
            dtype=np.int32
        )
        
        self.observation_space = spaces.Dict({
            'active_task_positions': active_task_space,
            'ugv_positions': ugv_pos_space,
            'wall_encoding': dir_wall_space,
            # 'wall_occupancy': wall_occupancy_space,
            'task_direction': task_dir_space,
            'steps2dest': steps2dest_space,
            'battery_levels': battery_space,
            'nb_traffic': nb_traffic_space,
        })
        
        # Initialize rendering if mode is 'human'
        if self.render_mode == "human":
            self._initialize_render()

    def _initialize_render(self):
        """Initializes pygame, screen, clock, fonts, and loads assets."""
        if self.screen is not None: # Avoid re-initialization
            return
        try:
            pygame.init()
            pygame.display.set_caption("LMD Simulation")
            self.clock = pygame.time.Clock()

            # Calculate screen dimensions based on maze size and cell pixel size
            # Add padding for potential info text display later
            padding = 200
            self.screen_width = self.max_col * self.cell_size_px
            self.screen_height = self.max_row * self.cell_size_px + padding # Add padding at the bottom
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            # Load font
            try:
                self.font = pygame.font.SysFont(None, 24) # Adjust font size as needed
            except Exception as e:
                print(f"Warning: Could not load system font. Text rendering might fail. Error: {e}")
                self.font = pygame.font.Font(None, 24) # Fallback to default font

            # Load images (adjust paths as necessary)
            try:
                # Assuming 'data' directory is in the same parent directory as 'envs'
                truck_img_path = os.path.join('data/truck.png')
                self.truck_image = pygame.image.load(truck_img_path).convert_alpha() # Use convert_alpha for transparency
                # Scale truck image to fit within a cell (e.g., 80% of cell size)
                scale_factor = 0.8
                img_size = int(self.cell_size_px * scale_factor)
                self.truck_image = pygame.transform.scale(self.truck_image, (img_size, img_size))
                # Optional: Load other images (warehouse, task, base station) here
                # self.warehouse_image = pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'warehouse.png')).convert_alpha(), (img_size, img_size))
                # self.task_image = pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'task.png')).convert_alpha(), (img_size // 2, img_size // 2))
                # self.bs_image = pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'base_station.png')).convert_alpha(), (img_size, img_size))

            except pygame.error as e:
                print(f"Warning: Could not load image assets. Rendering with shapes. Error: {e}")
                self.truck_image = None
                # self.warehouse_image = None
                # self.task_image = None
                # self.bs_image = None
        except Exception as e:
             print(f"Error initializing Pygame: {e}")
             self.render_mode = None # Disable rendering if initialization fails

    def _cell_to_pygame(self, row, col):
        """Converts maze (row, col) to pygame pixel coordinates (top-left of cell)."""
        # Pygame origin (0,0) is top-left
        px = (col - 1) * self.cell_size_px
        py = (row - 1) * self.cell_size_px # Adjusted for top-left origin
        return int(px), int(py)

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

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for node in G.nodes:
            wall_distances = []
            for dr, dc in directions:
                steps = 0
                current = node
                while steps < 2:
                    next_node = (current[0] + dr, current[1] + dc)
                    if not G.has_edge(current, next_node):
                        break
                    steps += 1
                    current = next_node
                wall_distances.append(steps)
            G.nodes[node]['wall_distance'] = wall_distances

        # # Wall Occupancy Grid 
        # kernel_size = 5
        # for i in range(1,self.max_row+1):
        #     for j in range(1,self.max_col+1):
        #         occupancy_grid = [0]*kernel_size*(kernel_size+1)*2
        #         m = len(occupancy_grid)
        #         for k in range(m):
        #             if k<m//2:
        #                 row_id = k//kernel_size
        #                 col_id = k%kernel_size
        #                 row_t_diff = row_id-2
        #                 row_b_diff = row_id-1
        #                 col_diff = col_id-1
        #                 cell_t = (i+row_t_diff,j+col_diff)
        #                 cell_b = (i+row_b_diff,j+col_diff)
        #                 if G.has_edge(cell_t,cell_b):
        #                     occupancy_grid[k] = 1
        #             else:
        #                 k_ = k-m//2
        #                 row_id = k_%kernel_size
        #                 col_id = k_//kernel_size
        #                 col_l_diff = col_id-2
        #                 col_r_diff = col_id-1
        #                 row_diff = row_id-1
        #                 cell_l = (j+row_diff,i+col_l_diff)
        #                 cell_r = (j+row_diff,i+col_r_diff)
        #                 if G.has_edge(cell_l,cell_r):
        #                     occupancy_grid[k] = 1
        #         G.nodes[(i,j)]['occupancy_grid'] = occupancy_grid

        return G
    
    def render(self):
        if self.render_mode == "human":
            self.sleep_flag = True
            if self.screen is None:
                self._initialize_render() # Initialize if not already done
                if self.screen is None: # Check if initialization failed
                    return # Cannot render

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close() # Proper cleanup
                    # Optionally signal the main loop to terminate
                    # This might require modifying the main loop logic
                    # For now, just close pygame and let the loop continue/error out
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("Escape key pressed - episode will be truncated")
                        self.escape_pressed = True

            # --- Drawing ---
            self.screen.fill(WHITE) # Clear screen

            # --- Draw Maze Area (Grid and Walls) ---
            maze_area_height = self.max_row * self.cell_size_px
            wall_thickness = 4 # Thinner walls
            # --- Draw Maze Area (Grid and Walls) using self.G ---
            for r in range(1, self.max_row + 1):
                for c in range(1, self.max_col + 1):
                    current_node = (r, c)
                    # Skip if node somehow doesn't exist in the graph (should not happen for grid)
                    if current_node not in self.G:
                        continue

                    px_top_left, py_top_left = self._cell_to_pygame(r, c)

                    # Check North Wall: Draw wall if no edge exists between current node and North neighbor (r-1, c)
                    # G.has_edge handles boundary cases (e.g., r=1, neighbor (0,c) doesn't exist, so has_edge is False)
                    north_neighbor = (r - 1, c)
                    if not self.G.has_edge(current_node, north_neighbor):
                        # Draw the North wall of cell (r, c)
                        pygame.draw.line(self.screen, BLACK, (px_top_left, py_top_left), (px_top_left + self.cell_size_px, py_top_left), wall_thickness)
                    else:
                        pygame.draw.line(self.screen, GRAY, (px_top_left, py_top_left), (px_top_left + self.cell_size_px, py_top_left), 3)

                    # Check West Wall: Draw wall if no edge exists between current node and West neighbor (r, c-1)
                    west_neighbor = (r, c - 1)
                    if not self.G.has_edge(current_node, west_neighbor):
                        # Draw the West wall of cell (r, c)
                        pygame.draw.line(self.screen, BLACK, (px_top_left, py_top_left), (px_top_left, py_top_left + self.cell_size_px), wall_thickness)
                    else:
                        pygame.draw.line(self.screen, GRAY, (px_top_left, py_top_left), (px_top_left, py_top_left + self.cell_size_px), 3)

            # Draw outer boundary walls (South and East edges explicitly)
            # This ensures the bottom border of the last row and the right border of the last column are drawn.
            pygame.draw.line(self.screen, BLACK, (0, maze_area_height), (self.screen_width, maze_area_height), wall_thickness) # Bottom boundary
            pygame.draw.line(self.screen, BLACK, (self.screen_width, 0), (self.screen_width, maze_area_height), wall_thickness) # Right boundary


            # --- Draw Traffic Roads ---
            road_thickness = max(1, int(self.cell_size_px * 0.15)) # Adjust thickness relative to cell size
            for u, v in self.red_roads:
                u_px_c, u_py_c = self._cell_to_pygame_center(u[0], u[1])
                v_px_c, v_py_c = self._cell_to_pygame_center(v[0], v[1])
                pygame.draw.line(self.screen, RED, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)
            for u, v in self.yellow_roads:
                u_px_c, u_py_c = self._cell_to_pygame_center(u[0], u[1])
                v_px_c, v_py_c = self._cell_to_pygame_center(v[0], v[1])
                pygame.draw.line(self.screen, YELLOW, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)


            # --- Draw Warehouse ---
            wh_r, wh_c = self.warehouse_pos
            wh_px_c, wh_py_c = self._cell_to_pygame_center(wh_r, wh_c)
            warehouse_radius = int(self.cell_size_px * 0.4)
            pygame.draw.circle(self.screen, BLUE, (wh_px_c, wh_py_c), warehouse_radius)

            # # --- Draw Base Stations ---
            # bs_radius = int(self.cell_size_px * 0.3)
            # for bs_r, bs_c in self.base_stations:
            #     bs_px_c, bs_py_c = self._cell_to_pygame_center(bs_r, bs_c)
            #     pygame.draw.circle(self.screen, DARK_GREEN, (bs_px_c, bs_py_c), bs_radius)
            #     pygame.draw.circle(self.screen, BLACK, (bs_px_c, bs_py_c), bs_radius, 1) # Border

            # --- Draw Tasks ---
            task_radius = int(self.cell_size_px * 0.25)
            task_r, task_c = self.active_task
            task_px_c, task_py_c = self._cell_to_pygame_center(task_r, task_c)
            pygame.draw.circle(self.screen, MAGENTA, (task_px_c, task_py_c), task_radius)

            # --- Draw UGVs and Battery ---

            ugv_r, ugv_c = self.ugv.position
            ugv_px_c, ugv_py_c = self._cell_to_pygame_center(ugv_r, ugv_c)
            agent_color = 0

            # Draw UGV (Image or Shape)
            if self.truck_image:
                img_rect = self.truck_image.get_rect(center=(ugv_px_c, ugv_py_c))
                self.screen.blit(self.truck_image, img_rect)
                # Draw a small colored circle on the truck for identification if needed
                # id_radius = int(self.cell_size_px * 0.1)
                # pygame.draw.circle(self.screen, agent_color, (ugv_px_c + img_rect.width // 3, ugv_py_c - img_rect.height // 3), id_radius)
            else:
                # Fallback to drawing a colored rectangle
                ugv_size = int(self.cell_size_px * 0.7)
                ugv_rect = pygame.Rect(0, 0, ugv_size, ugv_size)
                ugv_rect.center = (ugv_px_c, ugv_py_c)
                pygame.draw.rect(self.screen, agent_color, ugv_rect)
                pygame.draw.rect(self.screen, BLACK, ugv_rect, 1) # Border

            # Draw Battery Bar below UGV
            battery_width = int(self.cell_size_px * 0.8)
            battery_height = max(3, int(self.cell_size_px * 0.1)) # Ensure minimum height
            battery_x = ugv_px_c - battery_width // 2
            # Position below the center, slightly offset
            battery_y = ugv_py_c + (self.truck_image.get_height() // 2 if self.truck_image else self.cell_size_px // 2) + 2

            filled_width = int(self.ugv.current_range_percent * battery_width)

            # Background of battery bar (e.g., light gray)
            pygame.draw.rect(self.screen, GRAY, (battery_x, battery_y, battery_width, battery_height))
            # Filled portion (Green)
            pygame.draw.rect(self.screen, GREEN, (battery_x, battery_y, filled_width, battery_height))
            # Border
            pygame.draw.rect(self.screen, BLACK, (battery_x, battery_y, battery_width, battery_height), 1)

            # --- Draw Info Text ---
            if self.font:
                info_y_start = maze_area_height + 5 # Start below maze area
                info_line_height = 20
                texts = [
                    f"Time: {self.time_elapsed:.2f}s",
                    f"Timestep: {self.current_timestep}",
                    f"Tasks Done: {self.num_tasks_completed}",
                    f"Total Dist: {self.total_ev_distance:.1f}m",
                    f"Total Energy: {self.total_energy_consumed:.1f}",
                ]
                for i, text in enumerate(texts):
                    text_surface = self.font.render(text, True, BLACK)
                    self.screen.blit(text_surface, (5, info_y_start + i * info_line_height))


            # --- Update Display ---
            pygame.display.flip()
            
            # Wait until the Return key is pressed before ending render
            # if self.num_tasks_completed < 70:
            #     self.clock.tick(self.metadata["render_fps"])
            # else:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            waiting = False
                self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "print":
            # Print-based rendering
            print(f"--- Timestep: {self.current_timestep} ---")
            print(f"  Time Elapsed: {self.time_elapsed:.2f}")
            print(f"  Tasks Completed: {self.num_tasks_completed}")
            print(f"  Active Tasks: {self.active_task}")
            print(f"  UGV: Pos={self.ugv.position}, Batt={self.ugv.current_range_percent*100:.1f}%")
            print(f"  Total EV Distance: {self.total_ev_distance:.2f}")
            print(f"  Total Energy Consumed: {self.total_energy_consumed:.2f}")
            if self.traffic_b:
                 print(f"  Red Roads: {len(self.red_roads)}, Yellow Roads: {len(self.yellow_roads)}")
            print("-" * (len(f"--- Timestep: {self.current_timestep} ---")))


        elif self.render_mode == "rgb_array":
            # Return screen surface as numpy array
            if self.screen is None:
                 # Need to render one frame to get the surface if not initialized
                 if self.render_mode != "human": # Avoid re-initializing if already in human mode
                     self._initialize_render()
                     if self.screen is None: # Check init failure
                          raise RuntimeError("RGB array requested but Pygame screen initialization failed.")
                     # Perform a minimal render to get the surface state
                     self.screen.fill(WHITE)
                     # Potentially draw static elements like walls/warehouse here if needed for the array
                     # This part might need full render logic if static elements are complex
                     # For now, just return the blank state or call the full render logic once
                     self._render_frame_logic() # Call the core drawing logic
                 else:
                     # If called in human mode but screen somehow None, re-init
                     self._initialize_render()
                     if self.screen is None: # Check init failure
                          raise RuntimeError("RGB array requested but Pygame screen initialization failed.")


            if self.screen: # Check if screen was successfully created
                # Ensure the current frame is drawn before capturing
                self._render_frame_logic() # Call the core drawing logic
                return pygame.surfarray.array3d(pygame.display.get_surface()).transpose(1, 0, 2) # Transpose for correct HxWxC format
            else:
                # Return a placeholder array or raise an error if screen init failed
                # Placeholder: black screen of expected size
                if self.screen_width and self.screen_height:
                    return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                else:
                    # Cannot determine size, return empty or raise error
                    raise RuntimeError("RGB array requested but screen dimensions are unknown.")
        # else: # No rendering
        #     pass

    def _render_frame_logic(self):
        """Contains the core drawing logic, callable by render() and rgb_array mode."""
        # This duplicates the drawing section from render() for use in rgb_array mode
        # Ideally, refactor render() to call this method.
        # --- Drawing ---
        self.screen.fill(WHITE) # Clear screen
        maze_area_height = self.max_row * self.cell_size_px
        wall_thickness = 4 # Thinner walls

        # --- Draw Maze Area (Grid and Walls) using self.G ---
        for r in range(1, self.max_row + 1):
            for c in range(1, self.max_col + 1):
                current_node = (r, c)
                # Skip if node somehow doesn't exist in the graph (should not happen for grid)
                if current_node not in self.G:
                    continue

                px_top_left, py_top_left = self._cell_to_pygame(r, c)

                # Check North Wall: Draw wall if no edge exists between current node and North neighbor (r-1, c)
                north_neighbor = (r - 1, c)
                if not self.G.has_edge(current_node, north_neighbor):
                    pygame.draw.line(self.screen, BLACK, (px_top_left, py_top_left), (px_top_left + self.cell_size_px, py_top_left), wall_thickness)
                else: # Optional: Draw grid line if no wall
                    pygame.draw.line(self.screen, GRAY, (px_top_left, py_top_left), (px_top_left + self.cell_size_px, py_top_left), 1)


                # Check West Wall: Draw wall if no edge exists between current node and West neighbor (r, c-1)
                west_neighbor = (r, c - 1)
                if not self.G.has_edge(current_node, west_neighbor):
                    pygame.draw.line(self.screen, BLACK, (px_top_left, py_top_left), (px_top_left, py_top_left + self.cell_size_px), wall_thickness)
                else: # Optional: Draw grid line if no wall
                    pygame.draw.line(self.screen, GRAY, (px_top_left, py_top_left), (px_top_left, py_top_left + self.cell_size_px), 1)


        # Draw outer boundary walls (South and East edges explicitly)
        pygame.draw.line(self.screen, BLACK, (0, maze_area_height), (self.screen_width, maze_area_height), wall_thickness) # Bottom boundary
        pygame.draw.line(self.screen, BLACK, (self.screen_width, 0), (self.screen_width, maze_area_height), wall_thickness) # Right boundary

        # --- Draw Traffic Roads ---
        road_thickness = max(1, int(self.cell_size_px * 0.15))
        for u, v in self.red_roads:
            u_px_c, u_py_c = self._cell_to_pygame_center(u[0], u[1])
            v_px_c, v_py_c = self._cell_to_pygame_center(v[0], v[1])
            pygame.draw.line(self.screen, RED, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)
        for u, v in self.yellow_roads:
            u_px_c, u_py_c = self._cell_to_pygame_center(u[0], u[1])
            v_px_c, v_py_c = self._cell_to_pygame_center(v[0], v[1])
            pygame.draw.line(self.screen, YELLOW, (u_px_c, u_py_c), (v_px_c, v_py_c), road_thickness)
        # --- Draw Warehouse ---
        wh_r, wh_c = self.warehouse_pos
        wh_px_c, wh_py_c = self._cell_to_pygame_center(wh_r, wh_c)
        warehouse_radius = int(self.cell_size_px * 0.4)
        pygame.draw.circle(self.screen, BLUE, (wh_px_c, wh_py_c), warehouse_radius)
        # --- Draw Base Stations ---
        # bs_radius = int(self.cell_size_px * 0.3)
        # for bs_r, bs_c in self.base_stations:
        #     bs_px_c, bs_py_c = self._cell_to_pygame_center(bs_r, bs_c)
        #     pygame.draw.circle(self.screen, DARK_GREEN, (bs_px_c, bs_py_c), bs_radius)
        #     pygame.draw.circle(self.screen, BLACK, (bs_px_c, bs_py_c), bs_radius, 1)
        # --- Draw Tasks ---
        task_radius = int(self.cell_size_px * 0.25)
        task_r, task_c = self.active_task
        task_px_c, task_py_c = self._cell_to_pygame_center(task_r, task_c)
        pygame.draw.circle(self.screen, MAGENTA, (task_px_c, task_py_c), task_radius)
        # --- Draw UGVs and Battery ---
        ugv_r, ugv_c = self.ugv.position
        ugv_px_c, ugv_py_c = self._cell_to_pygame_center(ugv_r, ugv_c)
        agent_color = 0
        if self.truck_image:
            img_rect = self.truck_image.get_rect(center=(ugv_px_c, ugv_py_c))
            self.screen.blit(self.truck_image, img_rect)
        else:
            ugv_size = int(self.cell_size_px * 0.7)
            ugv_rect = pygame.Rect(0, 0, ugv_size, ugv_size); ugv_rect.center = (ugv_px_c, ugv_py_c)
            pygame.draw.rect(self.screen, agent_color, ugv_rect)
            pygame.draw.rect(self.screen, BLACK, ugv_rect, 1)
        battery_width = int(self.cell_size_px * 0.8)
        battery_height = max(3, int(self.cell_size_px * 0.1))
        battery_x = ugv_px_c - battery_width // 2
        battery_y = ugv_py_c + (self.truck_image.get_height() // 2 if self.truck_image else self.cell_size_px // 2) + 2
        filled_width = int(self.ugv.current_range_percent * battery_width)
        pygame.draw.rect(self.screen, GRAY, (battery_x, battery_y, battery_width, battery_height))
        pygame.draw.rect(self.screen, GREEN, (battery_x, battery_y, filled_width, battery_height))
        pygame.draw.rect(self.screen, BLACK, (battery_x, battery_y, battery_width, battery_height), 1)
        # --- Draw Info Text ---
        if self.font:
            info_y_start = maze_area_height + 5
            info_line_height = 20
            texts = [f"Time: {self.time_elapsed:.2f}s", f"Timestep: {self.current_timestep}", f"Tasks Done: {self.num_tasks_completed}", f"Total Dist: {self.ev_distance_traveled:.1f}m", f"Total Energy: {self.total_energy_consumed:.1f}"]
            for i, text in enumerate(texts):
                text_surface = self.font.render(text, True, BLACK)
                self.screen.blit(text_surface, (5, info_y_start + i * info_line_height))

    def _cell_to_pygame_center(self, row, col):
        """Converts maze (row, col) to pygame pixel coordinates (center of cell)."""
        px, py = self._cell_to_pygame(row, col)
        px += self.cell_size_px // 2
        py += self.cell_size_px // 2
        return int(px), int(py)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Call parent reset for seeding RNG
        if seed: # Seeding is handled by super().reset()
            random.seed(seed)
            np.random.seed(seed) # Also seed numpy for consistent sampling if used

        self.df_maze = generate_df_maze(
            self.max_row,
            self.max_col,
            self.maze_lp,
        )
        self.G = self._build_graph() # Rebuild graph with new maze

        self.action_response = None
        self.prev_task_distance = 100
        self.last_move_time = 0
        self.escape_pressed = False # Reset escape key flag

        self.ugv = UGV(ugv_id=0, base_position=self.warehouse_pos, cell_dist=self.cell_size, max_range=self.max_ugv_range,drain_rate=self.drain_rate, 
                       speed=self.ugv_speed, G=self.G)
        self.task_list = random.choices(list(self.G.nodes), k=self.num_tasks) # Randomly select tasks from graph nodes
        self.active_task = self.task_list[0]
        self.red_roads = []
        self.yellow_roads = []
        self.traffic_centeroids = []
        for u, v in self.G.edges():
            self.G[u][v]['traffic'] = 0 # Reset traffic on graph edges
        if self.traffic_b:
            self.fill_traffic_centroids() # Generate initial traffic if enabled
        # Initialize UGV states - pass physical cell_size

        # Reset metrics
        self.total_ev_distance = 0.0
        self.num_tasks_completed = 0
        self.current_timestep = 0
        self.time_elapsed = 0.0
        self.total_energy_consumed = 0.0

        initial_obs = self._get_observation()
        self.update_info() # Update info dict with initial state

        # Render initial state if mode is human
        # No need to call render here, the main loop should call it after reset
        # if self.render_mode == "human":
        #     self.render()

        return initial_obs, self.info

    def fill_task_list(self):
        if len(self.task_list) < self.num_tasks and random.random() < self.task_prob:
            m = self.num_tasks-len(self.task_list)
            new_task = random.sample(list(self.G.nodes),m)
            self.task_list.extend(new_task)
        if self.active_task == None:
            self.active_task = min(self.task_list, key=lambda x: nx.shortest_path_length(self.G, self.ugv.position, x))

    def fill_traffic_centroids(self):
        # Reset previous roads and weights
        self.red_roads = []
        self.yellow_roads = []
        for u, v in self.G.edges():
            self.G[u][v]['traffic'] = 0
            self.G[u][v]['weight'] = 1 # Reset weight

        if random.random() < self.traffic_prob:
            self.traffic_centeroids = random.sample(list(self.G.nodes), self.traffic_num_centroids)
            centroid_array = np.array(self.traffic_centeroids)

            for u, v in self.G.edges():
                midpoint = np.array([(u[0] + v[0])/2, (u[1] + v[1])/2])
                total_contrib = 0
                if centroid_array.size > 0: # Check if centroids exist
                    for centroid in centroid_array:
                        dist_sq = np.sum((midpoint - centroid)**2)
                        contrib = np.exp(-dist_sq / (2.0 * self.traffic_std_dev**2))
                        total_contrib += contrib

                # Normalize and set traffic level and weight
                # Define weight penalties for traffic levels
                weight_penalty_yellow = 2 # Example: Yellow roads take twice as long
                weight_penalty_red = 5    # Example: Red roads take five times as long

                if total_contrib < 0.33:
                    self.G[u][v]['traffic'] = 0
                    self.G[u][v]['weight'] = 1
                elif total_contrib < 0.66:
                    self.G[u][v]['traffic'] = 1
                    self.G[u][v]['weight'] = weight_penalty_yellow
                    self.yellow_roads.append((u, v))
                else:
                    self.G[u][v]['traffic'] = 2
                    self.G[u][v]['weight'] = weight_penalty_red
                    self.red_roads.append((u, v))
        else:
            self.traffic_centeroids = []
            # Ensure weights are reset even if no traffic is generated
            for u, v in self.G.edges():
                 self.G[u][v]['traffic'] = 0
                 self.G[u][v]['weight'] = 1


    def step(self, action):
        # Apply action, calculate reward, get next observation
        self._apply_action(action)
        reward = self._get_reward()
        if self.ugv.position == self.active_task:
            self.task_list.pop(0)
            self.active_task = self.task_list[0]
            self.num_tasks_completed += 1
            self.prev_task_distance = nx.shortest_path_length(self.G, self.ugv.position, self.active_task)
        obs = self._get_observation() # Gets observation *after* action/reward
        done = self._check_termination_condition() # Check termination based on new state

        # Update time and timestep *after* action and reward calculation for the current step
        self.current_timestep += 1
        self.time_elapsed += self.last_move_time # Accumulate time based on last move

        # Update traffic periodically *before* updating info for the next step's rendering
        if self.traffic_b:
            if self.current_timestep % self.traffic_reset_dur == 0:
                self.fill_traffic_centroids()

        # Update info dict *after* all state changes for the current step
        self.update_info()

        # Gymnasium expects terminated, truncated, info
        terminated = done
        truncated = False # Assuming truncation is handled by max_timesteps check in done
        
        # Check if escape key was pressed or max timesteps reached
        if terminated or self.escape_pressed:
             truncated = True
             if self.escape_pressed:
                 self.info["truncated_by_escape"] = True
             else:
                 terminated = True # Gym standard is that max timestep truncation implies termination

        self.last_move_time = 0
        self.latest_completed_tasks = []
        self.action_response = None

        return obs, reward, terminated, truncated, self.info # Return standard gym step tuple

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
        active_task_positions_flat = np.array(self.active_task, dtype=np.int32).flatten()
        ugv_positions_flat = np.array(list(self.ugv.position)).flatten()

        ugv_pos = self.ugv.position
        task_pos = self.active_task
        task_dir_id = None
        delta_row = task_pos[0] - ugv_pos[0]
        delta_col = task_pos[1] - ugv_pos[1]
        if delta_row == 0 and delta_col == 0:
            task_dir_id = 8  # No movement, indeterminate direction
        else:
            # Convert grid differences to an angle with 0 degrees = North and increasing clockwise.
            # Using math.atan2(delta_col, -delta_row) gives the desired angle.
            angle = math.degrees(math.atan2(delta_col, -delta_row)) % 360
            # Divide the circle into 8 sectors of 45° each.
            task_dir_id = int(((angle + 22.5) % 360) // 45)

        # Neighborhood Wall Encoding
        wall_encoding = np.array(self.G.nodes[ugv_pos]['wall_distance'], dtype=np.int32).flatten()
        # wall_occupancy = np.array(self.G.nodes[ugv_pos]['occupancy_grid'], dtype=np.int32).flatten()

        # Steps to destination in each direction
        steps2dest = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (0,0)]:
            r,c = ugv_pos[0] + dr, ugv_pos[1] + dc
            if r > 0 and r <= self.max_row and c > 0 and c <= self.max_col:
                tmp = min(nx.shortest_path_length(self.G, (r,c), task_pos), 9)
                steps2dest.append(tmp)
            else:
                steps2dest.append(9)
        steps2dest = np.array(steps2dest, dtype=np.int32).flatten()

        # Battery Levels
        battery_flat = np.array(self.ugv.current_range, dtype=np.int32)

        # Neighbor Traffic
        nb_traffic = self._get_nb_traffic(self.ugv.position)
        nb_traffic_flat = np.array(nb_traffic, dtype=np.int32).flatten()


        # Ensure observation matches the defined space structure
        obs_dict = {
            'active_task_positions': active_task_positions_flat,
            'ugv_positions': ugv_positions_flat,
            'wall_encoding': wall_encoding,
            # 'wall_occupancy': wall_occupancy,
            'task_direction': task_dir_id,
            'steps2dest': steps2dest,
            'battery_levels': battery_flat,
            'nb_traffic': nb_traffic_flat,
        }

        return obs_dict


    def _get_info(self):
        # This method is often used by wrappers, ensure it returns the latest info
        return self.info

    def update_info(self):
        self.total_ev_distance = self.ugv.distance_traveled
        self.total_energy_consumed = self.ugv.energy_consumed
        self.info = {
            "maze": self.df_maze,
            "graph": self.G,
            "warehouse_pos": self.warehouse_pos,
            "time_elapsed":self.time_elapsed,
            "num_tasks_completed": self.num_tasks_completed,
            "ev_distance_traveled": self.total_ev_distance,
            "total_energy_consumed": self.total_energy_consumed,
        }

    def _apply_action(self, action):
        action = int(action) # Ensure action is an integer

        max_move_time = self.cell_size/self.ugv_speed # Track the longest move time in this step for time_elapsed

        response, move_time = self.ugv.move(action, self.G)
        self.action_response = response
        max_move_time = max(max_move_time, move_time)

        # Recharge at warehouse
        if self.ugv.position == self.warehouse_pos:
            self.ugv.recharge()

        self.last_move_time = max_move_time # Use the max time for simulation clock progression


    def _get_reward(self):
        # Calculate reward based on the outcome of the action in the previous state
        reward = 0.0

        # --- Progress-Based Shaping Reward ---
        # Reward for moving closer to the nearest task
        bonus_factor = 5.0 # Adjust shaping reward magnitude
        current_distance = nx.shortest_path_length(self.G, self.ugv.position, self.active_task)
        reward += bonus_factor * (self.prev_task_distance - current_distance)
        if current_distance < self.prev_task_distance and current_distance > 0:
            self.prev_task_distance = current_distance
        
        # print(f"Reward: {reward:.2f}, Current Distance: {current_distance}, Previous Distance: {self.prev_task_distance}")

        # --- Task Completion Reward ---
        task_completion_reward = 50.0
        if self.ugv.position == self.active_task:
            reward += task_completion_reward

        # --- Time Penalty ---
        # Penalize based on the time taken for the action
        time_penalty_factor = 1.0 # Adjust this factor based on desired behavior
        reward -= time_penalty_factor * self.last_move_time/self.cell_size

        # --- Invalid Move Penalty ---
        invalid_move_penalty = 40.0
        if self.action_response == 0:
            reward -= invalid_move_penalty
        # No need to reset action_response here, it's reset in _apply_action


        return reward


    def _check_termination_condition(self):
        # Terminate if maximum timesteps are reached (handled by truncated flag in step).
        if self.time_elapsed >= self.max_time:
            # print(f"Termination: Max timesteps ({self.max_timesteps}) reached.")
            return True # Indicates termination

        # Optional: Terminate if any UGV runs out of battery *not* at the warehouse
        # for ugv in self.ugv_states:
        #     if ugv.current_range <= 0 and ugv.position != self.warehouse_pos:
        #          print(f"Termination: UGV {ugv.agent_id} ran out of battery at {ugv.position}.")
        #          return True

        return False # Not terminated based on these conditions

    def close(self):
        """Cleans up pygame resources."""
        if self.screen is not None:
            print("Closing Pygame window.")
            pygame.display.quit()
            pygame.quit()
            self.screen = None # Mark as closed
