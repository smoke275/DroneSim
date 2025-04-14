import os
import sys
import threading
from enum import Enum, auto
import time
import numpy as np
import random
from sklearn.cluster import KMeans
import yaml
import tkinter as tk
import numpy as np
import itertools
from collections import defaultdict
import pickle

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QPointF, QTimer
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# Monkey-patch the 'zoomed' state to 'normal'
tk.Tk.state = lambda self, s=None: self.wm_state('normal' if s == 'zoomed' else s)

import gymnasium as gym

import envs
from agents.dp import DPAgent
from agents.sarsa import SARSAAgent  

WRITE_to_file = True
SET_t = 5
SEARCH = False
END = False
CAPTURE = True

ITERATIONS = 90
sem = threading.Semaphore()
BOUNDARY_X = 500
BOUNDARY_Y = 500

if ITERATIONS == 1:
    DELAY = float('inf')

class OPERATION(Enum):
    point = auto()
    line = auto()
    dotted_line = auto()
    circle = auto()
    filled_circle = auto()
    border_circle = auto()
    polygon = auto()
    filled_polygon = auto()
    dotted_polygon = auto()
    border_polygon = auto()
    text = auto()
    image = auto()

def cell_to_canvas(row, col, cell_size):
    """
    Converts a (row, col) cell coordinate to canvas x,y coordinates using cell_size.
    """
    x = -BOUNDARY_X + (col - 1) * cell_size + cell_size / 2
    y = BOUNDARY_Y - (row - 1) * cell_size - cell_size / 2
    return x, y

drone_offsets = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(20)]

class Window(QMainWindow):

    def __init__(self, config_file, policy_name):
        super().__init__()

        self.title = "Simulation"

        self.action_stack = []
        self.main_stack = []
        self.permanent_elements = []

        # Load config
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.config = config

        # Check if the inference log directory exists, if not create it
        if not os.path.exists(f"runs/sarsa/inference/{policy_name}"):
            os.makedirs(f"runs/sarsa/inference/{policy_name}")

        self.log_file = f"runs/sarsa/inference/{policy_name}/log.txt"
        
        with open(self.log_file, 'w') as f:
            f.write("Inference Log\n")
            f.write("=============\n")
        
        # Load and scale down the truck image
        truck_image = QtGui.QPixmap('data/truck.png')
        self.scaled_truck_image = truck_image.scaled(40, 40, Qt.KeepAspectRatio)

        # Load and scale down the drone image
        drone_image = QtGui.QPixmap('data/transparent_drone.png')
        self.scaled_drone_image = drone_image.scaled(50, 50, Qt.KeepAspectRatio)  # Scale to 30x30 pixels
        
        # If no external world is provided, create one
        self.env = gym.make("LMDEnv-v0", config=self.config, render_mode="human")
        # Load the Q-table from a pickle file

         # Initialize the DP agent
        self.agent = SARSAAgent(self.env, policy_name=policy_name)
        
        self.InitWindow()

        # Timer for triggering updates
        timer = QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(1000)

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(400, 600, 700, 700)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setWindow(QRect(-BOUNDARY_X, -BOUNDARY_Y, 2 * BOUNDARY_X, 2 * BOUNDARY_Y))
        painter.setViewport(QRect(0, 0, 700, 700))
        painter.scale(1, -1)

        painter.setBrush(QBrush(QColor('#f9fafb'), Qt.SolidPattern))
        background_points = [
            QPointF(-BOUNDARY_X, BOUNDARY_Y),
            QPointF(BOUNDARY_X, BOUNDARY_Y),
            QPointF(BOUNDARY_X, -BOUNDARY_Y),
            QPointF(-BOUNDARY_X, -BOUNDARY_Y),
            QPointF(-BOUNDARY_X, BOUNDARY_Y)
        ]
        poly = QPolygonF(background_points)
        painter.drawPolygon(poly)

        sem.acquire()
        my_stack = self.main_stack
        sem.release()

        for item in my_stack:
            operation = item[0]

            if operation == OPERATION.point:
                painter.setPen(QPen(item[4], item[3], Qt.SolidLine))
                painter.drawPoint(item[1], item[2])

            elif operation == OPERATION.line:
                painter.setPen(QPen(item[6], item[5], Qt.SolidLine))
                painter.drawLine(QPointF(item[1], item[2]), QPointF(item[3], item[4]))

            elif operation == OPERATION.dotted_line:
                painter.setPen(QPen(item[6], item[5], Qt.DotLine))
                painter.drawLine(QPointF(item[1], item[2]), QPointF(item[3], item[4]))

            elif operation == OPERATION.circle:
                painter.setPen(QPen(item[5], item[4], Qt.SolidLine))
                painter.drawEllipse(QPointF(item[1], item[2]), item[3], item[3])

            elif operation == OPERATION.filled_circle:
                painter.setPen(QPen(item[5], item[4], Qt.SolidLine))
                painter.setBrush(QBrush(item[5], Qt.SolidPattern))
                painter.drawEllipse(QPointF(item[1], item[2]), item[3], item[3])

            elif operation == OPERATION.border_circle:
                painter.setPen(QPen(Qt.black, item[4], Qt.SolidLine))
                painter.setBrush(QBrush(item[5], Qt.SolidPattern))
                painter.drawEllipse(QPointF(item[1], item[2]), item[3], item[3])

            elif operation == OPERATION.polygon:
                painter.setPen(QPen(item[4], item[3], Qt.SolidLine))
                painter.setBrush(QBrush(Qt.NoBrush))
                x_val = item[1]
                y_val = item[2]
                poly_points = []
                for j in range(len(x_val)):
                    poly_points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(poly_points)
                painter.drawPolygon(poly)

            elif operation == OPERATION.dotted_polygon:
                painter.setPen(QPen(item[4], item[3], Qt.DotLine))
                painter.setBrush(QBrush(Qt.NoBrush))
                x_val = item[1]
                y_val = item[2]
                poly_points = []
                for j in range(len(x_val)):
                    poly_points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(poly_points)
                painter.drawPolygon(poly)

            elif operation == OPERATION.filled_polygon:
                painter.setPen(QPen(item[4], item[3], Qt.SolidLine))
                painter.setBrush(QBrush(item[4], Qt.SolidPattern))
                x_val = item[1]
                y_val = item[2]
                poly_points = []
                for j in range(len(x_val)):
                    poly_points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(poly_points)
                painter.drawPolygon(poly)

            elif operation == OPERATION.border_polygon:
                painter.setPen(QPen(Qt.black, item[3], Qt.SolidLine))
                painter.setBrush(QBrush(item[4], Qt.SolidPattern))
                x_val = item[1]
                y_val = item[2]
                poly_points = []
                for j in range(len(x_val)):
                    poly_points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(poly_points)
                painter.drawPolygon(poly)

            elif operation == OPERATION.text:
                painter.setPen(QPen(item[4], item[3], Qt.SolidLine))
                painter.save()
                painter.scale(1, -1)
                painter.drawText(item[1], item[2], item[5])
                painter.restore()

            elif operation == OPERATION.image:
                painter.drawPixmap(int(item[1]), int(item[2]), item[3])

    def draw(self, value):
        """
        Adds a drawing command to the action stack.
        """
        self.action_stack.append(value)

    def execute(self):
        """
        Merges permanent elements with newly added commands, then triggers a repaint.
        """
        sem.acquire()
        self.main_stack.clear()
        self.main_stack = self.permanent_elements + self.action_stack
        self.action_stack = []
        sem.release()
        self.update()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape:
            # world_state = self.env._get_info()
            # num_tasks_completed = world_state["num_tasks_completed"]
            # ev_distance_traveled = world_state["ev_distance_traveled"]
            # print("Simulation completed.")
            # print(f"Tasks completed: {num_tasks_completed}")
            # print(f"EV distance traveled: {ev_distance_traveled}")
            self.close()
        if e.key() == Qt.Key_Return:
            self.tmp = 1

    def draw_maze(self, df_maze, base_stations):
        """
        Reads the loaded maze DataFrame, calculates self.cell_size, and draws
        all walls plus the warehouse circle as permanent elements.
        """
        self.df_maze = df_maze
        self.max_row = df_maze['row'].max()
        self.max_col = df_maze['col'].max()

        # Calculate cell size to fit the maze within the canvas while maintaining aspect ratio
        self.canvas_width = 2 * BOUNDARY_X
        self.canvas_height = 2 * BOUNDARY_Y
        self.maze_width = self.max_col
        self.maze_height = self.max_row
        self.cell_size = min(self.canvas_width / self.maze_width, self.canvas_height / self.maze_height)

        # Find the warehouse center
        self.center_row = self.max_row // 2
        self.center_col = self.max_col // 2
        self.warehouse_x, self.warehouse_y = cell_to_canvas(self.center_row, self.center_col, self.cell_size)

        base_stations = [cell_to_canvas(x,y,self.cell_size) for x,y in base_stations]

        # Draw the maze grid
        # Draw vertical grid lines
        for col in range(self.max_col + 1):
            x = -BOUNDARY_X + col * self.cell_size
            self.permanent_elements.append([
                OPERATION.line, 
                x, BOUNDARY_Y, 
                x, BOUNDARY_Y - self.maze_height * self.cell_size, 
                1, Qt.gray
            ])

        # Draw horizontal grid lines
        for row in range(self.max_row + 1):
            y = BOUNDARY_Y - row * self.cell_size
            self.permanent_elements.append([
                OPERATION.line, 
                -BOUNDARY_X, y, 
                -BOUNDARY_X + self.maze_width * self.cell_size, y, 
                1, Qt.gray
            ])

        # Draw the maze walls
        processed_cells = set()
        wall_thickness = 3
        for index, row_data in df_maze.iterrows():
            row_idx, col_idx = row_data['row'], row_data['col']
            if (row_idx, col_idx) in processed_cells:
                continue

            # processed_cells.add((row_idx, col_idx))
            x, y = cell_to_canvas(row_idx, col_idx, self.cell_size)

            if row_data['E'] == 0:  # East wall
                if 0 < col_idx < self.max_col:# and (row_idx, col_idx + 1) not in processed_cells:
                    self.permanent_elements.append([
                        OPERATION.line,
                        x + self.cell_size / 2,
                        y - self.cell_size / 2,
                        x + self.cell_size / 2,
                        y + self.cell_size / 2,
                        wall_thickness,
                        Qt.black
                    ])
                    # processed_cells.add((row_idx, col_idx + 1))

            if row_data['W'] == 0:  # West wall
                if 0 < col_idx <= self.max_col:# and (row_idx, col_idx - 1) not in processed_cells:
                    self.permanent_elements.append([
                        OPERATION.line,
                        x - self.cell_size / 2,
                        y - self.cell_size / 2,
                        x - self.cell_size / 2,
                        y + self.cell_size / 2,
                        wall_thickness,
                        Qt.black
                    ])
                    # processed_cells.add((row_idx, col_idx - 1))

            if row_data['N'] == 0:  # North wall
                if 0 < row_idx < self.max_row:# and (row_idx + 1, col_idx) not in processed_cells:
                    self.permanent_elements.append([
                        OPERATION.line,
                        x - self.cell_size / 2,
                        y + self.cell_size / 2,
                        x + self.cell_size / 2,
                        y + self.cell_size / 2,
                        wall_thickness,
                        Qt.black
                    ])
                    # processed_cells.add((row_idx + 1, col_idx))

            if row_data['S'] == 0:  # South wall
                if 0 < row_idx <= self.max_row:# and (row_idx - 1, col_idx) not in processed_cells:
                    self.permanent_elements.append([
                        OPERATION.line,
                        x - self.cell_size / 2,
                        y - self.cell_size / 2,
                        x + self.cell_size / 2,
                        y - self.cell_size / 2,
                        wall_thickness,
                        Qt.black
                    ])
                    # processed_cells.add((row_idx - 1, col_idx))

        # Draw the warehouse as a permanent element
        warehouse_size = 15
        self.permanent_elements.append([
            OPERATION.filled_circle,
            self.warehouse_x,
            self.warehouse_y,
            warehouse_size,
            1,
            Qt.blue
        ])

        # Draw base stations using self.draw
        for x,y in base_stations:
            size = 20
            self.permanent_elements.append([OPERATION.filled_polygon,
                        [x - size / 2, x + size / 2, x + size / 2, x - size / 2],
                        [y - size / 2, y - size / 2, y + size / 2, y + size / 2],
                        1, get_color(7)])

    def draw_state(self, world_state, T=120):
        """
        Unpacks the relevant fields from world_state and draws EV positions, paths, tasks, and battery.
        Note: The values in patrol_positions, patrol_paths, and active_tasks are in cell coordinates,
        so we convert them to canvas coordinates first.
        """
        velocity = self.cell_size/T

        num_patrols = world_state["num_patrols"]
        if len(self.previous_positions) == 0:
            self.previous_positions = [(self.warehouse_x, self.warehouse_y) for _ in range(num_patrols)]    

        patrol_positions = world_state["patrol_positions"]  # cell coords
        # patrol_colors = world_state["patrol_colors"]
        # patrol_paths = world_state["patrol_paths"]          # each path in cell coords
        scaled_truck_image = self.scaled_truck_image
        R_P = world_state["R_P"]                            # battery fraction
        active_tasks = world_state["active_tasks"]          # list of cell coords for tasks
        # drone_positions = world_state['drone_positions']
        # drone_status = world_state['drone_status']
        # doff = drone_offsets[:8]

        # Convert numeric color IDs to actual QColor via get_color (if needed)
        # If patrol_colors are already valid PyQt colors, you can skip this.
        # patrol_colors = [get_color(i) for i in patrol_colors]
        
        for t in range(T):
            # 1) Draw current tasks as filled circles in canvas coords
            task_size = 5
            for idx, task in enumerate(active_tasks):
                if task is not None:
                    # Convert cell coords (task[0], task[1]) -> canvas coords
                    cx, cy = cell_to_canvas(task[0], task[1], self.cell_size)
                    self.draw([
                        OPERATION.filled_circle, 
                        cx, cy, task_size, 1, get_color(0)
                    ])
                else:
                    cx, cy = cell_to_canvas(patrol_positions[i][0], patrol_positions[i][1], self.cell_size)
                    self.draw([
                        OPERATION.filled_circle, 
                        cx, cy, task_size, 1, get_color(0)
                    ])

            # # 2) Draw dotted-line paths for each EV, converting path from cell coords to canvas coords
            # converted_paths = []
            # for i in range(num_patrols):
            #     cell_path = patrol_paths[i]  # list of (row, col)
            #     if len(cell_path) == 0:
            #         cell_path.append(patrol_positions[i])
            #     if not cell_path:
            #         converted_paths.append([])
            #         continue
                
                
            #     # Convert entire path to canvas
            #     canvas_path = []
            #     for (r, c) in cell_path:
            #         px, py = cell_to_canvas(r, c, self.cell_size)
            #         canvas_path.append((px, py))
                
            #     converted_paths.append(canvas_path)
                
            #     # Now draw the dotted lines in canvas coords
            #     x1,y1 = self.previous_positions[i]
            #     x2, y2 = canvas_path[0]
            #     self.draw([OPERATION.dotted_line, x1, y1, x2, y2, 1, patrol_colors[i]])
            #     for j in range(len(canvas_path) - 1):
            #         x1, y1 = canvas_path[j]
            #         x2, y2 = canvas_path[j + 1]
            #         self.draw([OPERATION.dotted_line, x1, y1, x2, y2, 1, patrol_colors[i]])

            # 3) Draw trucks, rotated based on direction of travel
            for i in range(num_patrols):
                # print("Patrol", i)
                # Convert EV's position to canvas coords
                cell_x, cell_y = patrol_positions[i]
                truck_x, truck_y = cell_to_canvas(cell_x, cell_y, self.cell_size)
                dest_truck_pos = np.array([truck_x, truck_y])
                last_truck_pos = np.array(self.previous_positions[i])
                # print("Dest position", dest_truck_pos)
                # print("Last position", last_truck_pos)
                # Compute the direction vector from last position to destination
                direction_vector = dest_truck_pos - last_truck_pos

                # Compute the distance to the destination
                distance = np.linalg.norm(direction_vector)

                if distance < velocity:
                    # If the remaining distance is less than the velocity, snap to the destination
                    next_truck_pos = dest_truck_pos
                else:
                    # Normalize the direction vector and move 'vel' units along it
                    direction_unit_vector = direction_vector / distance
                    next_truck_pos = last_truck_pos + direction_unit_vector * (velocity)
                next_truck_pos = tuple(next_truck_pos)
                # print(i, next_truck_pos)
                act_truck_x, act_truck_y = next_truck_pos
                self.previous_positions[i] = next_truck_pos
                
                # # Attempt to compute rotation angle from the first segment in converted_paths[i]
                # path_canvas = converted_paths[i]
                # angle = 0
                # if len(path_canvas) > 1:
                #     # dx, dy from the first segment
                #     x_cur, y_cur = path_canvas[0]   # where the EV starts on the path
                #     x_next, y_next = path_canvas[1] # next step
                #     dx = x_next - x_cur
                #     dy = y_next - y_cur
                #     angle = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0

                transform = QTransform().rotate(0)
                rotated_truck_image = scaled_truck_image.transformed(transform, Qt.SmoothTransformation)

                # Position the truck so it's centered at (truck_x, truck_y)
                top_left_x = act_truck_x - rotated_truck_image.width() // 2
                top_left_y = act_truck_y - rotated_truck_image.height() // 2
                self.draw([OPERATION.image, top_left_x, top_left_y, rotated_truck_image])

                # 4) Draw the battery indicator
                battery_width = 30
                battery_height = 5
                battery_x = act_truck_x - battery_width // 2
                battery_y = act_truck_y + 30
                filled_width = int(R_P[i] * battery_width)

                self.draw([
                    OPERATION.filled_polygon,
                    [battery_x, battery_x + filled_width, battery_x + filled_width, battery_x],
                    [battery_y, battery_y, battery_y + battery_height, battery_y + battery_height],
                    1,
                    Qt.green
                ])
                # Battery border
                self.draw([OPERATION.line, battery_x, battery_y, battery_x + battery_width, battery_y, 1, Qt.black])
                self.draw([OPERATION.line, battery_x + battery_width, battery_y, battery_x + battery_width, battery_y + battery_height, 1, Qt.black])
                self.draw([OPERATION.line, battery_x + battery_width, battery_y + battery_height, battery_x, battery_y + battery_height, 1, Qt.black])
                self.draw([OPERATION.line, battery_x, battery_y + battery_height, battery_x, battery_y, 1, Qt.black])

            # for idx, pos in enumerate(drone_positions):
            #     x,y = cell_to_canvas(pos[0], pos[1], self.cell_size)
            #     if drone_status[idx] == 0:
            #         x += doff[idx][0]
            #         y += doff[idx][1]

            #     # Calculate top-left corner to center the image at the drone's position
            #     top_left_x = x - self.scaled_drone_image.width() // 2
            #     top_left_y = y - self.scaled_drone_image.height() // 2

            #     # Draw the rotated drone image at the calculated position
            #     self.draw([OPERATION.image, top_left_x, top_left_y, self.scaled_drone_image])
            
            self.execute()
            time.sleep(1 / T)


    def run(self):
        """
        Main loop: 
          1) Draw the maze permanently
          2) Initialize the world
          3) Repeatedly: 
             - step the simulation (simulate)
             - fetch updated state (get_world_state)
             - draw it 
             - repaint 
             - small delay
        """
        # model = PPO.load("/home/shinobi-owl/PhD/battery/DroneSim/models/8ipq56jq/lmd_model_840000_steps.zip")
        # Draw the maze walls and warehouse once
        observation, info = self.env.reset(seed=47)
        df_maze = info['maze']

        base_stations = info['base_stations']
        self.draw_maze(df_maze, base_stations)

        acts = ["Up", "Right", "Down", "Left", "Stay"]

        # # Initialize the world (assign tasks, positions, etc.)
        self.previous_positions = []

        print("Starting Simulation")
        done = False
        frame_num = 0
        total_reward = 0.0
        self.tmp = 1
        while not done:
            self.tmp = 1
            print("Frame Number:", frame_num+1)
            action = self.agent.predict(observation)
            # Step the simulation
            observation, reward, done, _, world_state = self.env.step(action)
            total_reward += reward
            frame_num += 1
            print("Action taken:", acts[action[0]])
            print("Observation:", observation)
            print("Reward:", reward)

            # Render it
            self.draw_state(world_state, 120)
            while self.tmp == 0:
                time.sleep(1)
        
        num_tasks_completed = world_state["num_tasks_completed"]
        ev_distance_traveled = world_state["ev_distance_traveled"]
        print("Simulation completed.")
        print(f"Final frame: {frame_num}")
        print(f"Tasks completed: {num_tasks_completed}")
        print(f"EV distance traveled: {ev_distance_traveled}")
        with open(self.log_file, 'a') as f:
            f.write(f"Final frame: {frame_num}\n")
            f.write(f"Tasks completed: {num_tasks_completed}\n")
            f.write(f"EV distance traveled: {ev_distance_traveled}\n")
            f.write("=============\n")

        self.env.close()
        self.close()
        

def get_color(v):
    x = v % 8
    if x == 0:
        return Qt.darkGreen
    elif x == 1:
        return QColor('#52489c')
    elif x == 2:
        return QColor('#43aa8b')
    elif x == 3:
        return Qt.darkCyan
    elif x == 4:
        return QColor('#d1ffc6')
    elif x == 5:
        return Qt.darkRed
    elif x == 6:
        return QColor('#59c3c3')
    elif x == 7:
        return QColor('#c97d60')
    elif x == 8:
        return QColor('#edc7cf')
    elif x == 9:
        return QColor('#52489c')
    else:
        return Qt.darkGray

def startup(config_file, policy_name):
    """
    Entry point that creates the Window, starts the PyQt event loop,
    and runs the simulation in a separate thread.
    """
    App = QApplication(sys.argv)
    window = Window(config_file, policy_name)
    x = threading.Thread(target=window.run, args=())
    x.start()
    sys.exit(App.exec())
