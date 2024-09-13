import os
import sys
import threading
from enum import Enum, auto

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QPointF, QTimer
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QPolygonF, QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyamaze import maze
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import csv
import time
import math
import numpy as np
import random

WRITE_to_file = True
SET_t = 5
SEARCH = False
END = False
CAPTURE = True

# Used to plot the example

ITERATIONS = 10000
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
    image = auto()  # Add this line for image operations


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = "Simulation"

        self.action_stack = []

        self.main_stack = []

        self.InitWindow()

        timer = QTimer(self)

        # adding action to the timer
        # update the whole code
        timer.timeout.connect(self.update)

        # setting start time of timer i.e 1 second
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
        points = [QPointF(-BOUNDARY_X, BOUNDARY_Y), QPointF(BOUNDARY_X, BOUNDARY_Y), QPointF(BOUNDARY_X, -BOUNDARY_Y),
                  QPointF(-BOUNDARY_X, -BOUNDARY_Y), QPointF(-BOUNDARY_X, BOUNDARY_Y)]
        poly = QPolygonF(points)
        painter.drawPolygon(poly)
        #
        # painter.drawLine(0, 0, 500, -500)
        sem.acquire()
        my_stack = self.main_stack
        sem.release()
        for i in my_stack:
            if i[0] == OPERATION.point:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.drawPoint(i[1], i[2])
            elif i[0] == OPERATION.line:
                painter.setPen(QPen(i[6], i[5], Qt.SolidLine))
                painter.drawLine(QPointF(i[1], i[2]), QPointF(i[3], i[4]))  # Use QPointF for float coordinates
            elif i[0] == OPERATION.dotted_line:
                painter.setPen(QPen(i[6], i[5], Qt.DotLine))
                painter.drawLine(QPointF(i[1], i[2]), QPointF(i[3], i[4]))  # Use QPointF for float coordinates
            elif i[0] == OPERATION.circle:
                painter.setPen(QPen(i[5], i[4], Qt.SolidLine))
                # painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                painter.drawEllipse(QPointF(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.filled_circle:
                painter.setPen(QPen(i[5], i[4], Qt.SolidLine))
                painter.setBrush(QBrush(i[5], Qt.SolidPattern))
                painter.drawEllipse(QPointF(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.border_circle:
                painter.setPen(QPen(Qt.black, i[4], Qt.SolidLine))
                painter.setBrush(QBrush(i[5], Qt.SolidPattern))
                painter.drawEllipse(QPointF(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.polygon:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.setBrush(QBrush(Qt.NoBrush))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(points)
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.dotted_polygon:
                painter.setPen(QPen(i[4], i[3], Qt.DotLine))
                painter.setBrush(QBrush(Qt.NoBrush))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(points)
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.filled_polygon:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.setBrush(QBrush(i[4], Qt.SolidPattern))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(points)
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.border_polygon:
                painter.setPen(QPen(Qt.black, i[3], Qt.SolidLine))
                painter.setBrush(QBrush(i[4], Qt.SolidPattern))
                x_val = i[1]
                y_val = i[2]
                points = []
                for j in range(len(x_val)):
                    points.append(QPointF(x_val[j], y_val[j]))
                poly = QPolygonF(points)
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.text:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.save()
                painter.scale(1, -1)
                painter.drawText(i[1], i[2], i[5])
                painter.restore()
            elif i[0] == OPERATION.image:
                painter.drawPixmap(int(i[1]), int(i[2]), i[3])  # Draw the image at the specified position

    def draw(self, value):
        self.action_stack.append(value)

    def execute(self):
        sem.acquire()
        self.main_stack.clear()
        self.main_stack = self.action_stack
        self.action_stack = []
        sem.release()
        self.update()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()

    def run(self):
        # Load and scale down the drone image
        drone_image = QtGui.QPixmap('transparent_drone.png')
        scaled_drone_image = drone_image.scaled(50, 50, Qt.KeepAspectRatio)  # Scale to 30x30 pixels

        # Load and scale down the truck image
        truck_image = QtGui.QPixmap('truck.png')
        scaled_truck_image = truck_image.scaled(40, 40, Qt.KeepAspectRatio)

        # Initial positions of the four drones
        drone_positions = [(0, 0), (50, 50), (-50, 50), (0, -50)]
        drone_colors = [Qt.red, Qt.blue, Qt.green, Qt.yellow]

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        # Read the CSV file into a DataFrame
        df_maze = pd.read_csv('maze.csv')

        # Data for visualization
        drone_data = []
        patrol_data = []
        base_station_data = []

        # Initialize drone targets and returning status
        drone_targets = [None] * 4
        drone_returning = [False] * 4
        active_targets = set()  # Set to keep track of currently targeted patrol agents

        # Extract row and column indices from the 'cell' column
        df_maze[['row', 'col']] = (
            df_maze['  cell  '].astype(str).str.replace(r'[()]', '', regex=True).str.split(',', expand=True).astype(
                int))

        # Find the maximum value in the 'row' column
        max_row = df_maze['row'].max()

        # Find the maximum value in the 'col' column
        max_col = df_maze['col'].max()

        # Calculate cell size to fit the maze within the canvas while maintaining aspect ratio
        canvas_width = 2 * BOUNDARY_X
        canvas_height = 2 * BOUNDARY_Y
        maze_width = max_col
        maze_height = max_row
        cell_size = min(canvas_width / maze_width, canvas_height / maze_height)

        # Calculate actual distance per cell
        distance_per_cell = cell_size

        # Update the cell_to_canvas function to use the calculated cell_size
        def cell_to_canvas(row, col):
            x = -BOUNDARY_X + (col - 1) * cell_size + cell_size / 2
            y = BOUNDARY_Y - (row - 1) * cell_size - cell_size / 2
            return x, y

        def find_shortest_path(maze_df, start, end):
            """
            Finds the shortest path between two cells in a maze represented by a DataFrame.

            Args:
                maze_df: DataFrame containing maze information with columns 'row', 'col', 'E', 'W', 'N', 'S'.
                start: Tuple representing the starting cell coordinates (row, col).
                end: Tuple representing the ending cell coordinates (row, col).

            Returns:
                List of cell coordinates representing the shortest path if a path exists, or None otherwise.
            """

            # Create a graph representation of the maze
            G = nx.Graph()
            for _, row in maze_df.iterrows():
                cell = (row['row'], row['col'])
                G.add_node(cell)
                if row['E'] == 1:
                    G.add_edge(cell, (row['row'], row['col'] + 1))
                if row['W'] == 1:
                    G.add_edge(cell, (row['row'], row['col'] - 1))
                if row['N'] == 1:
                    G.add_edge(cell, (row['row'] - 1, row['col']))
                if row['S'] == 1:
                    G.add_edge(cell, (row['row'] + 1, row['col']))

            # Find the shortest path using BFS
            try:
                path = nx.shortest_path(G, source=start, target=end)
            except nx.NetworkXNoPath:
                return None

            # # Convert cell coordinates to canvas coordinates
            # shortest_path_canvas = [cell_to_canvas(cell[0], cell[1]) for cell in path]

            return path

        # Sample 4 unique rows from the DataFrame
        selected_cells = df_maze.sample(4)

        # Extract the 'row' and 'col' values from the selected cells
        selected_rows = selected_cells['row'].tolist()
        selected_cols = selected_cells['col'].tolist()

        # Apply the cell_to_canvas function to each pair of 'row' and 'col'
        base_stations = [cell_to_canvas(row, col) for row, col in zip(selected_rows, selected_cols)]

        # Print the base_stations
        print("Base Stations:", base_stations)

        base_station_colors = [Qt.gray, Qt.cyan, Qt.magenta, Qt.darkYellow]

        num_patrols = 6

        # Define range and distance variables
        R = 3000  # Initial range of patrol agents in kilometers

        # Initialize remaining range for each patrol agent
        R_D = [R] * num_patrols

        # Find the centermost cell
        center_row = max_row // 2
        center_col = max_col // 2
        warehouse_x, warehouse_y = cell_to_canvas(center_row, center_col)

        # Generate 25-30 random cells as tasks
        all_tasks = df_maze.sample(random.randint(35, 40))[['row', 'col']].values.tolist()

        # Perform k-means clustering to group tasks into num_patrols clusters
        kmeans = KMeans(n_clusters=num_patrols, random_state=0).fit(all_tasks)
        task_clusters = kmeans.labels_

        # Assign tasks to patrol agents based on cluster labels
        patrol_paths = [[] for _ in range(num_patrols)]
        initial_patrol_tasks = [[] for _ in range(num_patrols)]
        for i, task in enumerate(all_tasks):
            patrol_paths[task_clusters[i]].append(task)
            initial_patrol_tasks[task_clusters[i]].append(tuple(task))

        num_patrols = len(patrol_paths)

        # Update the initial patrol positions to be at the warehouse
        patrol_positions = [(warehouse_x, warehouse_y) for _ in range(num_patrols)]

        patrol_path_indices = [0] * num_patrols

        # Initialize variables for dynamic pathfinding
        current_patrol_paths = [None] * num_patrols
        current_patrol_paths_canvas = [None] * num_patrols
        completed_tasks = [set() for _ in range(num_patrols)]
        needs_path_update = [True] * num_patrols  # Flag to track path recomputation
        current_task_index = [0] * num_patrols  # New variable to track the current task index

        # Convert patrol paths to canvas coordinates
        patrol_paths_canvas = []
        for path in patrol_paths:
            patrol_paths_canvas.append([cell_to_canvas(cell[0], cell[1]) for cell in path])

        patrol_colors = [Qt.darkCyan, Qt.darkMagenta, Qt.darkRed, Qt.darkGreen, Qt.darkBlue,
                         Qt.darkYellow, Qt.darkGray, Qt.black]

        # Initialize a flag to track if each patrol is at the warehouse
        at_warehouse = [True] * num_patrols

        # Drone flight range
        Z = 2000  # Adjust as needed

        # Initial positions of the drones (now associated with base stations)
        drones_per_station = 2  # Number of drones per base station
        drone_positions = []
        drone_colors = []
        drone_home_stations = []
        drone_remaining_ranges = []
        for i in range(4):
            for _ in range(drones_per_station):
                # Position drones slightly offset from their base stations
                x, y = base_stations[i]
                offset_x, offset_y = random.uniform(-10, 10), random.uniform(-10, 10)
                drone_positions.append([x + offset_x, y + offset_y])  # Use a list instead of a tuple
                drone_colors.append(Qt.red)  # You can add more colors if needed
                drone_home_stations.append(i)  # Store the index of the home station
                drone_remaining_ranges.append(Z)  # Initialize remaining range

        # Initialize drone targets and returning status
        drone_targets = [None] * len(drone_positions)
        drone_returning = [False] * len(drone_positions)
        active_targets = set()

        for frame in range(ITERATIONS):
            # Draw the maze walls (modified to avoid redundancy)
            processed_cells = set()  # Keep track of processed cells
            for index, row in df_maze.iterrows():
                row_idx, col_idx = row['row'], row['col']
                if (row_idx, col_idx) in processed_cells:
                    continue  # Skip if already processed

                processed_cells.add((row_idx, col_idx))
                x, y = cell_to_canvas(row_idx, col_idx)

                wall_thickness = 3

                if row['E'] == 0:  # East wall
                    if 0 < col_idx < max_col and (row_idx, col_idx + 1) not in processed_cells:
                        self.draw([OPERATION.line, x + cell_size / 2, y - cell_size / 2,
                                   x + cell_size / 2, y + cell_size / 2, wall_thickness, Qt.black])
                        processed_cells.add((row_idx, col_idx + 1))
                if row['W'] == 0:  # West wall
                    if 0 < col_idx <= max_col and (row_idx, col_idx - 1) not in processed_cells:
                        self.draw([OPERATION.line, x - cell_size / 2, y - cell_size / 2,
                                   x - cell_size / 2, y + cell_size / 2, wall_thickness, Qt.black])
                        processed_cells.add((row_idx, col_idx - 1))
                if row['N'] == 0:  # North wall
                    if 0 < row_idx < max_row and (row_idx + 1, col_idx) not in processed_cells:
                        self.draw([OPERATION.line, x - cell_size / 2, y + cell_size / 2,
                                   x + cell_size / 2, y + cell_size / 2, wall_thickness, Qt.black])
                        processed_cells.add((row_idx + 1, col_idx))
                if row['S'] == 0:  # South wall
                    if 0 < row_idx <= max_row and (row_idx - 1, col_idx) not in processed_cells:
                        self.draw([OPERATION.line, x - cell_size / 2, y - cell_size / 2,
                                   x + cell_size / 2, y - cell_size / 2, wall_thickness, Qt.black])
                        processed_cells.add((row_idx - 1, col_idx))

            # Highlight the initial cells chosen for the patrol points as tasks (Keep this in the loop)
            size = 5  # Adjust the size as needed
            for i, path in enumerate(initial_patrol_tasks):  # Iterate through initial_patrol_tasks
                color = patrol_colors[i]
                for cell in path:
                    if tuple(cell) in completed_tasks[i]:  # Skip highlighting if task is completed
                        continue
                    x, y = cell_to_canvas(cell[0], cell[1])
                    self.draw([OPERATION.filled_polygon,
                               [x - size / 2, x + size / 2, x + size / 2, x - size / 2],
                               [y - size / 2, y - size / 2, y + size / 2, y + size / 2],
                               1, color])

            # Draw the warehouse
            warehouse_size = 15
            self.draw([OPERATION.filled_circle, warehouse_x, warehouse_y, warehouse_size, 1,
                       Qt.blue])  # Blue warehouse

            # Draw base stations using self.draw
            for i in range(4):
                x, y = base_stations[i]
                size = 20
                self.draw([OPERATION.filled_polygon,
                           [x - size / 2, x + size / 2, x + size / 2, x - size / 2],
                           [y - size / 2, y - size / 2, y + size / 2, y + size / 2],
                           1, base_station_colors[i]])

            # Draw paths for each patrol (modified)
            for i in range(num_patrols):
                if current_patrol_paths_canvas[i]:
                    path = current_patrol_paths_canvas[i]
                    for j in range(len(path) - 1):
                        x1, y1 = path[j]
                        x2, y2 = path[j + 1]
                        self.draw([OPERATION.dotted_line, x1, y1, x2, y2, 1, patrol_colors[i]])

            # Draw patrol objects as trucks (with rotation)
            for i in range(num_patrols):
                x, y = patrol_positions[i]
                # Modified condition to include check for warehouse position
                if current_patrol_paths_canvas[i]:
                    path = current_patrol_paths_canvas[i]
                    current_index = patrol_path_indices[i]
                    next_index = (current_index + 1) % len(path)

                    dx = path[next_index][0] - path[current_index][0]
                    dy = path[next_index][1] - path[current_index][1]

                    angle = math.degrees(math.atan2(dy, dx))

                    transform = QTransform().rotate(angle)
                    rotated_truck_image = scaled_truck_image.transformed(transform, Qt.SmoothTransformation)

                    top_left_x = x - rotated_truck_image.width() // 2
                    top_left_y = y - rotated_truck_image.height() // 2

                    self.draw([OPERATION.image, top_left_x, top_left_y, rotated_truck_image])

                    # Draw battery indicator (remains within the modified condition)
                    battery_width = 30
                    battery_height = 5
                    battery_x = patrol_positions[i][0] - battery_width // 2
                    battery_y = patrol_positions[i][1] + 30
                    filled_width = int(R_D[i] / R * battery_width)

                    self.draw([OPERATION.filled_polygon,
                               [battery_x, battery_x + filled_width, battery_x + filled_width, battery_x],
                               [battery_y, battery_y, battery_y + battery_height, battery_y + battery_height],
                               1, Qt.green])

                    self.draw([OPERATION.line, battery_x, battery_y, battery_x + battery_width, battery_y, 1,
                               Qt.black])
                    self.draw([OPERATION.line, battery_x + battery_width, battery_y, battery_x + battery_width,
                               battery_y + battery_height, 1, Qt.black])
                    self.draw([OPERATION.line, battery_x + battery_width, battery_y + battery_height, battery_x,
                               battery_y + battery_height, 1, Qt.black])
                    self.draw([OPERATION.line, battery_x, battery_y + battery_height, battery_x, battery_y, 1,
                               Qt.black])


            # Drone behavior (modified)
            for i in range(len(drone_positions)):
                x, y = drone_positions[i]
                home_station = drone_home_stations[i]

                if drone_targets[i] is not None and not drone_returning[i]:
                    # Calculate the direction to the target patrol agent
                    target_x, target_y = patrol_positions[drone_targets[i]]
                    dx = target_x - x
                    dy = target_y - y

                    # Update remaining range
                    distance_to_target = math.sqrt(dx ** 2 + dy ** 2)
                    drone_remaining_ranges[i] -= distance_to_target

                    # Check if drone needs to return
                    if drone_remaining_ranges[i] < Z / 2:
                        drone_returning[i] = True
                        active_targets.remove(drone_targets[i])
                        drone_targets[i] = None

                    if distance_to_target > 5:
                        # Modify the list elements directly
                        drone_positions[i][0] += 3 * dx / distance_to_target
                        drone_positions[i][1] += 3 * dy / distance_to_target

                elif drone_returning[i]:
                    # Calculate the direction to the home base station
                    nearest_station = base_stations[home_station]
                    dx = nearest_station[0] - x
                    dy = nearest_station[1] - y

                    # Update remaining range
                    distance_to_station = math.sqrt(dx ** 2 + dy ** 2)
                    drone_remaining_ranges[i] -= distance_to_station

                    # Check if drone reached the station
                    if distance_to_station < 3:
                        drone_returning[i] = False
                        drone_remaining_ranges[i] = Z  # Recharge

                    if distance_to_station > 3:
                        # Modify the list elements directly
                        drone_positions[i][0] += 3 * dx / distance_to_station
                        drone_positions[i][1] += 3 * dy / distance_to_station
                else:
                    dx = dy = 0

                # Calculate the angle of rotation in degrees
                angle = math.degrees(math.atan2(dy, dx))

                # Apply rotation to the drone image
                transform = QTransform().rotate(angle)
                rotated_drone_image = scaled_drone_image.transformed(transform, Qt.SmoothTransformation)

                # Calculate top-left corner to center the image at the drone's position
                top_left_x = x - rotated_drone_image.width() // 2
                top_left_y = y - rotated_drone_image.height() // 2

                # Draw the rotated drone image at the calculated position
                self.draw([OPERATION.image, top_left_x, top_left_y, rotated_drone_image])

                # Display drone status text at the bottom
                status_text = f'Moving to Patrol {drone_targets[i]} ' if not drone_returning[
                    i] else "Returning to Base"
                self.draw(
                    [OPERATION.text, -BOUNDARY_X + 10 + i * 150, -BOUNDARY_Y + 20, 1, drone_colors[0],
                     status_text])


            # Update patrol object positions and paths
            for i in range(num_patrols):
                if needs_path_update[i]:
                    if at_warehouse[i]:  # Truck is at the warehouse
                        # Find the next uncompleted task using current_task_index
                        while current_task_index[i] < len(initial_patrol_tasks[i]) and \
                                tuple(initial_patrol_tasks[i][current_task_index[i]]) in completed_tasks[i]:
                            current_task_index[i] += 1

                        if current_task_index[i] < len(initial_patrol_tasks[i]):
                            next_task = initial_patrol_tasks[i][current_task_index[i]]

                            # Calculate the path to the next task from the warehouse
                            current_patrol_paths[i] = find_shortest_path(df_maze, (center_row, center_col),
                                                                         tuple(next_task))
                            if current_patrol_paths[i]:
                                current_patrol_paths_canvas[i] = [cell_to_canvas(cell[0], cell[1]) for cell in
                                                                  current_patrol_paths[i]]

                                # Calculate distances for fuel check
                                distance_next_task = (len(current_patrol_paths[i]) - 1) * distance_per_cell
                                shortest_path_back = find_shortest_path(df_maze, tuple(next_task),
                                                                        (center_row, center_col))
                                distance_back_to_warehouse = (
                                                                         len(shortest_path_back) - 1) * distance_per_cell if shortest_path_back else 0

                                # Check if enough fuel to go to the task and return
                                if R_D[i] >= distance_next_task + distance_back_to_warehouse:
                                    patrol_path_indices[i] = 0  # Start moving towards the task
                                    at_warehouse[i] = False
                                else:
                                    current_patrol_paths[i] = None  # Stay at the warehouse
                                    current_patrol_paths_canvas[i] = None

                    else:  # Truck is at a task
                        # Access the current task using current_task_index
                        current_task = patrol_paths[i][current_task_index[i]-1]

                        if len(initial_patrol_tasks[i]) > len(completed_tasks[i]):
                            # Find the next uncompleted task
                            next_task = next(
                                task for task in initial_patrol_tasks[i] if tuple(task) not in completed_tasks[i])

                            # Calculate the path to the next task from the current task
                            current_patrol_paths[i] = find_shortest_path(df_maze, tuple(current_task), tuple(next_task))
                            if current_patrol_paths[i]:
                                current_patrol_paths_canvas[i] = [cell_to_canvas(cell[0], cell[1]) for cell in
                                                                  current_patrol_paths[i]]

                                # Calculate distances for fuel check
                                distance_next_task = (len(current_patrol_paths[i]) - 1) * distance_per_cell
                                shortest_path_back = find_shortest_path(df_maze, tuple(next_task),
                                                                        (center_row, center_col))
                                distance_back_to_warehouse = (
                                                                         len(shortest_path_back) - 1) * distance_per_cell if shortest_path_back else 0

                                # Check if enough fuel to go to the next task and return to the warehouse
                                if R_D[i] >= distance_next_task + distance_back_to_warehouse:
                                    patrol_path_indices[i] = 0
                                else:
                                    # Return to warehouse if not enough fuel
                                    current_patrol_paths[i] = find_shortest_path(df_maze, tuple(current_task),
                                                                                 (center_row, center_col))
                                    if current_patrol_paths[i]:
                                        current_patrol_paths_canvas[i] = [cell_to_canvas(cell[0], cell[1]) for cell in
                                                                          current_patrol_paths[i]]
                                        patrol_path_indices[i] = 0

                        else:  # No more tasks, return to the warehouse
                            current_patrol_paths[i] = find_shortest_path(df_maze, tuple(current_task),
                                                                         (center_row, center_col))
                            if current_patrol_paths[i]:
                                current_patrol_paths_canvas[i] = [cell_to_canvas(cell[0], cell[1]) for cell in
                                                                  current_patrol_paths[i]]
                                patrol_path_indices[i] = 0

                    needs_path_update[i] = False

                # Movement and fuel consumption
                if current_patrol_paths_canvas[i]:
                    path = current_patrol_paths_canvas[i]
                    current_index = patrol_path_indices[i]
                    next_index = (current_index + 1) % len(path)

                    dx = path[next_index][0] - patrol_positions[i][0]
                    dy = path[next_index][1] - patrol_positions[i][1]
                    distance = math.sqrt(dx ** 2 + dy ** 2)

                    prev_x, prev_y = patrol_positions[i]

                    if distance > 1:
                        patrol_positions[i] = (
                            patrol_positions[i][0] + dx / distance,
                            patrol_positions[i][1] + dy / distance
                        )
                        distance_traveled = math.sqrt((patrol_positions[i][0] - prev_x) ** 2 + (
                                patrol_positions[i][1] - prev_y) ** 2)
                        R_D[i] -= distance_traveled
                    else:
                        patrol_path_indices[i] = next_index

                        # Check if reached destination
                        if patrol_path_indices[i] == len(path) - 1:
                            if current_patrol_paths[i][-1] == (center_row, center_col):  # Reached warehouse
                                R_D[i] = R  # Replenish fuel
                                current_patrol_paths[i] = None
                                current_patrol_paths_canvas[i] = None
                                patrol_path_indices[i] = 0
                                needs_path_update[i] = True  # Need to recompute path after refueling
                                at_warehouse[i] = True
                            else:  # Reached a task
                                completed_tasks[i].add(tuple(current_patrol_paths[i][-1]))
                                current_task_index[i] += 1  # Increment the task index
                                current_patrol_paths[i] = None
                                current_patrol_paths_canvas[i] = None
                                patrol_path_indices[i] = 0
                                needs_path_update[i] = True  # Need to recompute path after completing a task
                                at_warehouse[i] = False
                                if random.random() < 0.9 and i not in active_targets:  # Truck reached a task and requests a drone
                                    # Find the nearest available drone within range
                                    nearest_drone = None
                                    min_distance = float('inf')
                                    for j, (drone_x, drone_y) in enumerate(drone_positions):
                                        if drone_targets[j] is None and drone_remaining_ranges[j] >= Z / 2:
                                            distance = math.hypot(drone_x - patrol_positions[i][0],
                                                                  drone_y - patrol_positions[i][1])
                                            if distance < min_distance:
                                                min_distance = distance
                                                nearest_drone = j

                                    if nearest_drone is not None:
                                        drone_targets[nearest_drone] = i
                                        active_targets.add(i)

            self.execute()
            time.sleep(1 / 120)


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


def startup():
    App = QApplication(sys.argv)
    window = Window()
    # window.run()
    x = threading.Thread(target=window.run, args=())
    x.start()
    sys.exit(App.exec())
