import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import random
import math

random.seed(47)

class World:
    def __init__(self, config):
        """
        Initializes the World with a maze loaded from CSV, the number of EVs (patrols),
        and the number of tasks to generate.
        """
        # Read the maze CSV file into a DataFrame
        self.df_maze = pd.read_csv(config['maze'])
        self.num_patrols = config['num_evs']
        self.num_tasks = config['num_tasks']

        # Extract row and column indices from the 'cell' column
        self.df_maze[['row', 'col']] = (
            self.df_maze['  cell  '].astype(str)
            .str.replace(r'[()]', '', regex=True)
            .str.split(',', expand=True)
            .astype(int)
        )

        # Find the maximum row/col for reference
        self.max_row = self.df_maze['row'].max()
        self.max_col = self.df_maze['col'].max()

        # Set the warehouse position roughly at the center
        center_row = self.max_row // 2
        center_col = self.max_col // 2
        self.warehouse_pos = (center_row, center_col)

        # Build a graph from the maze for pathfinding
        self.G = self._build_graph()

        # Store runtime info about tasks and patrols here
        self.all_tasks = []
        self.clustered_tasks = [[] for _ in range(self.num_patrols)]
        self.patrol_positions = []
        self.task_indices = []      # Tracks which task each EV is currently pursuing

        # Any other state information can go in this dictionary
        self.world_state = {
            "num_patrols": self.num_patrols,
            "patrol_positions": [],
        }

    def _build_graph(self):
        """
        Builds a graph from the maze DataFrame. Each cell is a node, and edges
        exist if E/W/N/S = 1 indicates connectivity in that direction.
        """
        G = nx.Graph()
        for _, row in self.df_maze.iterrows():
            r, c = row['row'], row['col']
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

    def initialize(self):
        """
        Creates random tasks, clusters them using k-means, and initializes the
        EV (patrol) positions at the warehouse.
        """
        # Randomly sample 'num_tasks' cells from the maze for tasks
        # all_tasks is a list of (row, col)
        self.all_tasks = self.df_maze.sample(self.num_tasks)[['row', 'col']].values.tolist()

        # Perform k-means clustering to group tasks into num_patrols clusters
        kmeans = KMeans(n_clusters=self.num_patrols, random_state=0)
        kmeans.fit(self.all_tasks)
        task_clusters = kmeans.labels_

        # Assign tasks to each patrol
        for i in range(self.num_patrols):
            self.clustered_tasks[i] = []

        for i, task in enumerate(self.all_tasks):
            cluster_idx = task_clusters[i]
            self.clustered_tasks[cluster_idx].append(tuple(task))

        # Initialize all patrols at the warehouse
        self.patrol_positions = [self.warehouse_pos for _ in range(self.num_patrols)]
        # Each patrol starts at the first task (index 0) in its cluster
        self.task_indices = [0 for _ in range(self.num_patrols)]

        # Update world_state
        self.world_state["patrol_positions"] = list(self.patrol_positions)
        self.world_state["clustered_tasks"] = self.clustered_tasks

    def simulate(self, timesteps=1):
        """
        Simulates the EV dispatch logic for the given number of timesteps.
        - For each timestep, each EV tries to move 1 step closer to its next task.
        - If it completes a task, it moves on to the next one in its cluster.
        - If no more tasks remain, it stays put (or could return to warehouse, if desired).
        """
        for _ in range(timesteps):
            for i in range(self.num_patrols):
                # If this patrol has no tasks, skip
                if len(self.clustered_tasks[i]) == 0:
                    continue

                # If we've completed all tasks in this cluster, skip
                if self.task_indices[i] >= len(self.clustered_tasks[i]):
                    continue

                current_pos = self.patrol_positions[i]
                next_task = self.clustered_tasks[i][self.task_indices[i]]

                # Compute shortest path in the maze
                path = self._compute_shortest_path(current_pos, next_task)

                if not path:
                    # No path found - handle error or skip
                    continue

                # If the patrol is already at the next task
                if current_pos == next_task:
                    # Mark task as completed, move to the next task
                    self.task_indices[i] += 1
                else:
                    # Move one step along the path
                    # path[0] should be current_pos, path[1] is the next step
                    if len(path) > 1:
                        self.patrol_positions[i] = path[1]
                    else:
                        # If the path is exactly one node, it means we're already at the task
                        self.task_indices[i] += 1

            # Update world_state after this timestep
            self.world_state["patrol_positions"] = list(self.patrol_positions)

    def _compute_shortest_path(self, start, end):
        """
        Returns the list of nodes representing the shortest path from 'start' to 'end'
        within the maze's graph G using NetworkX. If no path exists, returns None.
        """
        try:
            path = nx.shortest_path(self.G, source=start, target=end)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_world_state(self):
        """
        Returns a dictionary describing the current world state (positions, tasks, etc.).
        This can be used by other modules (e.g., for drawing or logging).
        """
        return self.world_state
