"""Maze geometry, connectivity graph, and pathfinding. No Qt dependencies."""

import networkx as nx
import pandas as pd

from . import config


class World:
    """Loads the maze CSV and exposes cell geometry, walls, and shortest paths."""

    def __init__(self, csv_path='maze.csv'):
        df = pd.read_csv(csv_path)
        df[['row', 'col']] = (
            df['  cell  '].astype(str)
            .str.replace(r'[()]', '', regex=True)
            .str.split(',', expand=True)
            .astype(int))
        self.df = df

        self.max_row = int(df['row'].max())
        self.max_col = int(df['col'].max())
        self.cell_size = min(2 * config.BOUNDARY_X / self.max_col,
                             2 * config.BOUNDARY_Y / self.max_row)

        # Connectivity graph, built once
        self.graph = nx.Graph()
        for _, row in df.iterrows():
            cell = (row['row'], row['col'])
            self.graph.add_node(cell)
            if row['E'] == 1:
                self.graph.add_edge(cell, (row['row'], row['col'] + 1))
            if row['W'] == 1:
                self.graph.add_edge(cell, (row['row'], row['col'] - 1))
            if row['N'] == 1:
                self.graph.add_edge(cell, (row['row'] - 1, row['col']))
            if row['S'] == 1:
                self.graph.add_edge(cell, (row['row'] + 1, row['col']))

        self.warehouse_cell = (self.max_row // 2, self.max_col // 2)
        self.warehouse_pos = self.cell_to_canvas(*self.warehouse_cell)

        self.wall_segments = self._build_wall_segments()

    def cell_to_canvas(self, row, col):
        x = -config.BOUNDARY_X + (col - 1) * self.cell_size + self.cell_size / 2
        y = config.BOUNDARY_Y - (row - 1) * self.cell_size - self.cell_size / 2
        return x, y

    def canvas_to_cell(self, x, y):
        """Nearest (row, col) cell for a canvas position (inverse of cell_to_canvas)."""
        col = round((x + config.BOUNDARY_X - self.cell_size / 2) / self.cell_size) + 1
        row = round((config.BOUNDARY_Y - y - self.cell_size / 2) / self.cell_size) + 1
        return (max(1, min(self.max_row, row)), max(1, min(self.max_col, col)))

    def find_shortest_path(self, start, end):
        try:
            return nx.shortest_path(self.graph, source=tuple(start), target=tuple(end))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def sample_cells(self, n, random_state=None):
        """Return n random (row, col) cells.

        Pass random_state for reproducible draws — df.sample() uses numpy's
        RNG, which random.seed() does NOT touch."""
        return self.df.sample(n, random_state=random_state)[['row', 'col']].values.tolist()

    def _build_wall_segments(self):
        segments = []
        half = self.cell_size / 2
        processed = set()
        for _, row in self.df.iterrows():
            r, c = row['row'], row['col']
            if (r, c) in processed:
                continue
            processed.add((r, c))
            x, y = self.cell_to_canvas(r, c)
            if row['E'] == 0 and 0 < c < self.max_col and (r, c + 1) not in processed:
                segments.append((x + half, y - half, x + half, y + half))
                processed.add((r, c + 1))
            if row['W'] == 0 and 0 < c <= self.max_col and (r, c - 1) not in processed:
                segments.append((x - half, y - half, x - half, y + half))
                processed.add((r, c - 1))
            if row['N'] == 0 and 0 < r < self.max_row and (r + 1, c) not in processed:
                segments.append((x - half, y + half, x + half, y + half))
                processed.add((r + 1, c))
            if row['S'] == 0 and 0 < r <= self.max_row and (r - 1, c) not in processed:
                segments.append((x - half, y - half, x + half, y - half))
                processed.add((r - 1, c))
        return segments
