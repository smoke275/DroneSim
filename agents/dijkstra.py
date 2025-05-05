import numpy as np
import networkx as nx
from agents.agent import Agent

class DijkstraAgent(Agent):
    """
    Dijkstra's Agent that finds shortest paths to tasks.
    Does not learn - simply uses Dijkstra's algorithm for pathfinding.
    """
    
    def __init__(self, env, info, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.current_path = []
        self.G = info['graph'].copy()
        self.warehouse_pos = info['warehouse_pos']
        for node in self.G.nodes():
            self.G.nodes[node]['cost2warehouse'] = nx.shortest_path_length(self.G, source=node, target=self.warehouse_pos)
        
    def predict(self, observation):
        """
        Return the next action in the path to nearest task or to the warehouse if battery is low.
        The agent checks that it has enough battery to go to a task and return to the warehouse.
        """
        ugv_pos = tuple(map(int, observation['ugv_positions']))
        battery = int(observation['battery_levels'])  # Assuming single agent control

        task_positions = observation['active_task_positions']
        # Convert task positions to list of tuples
        task_coords = tuple(task_positions.tolist()) if task_positions is not None else []

        # If no current path or reached end of path, plan a new route.
        if not self.current_path:
            valid_tasks = []
            if task_coords:
                cost_to_task = nx.shortest_path_length(self.G, source=ugv_pos, target=task_coords)
                cost_to_warehouse = self.G.nodes[task_coords]['cost2warehouse']
                total_cost = cost_to_task + cost_to_warehouse
                if battery >= total_cost:
                    valid_tasks.append((task_coords, cost_to_task))

            if valid_tasks:
                # Select the nearest valid task
                nearest_task = min(valid_tasks, key=lambda x: x[1])[0]
                self.current_path = nx.shortest_path(self.G, ugv_pos, nearest_task)[1:]
            else:
                # Not enough battery for any task route; plan a route to the warehouse.
                try:
                    self.current_path = nx.shortest_path(self.G, ugv_pos, self.warehouse_pos)[1:]
                except nx.NetworkXNoPath:
                    # No valid move available.
                    return 4  # Action 4 is "stay"
                    
        # Follow the planned path, if any.
        if self.current_path:
            next_pos = self.current_path[0]
            self.current_path = self.current_path[1:]
            
            # Convert position difference to action
            dy = next_pos[0] - ugv_pos[0]
            dx = next_pos[1] - ugv_pos[1]
            
            if dy == -1 and dx == 0:    # Up
                return 0
            elif dy == 0 and dx == 1:    # Right
                return 1
            elif dy == 1 and dx == 0:    # Down
                return 2
            elif dy == 0 and dx == -1:   # Left
                return 3
        
        return 4  # Stay in place if no valid move found
        
    def learn(self):
        """
        This agent doesn't learn - it just uses Dijkstra's algorithm
        """
        pass
        
    def _observation_to_state(self, observation):
        """
        Not used by this agent, but implemented to satisfy abstract method
        """
        pass