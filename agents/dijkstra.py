import numpy as np
import networkx as nx
from agents.agent import Agent

class DijkstraAgent(Agent):
    """
    Dijkstra's Agent that finds shortest paths to tasks.
    Does not learn - simply uses Dijkstra's algorithm for pathfinding.
    """
    
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.G = None
        self.current_path = []
        _, info = self.env.reset()
        self.G = info['graph']
        
    def predict(self, observation):
        """
        Return the next action in the path to nearest task.
        If no path exists, generates new path to nearest task.
        """
        ugv_pos = tuple(map(int, observation['ugv_positions']))
        task_positions = observation['task_positions']
        
        # Convert task positions to list of tuples
        task_coords = []
        for i in range(0, len(task_positions), 2):
            task_coords.append((int(task_positions[i]), int(task_positions[i+1])))
            
        # If no current path or reached end of path, find new path
        if not self.current_path:
            if task_coords:
                # Find nearest task using network distance
                nearest_task = min(task_coords, 
                                 key=lambda t: nx.shortest_path_length(self.G, 
                                                                     source=ugv_pos, 
                                                                     target=t))
                # Get shortest path
                self.current_path = nx.shortest_path(self.G, ugv_pos, nearest_task)[1:]
            else:
                # No tasks available - stay in place
                return [4]  # Action 4 is "stay"
                
        # If we have a path, determine next action to follow it
        if self.current_path:
            next_pos = self.current_path[0]
            self.current_path = self.current_path[1:]
            
            # Convert position difference to action
            dy = next_pos[0] - ugv_pos[0]
            dx = next_pos[1] - ugv_pos[1]
            
            if dy == -1 and dx == 0:    # Up
                return [0]
            elif dy == 0 and dx == 1:    # Right
                return [1]
            elif dy == 1 and dx == 0:    # Down
                return [2]
            elif dy == 0 and dx == -1:   # Left
                return [3]
        
        return [4]  # Stay in place if no valid move found
        
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