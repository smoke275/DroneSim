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
        self.env = env
        
    def predict(self, observation):
        steps2dest = observation['steps2dest']
        wall_encoding = observation['wall_encoding']
        min_steps, best_act=None,None
        for i, steps in enumerate(steps2dest):
            if i==4 or wall_encoding[i]>0:
                if min_steps is None or steps<min_steps:
                    min_steps = steps
                    best_act = i

        return best_act
        
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