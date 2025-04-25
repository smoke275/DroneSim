import numpy as np
import pickle
import os
import random
from agents.agent import Agent
import networkx as nx

class SARSAAgent(Agent):
    """
    SARSA (State-Action-Reward-State-Action) Learning Agent.
    
    This on-policy TD control agent learns a Q-table via the SARSA update:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
      
    It uses the environment's reward function.
    """
    def __init__(self, env, config, gamma=0.99, alpha=0.1, epsilon=0.2, epsilon_decay=0.995,
                 policy_path=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.G = None
        _, info = self.env.reset()
        self.G = info['graph']
        self.cell_size = config['world']['cell_size']
        self.max_range = config['ugv']['range']
        self.max_cell_range = int(self.max_range/self.cell_size)
        self.max_rows = config['world']['maze_size']
        self.max_cols = self.max_rows
        self.max_load = config['ugv']['max_load']

        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        self.Q = {}

        if policy_path is not None:
            with open(policy_path, 'rb') as f:
                self.Q = pickle.load(f)
            print(f"Policy loaded from {policy_path}")

    def _observation_to_state(self, observation):
        # # Nearest task Only
        # ugv = tuple(map(int, observation['ugv_positions']))
        # task_flat = observation['task_positions']
        # task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)]
        # # Find nearest task
        # if task_coords:
        #     nearest_task = min(task_coords, key=lambda t: abs(ugv[0]-t[0]) + abs(ugv[1]-t[1]))
        # else:
        #     nearest_task = (-1, -1)  # dummy if no tasks
        # return (ugv, nearest_task)

        # # Distance to the nearest task
        # ugv = tuple(map(int, observation['ugv_positions']))
        # task_flat = observation['task_positions']
        # task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)]
        # if task_coords:
        #     min_dist = min([abs(ugv[0] - t[0]) + abs(ugv[1] - t[1]) for t in task_coords])
        # else:
        #     min_dist = 0
        # return (ugv, min_dist)

        # # Graph Nearest task Only
        # ugv = tuple(map(int, observation['ugv_positions']))
        # task_flat = observation['task_positions']
        # task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)]
        # # Find nearest task
        # if task_coords:
        #     nearest_task = min(task_coords, key=lambda t: nx.shortest_path_length(self.G, source=ugv, target=t))
        # else:
        #     nearest_task = (-1, -1)  # dummy if no tasks
        # return (ugv, nearest_task)

        # # Graph Nearest Direction Only
        # ugv = tuple(map(int, observation['ugv_positions']))
        # task_flat = observation['task_positions']
        # task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)]
        # def direction_to(ugv, task):
        #     # 8 directions
        #     dy = task[0] - ugv[0]
        #     dx = task[1] - ugv[1]
        #     if dy == 1 and dx == 0:
        #         return "S"
        #     elif dy == -1 and dx == 0:
        #         return "N"
        #     elif dy == 0 and dx == 1:
        #         return "E"
        #     elif dy == 0 and dx == -1:
        #         return "W"
        #     elif dy == 1 and dx == 1:
        #         return "SE"
        #     elif dy == 1 and dx == -1:
        #         return "SW"
        #     elif dy == -1 and dx == 1:
        #         return "NE"
        #     elif dy == -1 and dx == -1:
        #         return "NW"
        #     else:
        #         return "STAY" 
        # # Find nearest task
        # if task_coords:
        #     nearest_task = min(task_coords, key=lambda t: nx.shortest_path_length(self.G, source=ugv, target=t))
        #     direction = direction_to(ugv, nearest_task)
        # else:
        #     nearest_task = (-1, -1)  # dummy if no tasks
        #     direction = "STAY"
        # return (ugv, direction)

        # Graph Nearest task  and battery
        ugv = tuple(map(int, observation['ugv_positions']))
        task_flat = observation['task_positions']
        task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)] 
        if task_coords:
            nearest_task = min(task_coords, key=lambda t: nx.shortest_path_length(self.G, source=ugv, target=t))
        else:
            nearest_task = (-1, -1)  # dummy if no tasks
        batt_range = float(observation['battery_levels'][0])
        cell_range = int(batt_range/self.cell_size)
        ceil_range = self.max_cols+self.max_rows
        band_size = ceil_range/4
        band = min(int(cell_range // band_size), 4)
        ugv_load = int(observation['ugv_loads'][0])
        load_band_size = self.max_load/5
        load_band = min(int(ugv_load // load_band_size), 5)
        nb_traffic = tuple([int(i) for i in observation['nb_traffic']])
        return (ugv, nearest_task, band, load_band, nb_traffic)

    def predict(self, observation):
        """
        For inference, return the best action (greedy with respect to Q-values)
        for the given observation.
        """
        state = self._observation_to_state(observation)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
        return [int(np.argmax(self.Q[state]))]
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        If the state is not in the Q-table, initialize its Q-values to zeros.
        """
        if random.random() < self.epsilon:
            return random.choice(self.action_list)
        else:
            return int(np.argmax(self.Q[state]))

    def learn(self, num_episodes=1000, max_steps_per_episode=100, policy_path=None, log_path=None):
        """
        Run SARSA learning over multiple episodes. Each episode starts with an env.reset()
        and runs until done or a max number of steps is reached.
        """
        results = []
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state = self._observation_to_state(obs)
            if state not in self.Q:
                self.Q[state] = np.zeros(len(self.action_list))
            action = self.choose_action(state)
            total_reward = 0.0

            for step in range(max_steps_per_episode):
                next_obs, reward, done, trunc, info = self.env.step([action])
                next_state = self._observation_to_state(next_obs)
                if next_state not in self.Q:
                    self.Q[next_state] = np.zeros(len(self.action_list))
                next_action = self.choose_action(next_state)

                # SARSA update rule:
                td_target = reward + self.gamma * self.Q[next_state][next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                state = next_state
                action = next_action
                total_reward += reward

                if done or trunc:
                    break

            self.epsilon *= self.epsilon_decay
            print(f"Episode {episode+1}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {self.epsilon:.4f}")
            results.append((total_reward, self.epsilon))

            if episode%100 == 0:
                with open(policy_path, 'wb') as f:
                    pickle.dump(self.Q, f)

        if log_path is not None:
            with open(log_path, 'a') as f:
                f.write("Episode, Total Reward, Epsilon\n")
                i=0
                for total_reward, epsilon in results:
                    f.write(f"{i+1},{total_reward:.2f}, {epsilon:.4f}\n")
                    i += 1
        with open(policy_path, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Policy saved to {policy_path}")
