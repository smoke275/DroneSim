import numpy as np
import pickle
import os
import random
from agents.agent import Agent

class SARSAAgent(Agent):
    """
    SARSA (State-Action-Reward-State-Action) Learning Agent.
    
    This on-policy TD control agent learns a Q-table via the SARSA update:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
      
    It uses the environment's reward function.
    """
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.2, epsilon_decay=0.995,
                 model_dir="runs/sarsa", policy_name=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize action list from the environment's action space
        if hasattr(env.action_space, 'n'):
            self.action_list = list(range(env.action_space.n))
        elif hasattr(env.action_space, 'nvec'):
            # Use the first dimension (assuming consistent action choices across agents)
            self.action_list = list(range(env.action_space.nvec[0]))
        else:
            raise ValueError("Unsupported action space type.")

        # Initialize Q-table as a dictionary: state -> np.array of Q-values per action.
        self.Q = {}

        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(f"{self.model_dir}/policies"):
            os.makedirs(f"{self.model_dir}/policies")
        self.policy_name = policy_name
        if self.policy_name is not None:
            with open(f"{self.model_dir}/policies/policy_{self.policy_name}.pkl", 'rb') as f:
                self.Q = pickle.load(f)
            print(f"Policy loaded from {self.policy_name}")

    def _observation_to_state(self, observation):
        # Nearest task Only
        ugv = tuple(map(int, observation['ugv_positions']))
        task_flat = observation['task_positions']
        task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)]
        # Find nearest task
        if task_coords:
            nearest_task = min(task_coords, key=lambda t: abs(ugv[0]-t[0]) + abs(ugv[1]-t[1]))
        else:
            nearest_task = (-1, -1)  # dummy if no tasks
        return (ugv, nearest_task)

        # # Distance to the nearest task
        # ugv = tuple(map(int, observation['ugv_positions']))
        # task_flat = observation['task_positions']
        # task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)]
        # if task_coords:
        #     min_dist = min([abs(ugv[0] - t[0]) + abs(ugv[1] - t[1]) for t in task_coords])
        # else:
        #     min_dist = 0
        # return (ugv, min_dist)

        # # Direction to Nearest Task
        # ugv = tuple(map(int, observation['ugv_positions']))
        # task_flat = observation['task_positions']
        # task_coords = [(int(task_flat[i]), int(task_flat[i+1])) for i in range(0, len(task_flat), 2)]

        # def direction_to(ugv, task):
        #     dy = task[0] - ugv[0]
        #     dx = task[1] - ugv[1]
        #     if abs(dy) > abs(dx):
        #         return "S" if dy > 0 else "N"
        #     elif dx != 0:
        #         return "E" if dx > 0 else "W"
        #     else:
        #         return "STAY"

        # if task_coords:
        #     nearest_task = min(task_coords, key=lambda t: abs(ugv[0]-t[0]) + abs(ugv[1]-t[1]))
        #     direction = direction_to(ugv, nearest_task)
        # else:
        #     direction = "STAY"

        # return (ugv, direction)



    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        If the state is not in the Q-table, initialize its Q-values to zeros.
        """
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
        if random.random() < self.epsilon:
            return random.choice(self.action_list)
        else:
            return int(np.argmax(self.Q[state]))

    def predict(self, observation):
        """
        For inference, return the best action (greedy with respect to Q-values)
        for the given observation.
        """
        state = self._observation_to_state(observation)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.action_list))
        return [int(np.argmax(self.Q[state]))]

    def learn(self, num_episodes=1000, max_steps_per_episode=100):
        """
        Run SARSA learning over multiple episodes. Each episode starts with an env.reset()
        and runs until done or a max number of steps is reached.
        """
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state = self._observation_to_state(obs)
            if state not in self.Q:
                self.Q[state] = np.zeros(len(self.action_list))
            action = self.choose_action(state)
            total_reward = 0.0

            for step in range(max_steps_per_episode):
                # Environment step expects an action wrapped in a list (like in DPAgent)
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

        # Save the learned policy (Q-table)
        save_file_name = f"policy_{os.urandom(4).hex()}.pkl"
        with open(f'{self.model_dir}/policies/{save_file_name}', 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Policy saved to {self.model_dir}/policies/{save_file_name}")
