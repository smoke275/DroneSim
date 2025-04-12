import numpy as np
import random
from collections import defaultdict
import itertools
import yaml

import gymnasium as gym
import pickle
# Make sure you have your environment accessible to gym or properly imported
# from your_file import LMDEnv  # e.g. if the environment is in your file

import envs  # This will register the environment

def obs_to_tuple(obs):
    """
    Convert the observation dictionary into a tuple, so it can be used as a dict key.
    Here, we simply flatten tasks, flatten UGV positions, and discretize battery levels.
    """
    tasks = tuple(obs['task_positions'].tolist())
    ugvs = tuple(obs['ugv_positions'].tolist())
    # Round battery levels to nearest integer for smaller state space
    battery = tuple(map(int, obs['battery_levels']))
    return (tasks, ugvs)


def make_qlearning_agent(env, 
                         num_episodes=500,
                         alpha=0.1,     # learning rate
                         gamma=0.99,    # discount factor
                         epsilon=1.0,   # exploration (start)
                         epsilon_min=0.01,
                         epsilon_decay=0.995):
    """
    A simple Q-learning loop for demonstration.
    Returns the learned Q-table (as a defaultdict).
    """
    # Q[(state, action)] -> float
    Q = defaultdict(float)
    
    # For easier notation
    num_ugvs = 1
    action_space_dims = env.action_space.nvec  # array of shape [num_ugvs], each dimension's size

    def choose_action(state):
        """
        Epsilon-greedy selection of a *joint* action (one action for each UGV),
        using a limited search for the best action in Q.
        """
        if random.random() < epsilon:
            # Random joint action
            # env.action_space.sample() returns a random array, which we convert to tuple
            return tuple(env.action_space.sample())
        else:
            # For the given state, pick the action that maximizes Q-value
            # However, enumerating the entire multi-discrete action space might be huge.
            # We'll do a partial search with a small random sample from each dimension.
            best_action = None
            best_q_val = float('-inf')

            # For each UGV, pick a small random subset of possible actions
            # e.g. up to 3 random actions per UGV
            # Combine them in a Cartesian product to see the best joint action
            subsets = []
            for dim_size in action_space_dims:
                # sample up to 3 distinct actions
                n_candidates = min(dim_size, 3)
                candidate_actions = np.random.choice(dim_size, size=n_candidates, replace=False)
                subsets.append(candidate_actions)

            for candidate_joint_action in itertools.product(*subsets):
                q_val = Q[(state, candidate_joint_action)]
                if q_val > best_q_val:
                    best_q_val = q_val
                    best_action = candidate_joint_action

            # In case we found nothing, fallback to random
            if best_action is None:
                best_action = tuple(env.action_space.sample())

            return best_action
    
    def update_Q(state, action, reward, next_state):
        """
        Update rule for Q-learning:
            Q(s,a) ← Q(s,a) + α * [ r + γ * max_a' Q(s', a') - Q(s,a) ]
        We do a partial search for max_a' Q(s', a') for next_state.
        """
        # partial search
        max_q_next = float('-inf')
        
        subsets = []
        for dim_size in action_space_dims:
            n_candidates = min(dim_size, 3)
            candidate_actions = np.random.choice(dim_size, size=n_candidates, replace=False)
            subsets.append(candidate_actions)

        for candidate_joint_action in itertools.product(*subsets):
            candidate_q = Q[(next_state, candidate_joint_action)]
            if candidate_q > max_q_next:
                max_q_next = candidate_q

        # If no action found, treat it as zero
        if max_q_next == float('-inf'):
            max_q_next = 0.0

        old_val = Q[(state, action)]
        Q[(state, action)] = old_val + alpha * (reward + gamma * max_q_next - old_val)

    print("Starting Q-learning training...")
    # Main training loop
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = obs_to_tuple(obs)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = choose_action(state)
            # Step in environment
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = obs_to_tuple(next_obs)

            # Update Q
            update_Q(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            steps += 1
            
            # if steps % 10 == 0:  # Print every 10 steps
            #     print(f"Episode {episode+1}, Step {steps}: Reward = {reward:.2f}, Total Reward = {total_reward:.2f}")

        # Decay epsilon
        nonlocal_epsilon = epsilon  # to avoid confusion in closures
        nonlocal_epsilon = max(nonlocal_epsilon*epsilon_decay, epsilon_min)
        epsilon = nonlocal_epsilon  # reassign

        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Eps: {epsilon:.3f}")

    return Q


if __name__ == "__main__":
    # Load config
    with open("config/ral.yaml", "r") as file:
        env_config = yaml.safe_load(file)

    # Create the environment
    env = gym.make('LMDEnv-v0', config=env_config)
    
    # Train a Q-learning agent
    q_table = make_qlearning_agent(env,
                                   num_episodes=200,
                                   alpha=0.1,
                                   gamma=0.99,
                                   epsilon=1.0,
                                   epsilon_min=0.05,
                                   epsilon_decay=0.98)
    
    print("Training complete!")
    env.close()

    # Save the Q-table to a file
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    print("Q-table saved to q_table.pkl")
