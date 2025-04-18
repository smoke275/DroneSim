import numpy as np
import yaml
import os

import gymnasium as gym
# Make sure you have your environment accessible to gym or properly imported
# from your_file import LMDEnv  # e.g. if the environment is in your file

import envs  # This will register the environment
from agents.sarsa import SARSAAgent

if __name__ == "__main__":
    # Load config
    with open("config/ral.yaml", "r") as file:
        env_config = yaml.safe_load(file)

    policy_name = os.urandom(4).hex()
    root_dir = "runs/sarsa/"
    policy_dir = os.path.join(root_dir, policy_name)
    os.makedirs(policy_dir, exist_ok=True)

    config_file = os.path.join(policy_dir, "config.yaml")
    env_config['policy_name'] = policy_name
    env_config['policy_path'] = os.path.join(policy_dir, "policy.pkl")

    with open(config_file, "w") as file:
        yaml.dump(env_config, file)

    env = gym.make('LMDEnv-v0', config=env_config)

    dp_agent = SARSAAgent(env)

    dp_agent.learn(num_episodes=1000, max_steps_per_episode=1000, policy_path=env_config['policy_path'],
                   log_path=os.path.join(policy_dir, "train_log.txt"),)
    env.close()


    