import numpy as np
import yaml

import gymnasium as gym
# Make sure you have your environment accessible to gym or properly imported
# from your_file import LMDEnv  # e.g. if the environment is in your file

import envs  # This will register the environment
from agents.sarsa import SARSAAgent

if __name__ == "__main__":
    # Load config
    with open("config/ral.yaml", "r") as file:
        env_config = yaml.safe_load(file)

    # Create the environment
    env = gym.make('LMDEnv-v0', config=env_config)

    # Initialize the DP agent
    dp_agent = SARSAAgent(env)

    dp_agent.learn()

    env.close()


    