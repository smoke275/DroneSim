import numpy as np
import yaml
import os

import gymnasium as gym
import envs
from agents.dqn import DQNAgent

if __name__ == "__main__":
    # Load config
    with open("config/ral.yaml", "r") as file:
        env_config = yaml.safe_load(file)

    policy_name = os.urandom(4).hex()
    root_dir = "runs/dqn/"
    policy_dir = os.path.join(root_dir, policy_name)
    os.makedirs(policy_dir, exist_ok=True)

    config_file = os.path.join(policy_dir, "config.yaml")
    env_config['policy_name'] = policy_name
    env_config['policy_path'] = os.path.join(policy_dir, "policy")
    env_config['algo'] = 'dqn'

    with open(config_file, "w") as file:
        yaml.dump(env_config, file)

    env = gym.make('LMDEnv-v0', config=env_config)

    dqn_agent = DQNAgent(env, env_config)

    print(f"Policy Name: {policy_name}")
    dqn_agent.learn(
        num_timesteps=1000000,
        policy_path=env_config['policy_path'],
        log_path=os.path.join(policy_dir, "train_log.txt")
    )
    env.close()
