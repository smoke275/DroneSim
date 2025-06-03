import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import time # Added time for potential delay
import logging

import envs
from agents import get_agent

def setup_logging(mode, log_file="simulation_log.txt"):
    """
    Configures logging based on the mode.
    :param mode: 'logging.info' for both console and file logging, 'quiet' for file logging only.
    :param log_file: Path to the log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the base logging level

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')

    # File handler (always enabled)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (only if mode is 'logging.info')
    if mode != 'quiet':
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Adjust console log level
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def extract_observations(observation, num_agents):
    ugv_status = observation['ugv_status'].tolist()
    steps2dest = observation['steps2dest'].tolist()
    nb_traffic = observation['nb_traffic'].tolist()

    steps2dest_list = [steps2dest[i*5:(i+1)*5] for i in range(num_agents)]
    nb_traffic_list = [nb_traffic[i*4:(i+1)*4] for i in range(num_agents)]
    
    all_obs = []
    for i in range(num_agents):
        obs = {
            'ugv_status': np.array(ugv_status[i]),
            'steps2dest': np.array(steps2dest_list[i]),
            'nb_traffic': np.array(nb_traffic_list[i])
        }
        all_obs.append(obs)
    return all_obs

def lmd_simulator(config):
    if config['simulation']['train']:
        train_config = config.copy()
        train_config['fleet']['ugv']['count'] = 1

        policy_name = os.urandom(4).hex()
        root_dir = "runs/"
        policy_dir = os.path.join(root_dir, policy_name)
        policy_path = os.path.join(policy_dir, "policy.pkl")
        os.makedirs(policy_dir, exist_ok=True)
        train_config['policy_name'] = policy_name
        config_file = os.path.join(policy_dir, "config.yaml")
        train_config['simulation']['train'] = False
        with open(config_file, "w") as file:
            yaml.dump(train_config, file)
        print(f"Policy Name: {policy_name}")

        env = gym.make("MultiAgentLMDEnv-v0", config=train_config)
        agent = get_agent(train_config, env)
        agent.learn(num_episodes=200000, policy_path=policy_path,
                    log_path=os.path.join(policy_dir, "train_log.txt"),)
        env.close()
        

    env = gym.make("MultiAgentLMDEnv-v0", config=config)
    num_agents = config['fleet']['ugv']['count']
    render_mode = config['simulation']['render_mode']
    policy_name = config["policy_name"]
    policy_dir = f"runs/{policy_name}/"
    log_file = f"{policy_dir}/server_log.txt"
    open(log_file, "w").close()
    setup_logging(render_mode, log_file=log_file)

    agents = get_agent(config, env)

    logging.info("Starting Simulation")
    observation, world_state = env.reset(seed=42)
    env.render()

    acts = ["Up", "Right", "Down", "Left", "Stay"]
    terminated = False
    total_reward = 0.0
    step_idx = 0
    while not terminated:
        action_list = []
        all_obs = extract_observations(observation, num_agents)
        for i, obs in enumerate(all_obs):
            agent = agents[i]
            if not obs['ugv_status']:
                logging.info(f"UGV {i} is not active. Skipping.")
                action_list.append(4)
                continue

            # Get action from policy
            action = agent.predict(obs)
            action_list.append(action)

        observation, reward, terminated, truncated, world_state = env.step(action_list)
        total_reward += reward
        step_idx += 1

        env.render()

        # Log action and results
        logging.info(f"Observation: {observation}") # Observation can be large, maybe omit from console
        logging.info(f"Reward: {reward}")
        logging.info(f"Terminated: {terminated}, Truncated: {truncated}")
        for ugv_id in range(num_agents):
            logging.info(f"UGV {ugv_id} Info...")
            action_str = acts[action_list[ugv_id]] # Format action list for logging
            logging.info(f"Action taken: {action_str}")

        logging.info("-------------------------------------------\n")

    # Log final metrics (use the last world_state)
    num_tasks_completed = world_state["num_tasks_completed"]
    ev_distance_traveled = world_state["ev_distance_traveled"]
    total_time_taken = world_state["time_elapsed"]
    total_energy_consumed = world_state["total_energy_consumed"]
    avg_task_completion_time = world_state["avg_task_completion_time"]

    logging.info("\nSimulation completed.")
    logging.info(f"Termination reason: {'Terminated' if terminated else 'Truncated' if truncated else 'Unknown'}")
    logging.info(f"Final frame: {step_idx}")
    logging.info(f"Total reward: {total_reward}")
    logging.info(f"Tasks completed: {num_tasks_completed}")
    logging.info(f"EV distance traveled: {ev_distance_traveled}")
    logging.info(f"Total time taken: {total_time_taken}")
    logging.info(f"Total energy consumed: {total_energy_consumed}")

    env.close() # Ensure pygame resources are cleaned up
    return {
        'frames': step_idx,
        'total_reward': total_reward,
        'tasks_completed': num_tasks_completed,
        'distance_traveled': ev_distance_traveled,
        'time_taken': total_time_taken,
        'energy_consumed': total_energy_consumed,
        'avg_task_completion_time': avg_task_completion_time,
    }