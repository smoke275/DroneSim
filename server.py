import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import time # Added time for potential delay
import logging

import envs
from agents.sarsa import SARSAAgent
from agents.dijkstra import DijkstraAgent
from agents.dqn import DQNAgent
from agents.a2c import A2CAgent
from agents.sarsa_l import SARSALambdaAgent


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
    if mode == 'print':
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Adjust console log level
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def extract_observations(observation, num_agents):
    ugv_status = observation['ugv_status'].tolist()
    wall_encoding = observation['wall_encoding'].tolist()
    task_direction = observation['task_direction'].tolist()
    steps2dest = observation['steps2dest'].tolist()
    nb_traffic = observation['nb_traffic'].tolist()

    wall_encoding_list = [wall_encoding[i*4:(i+1)*4] for i in range(num_agents)]
    steps2dest_list = [steps2dest[i*5:(i+1)*5] for i in range(num_agents)]
    nb_traffic_list = [nb_traffic[i*4:(i+1)*4] for i in range(num_agents)]
    
    all_obs = []
    for i in range(num_agents):
        obs = {
            'ugv_status': np.array(ugv_status[i]),
            'wall_encoding': np.array(wall_encoding_list[i]),
            'task_direction': np.array(task_direction[i]),
            'steps2dest': np.array(steps2dest_list[i]),
            'nb_traffic': np.array(nb_traffic_list[i])
        }
        all_obs.append(obs)
    return all_obs

def run_simulation(config, render_mode='human', env_type='multi'): # Default to 'human' for visualization
    """
    Runs the simulation, optionally with GUI elements, and returns metrics
    """
    policy_name = config["policy_name"]
    policy_path = config["policy_path"]
    policy_dir = os.path.dirname(policy_path)
    log_file = f"{policy_dir}/server_log.txt"
    open(log_file, "w").close()
    setup_logging(render_mode, log_file=log_file)
    
    algo = config['algo']
    num_agents = config['world']['num_ugvs']

    if env_type == 'multi':
        env = gym.make("MultiAgentLMDEnv-v0", config=config, render_mode=render_mode)
    else:
        if num_agents > 1:
            logging.info("Warning: Using a single agent environment with multiple agents. This may not work as expected.")
            num_agents = 1
        env = gym.make("LMDEnv-v0", config=config, render_mode=render_mode)
    
    observation, info = env.reset(seed=42)

    if algo == 'sarsa':
        agents = [SARSAAgent(env, config=config, policy_path=policy_path) for _ in range(num_agents)]
    elif algo == 'dqn':
        agent = DQNAgent(env, config=config, policy_path=policy_path)
    elif algo == 'a2c':
        agent = A2CAgent(env, config=config, policy_path=policy_path)
    elif algo == 'sarsa_l':
        agent = SARSALambdaAgent(env, config=config, policy_path=policy_path)
    else:
        agents = [DijkstraAgent(i, env) for i in range(num_agents)]

    acts = ["Up", "Right", "Down", "Left", "Stay"]

    terminated = False
    truncated = False
    frame_num = 0
    total_reward = 0.0

    logging.info("Starting Simulation")
    env.render()

    # The main loop now uses terminated and truncated flags
    while not terminated:
        action_list = None
        if env_type == 'multi':
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
        else:
            action_list = agents[0].predict(observation)
            if isinstance(action_list, np.ndarray):
                action_list = action_list.tolist()

        observation, reward, terminated, truncated, world_state = env.step(action_list)
        total_reward += reward
        frame_num += 1

        env.render()

        # Log action and results
        logging.info(f"Observation: {observation}") # Observation can be large, maybe omit from console
        logging.info(f"Reward: {reward}")
        logging.info(f"Terminated: {terminated}, Truncated: {truncated}")
        if env_type == 'multi':
            for ugv_id in range(num_agents):
                logging.info(f"UGV {ugv_id} Info...")
                action_str = acts[action_list[ugv_id]] # Format action list for logging
                logging.info(f"Action taken: {action_str}")
        else:
            logging.info(f"Action taken: {acts[action_list]}")


        logging.info("-------------------------------------------\n")

    # Log final metrics (use the last world_state)
    num_tasks_completed = world_state["num_tasks_completed"]
    ev_distance_traveled = world_state["ev_distance_traveled"]
    total_time_taken = world_state["time_elapsed"]
    total_energy_consumed = world_state["total_energy_consumed"]
    avg_task_completion_time = world_state["avg_task_completion_time"]

    logging.info("\nSimulation completed.")
    logging.info(f"Termination reason: {'Terminated' if terminated else 'Truncated' if truncated else 'Unknown'}")
    logging.info(f"Final frame: {frame_num}")
    logging.info(f"Total reward: {total_reward}")
    logging.info(f"Tasks completed: {num_tasks_completed}")
    logging.info(f"EV distance traveled: {ev_distance_traveled}")
    logging.info(f"Total time taken: {total_time_taken}")
    logging.info(f"Total energy consumed: {total_energy_consumed}")


    env.close() # Ensure pygame resources are cleaned up
    return {
        'frames': frame_num,
        'total_reward': total_reward,
        'tasks_completed': num_tasks_completed,
        'distance_traveled': ev_distance_traveled,
        'time_taken': total_time_taken,
        'energy_consumed': total_energy_consumed,
        'avg_task_completion_time': avg_task_completion_time,
    }

def startup(config_file, render_mode='human', env_type='multi'): # Pass render_mode through
    """
    Entry point that runs the simulation and returns metrics
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return run_simulation(config, render_mode, env_type)
