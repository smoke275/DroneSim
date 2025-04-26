import os
import sys
import yaml
import gymnasium as gym
import numpy as np

import envs
from agents.sarsa import SARSAAgent
from agents.dijkstra import DijkstraAgent

def run_simulation(config_file):
    """
    Runs the simulation without GUI elements and returns metrics
    """
    # Load config
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    policy_dir = os.path.dirname(config_file)
    policy_name = config["policy_name"]
    policy_path = config["policy_path"]
    algo = config['algo']

    # Setup logging
    log_file = f"{policy_dir}/server_log.txt"
    with open(log_file, 'w') as f:
        f.write("Server Simulation Log\n")
        f.write("===================\n")

    # Initialize environment and agent
    env = gym.make("LMDEnv-v0", config=config)
    if algo[0] == 's':
        agent = SARSAAgent(env, config=config, policy_path=policy_path)
    else:
        agent = DijkstraAgent(env, config=config)

    # Run simulation
    observation, info = env.reset(seed=47)
    acts = ["Up", "Right", "Down", "Left", "Stay"]
    
    done = False
    frame_num = 0
    total_reward = 0.0

    with open(log_file, 'a') as log_f:
        print("Starting Simulation")
        log_f.write("Starting Simulation\n")

        while not done:
            print(f"Frame Number: {frame_num+1}")
            log_f.write(f"Frame Number: {frame_num+1}\n")

            # Get action from policy
            action = agent.predict(observation)

            # Step the simulation
            observation, reward, done, _, world_state = env.step(action)
            total_reward += reward
            frame_num += 1

            print(f"Action taken: {acts[action[0]]}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            log_f.write(f"Action taken: {acts[action[0]]}\n")
            log_f.write(f"Observation: {observation}\n")
            log_f.write(f"Reward: {reward}\n")

        # Log final metrics
        num_tasks_completed = world_state["num_tasks_completed"]
        ev_distance_traveled = world_state["ev_distance_traveled"]
        total_time_taken = world_state["time_elapsed"]
        total_energy_consumed = world_state["total_energy_consumed"]
        
        print("\nSimulation completed.")
        print(f"Final frame: {frame_num}")
        print(f"Total reward: {total_reward}")
        print(f"Tasks completed: {num_tasks_completed}")
        print(f"EV distance traveled: {ev_distance_traveled}")
        print(f"Total time taken: {total_time_taken}")
        print(f"Total energy consumed: {total_energy_consumed}")
        
        log_f.write("\nSimulation completed\n")
        log_f.write(f"Final frame: {frame_num}\n")
        log_f.write(f"Total reward: {total_reward}\n")
        log_f.write(f"Tasks completed: {num_tasks_completed}\n")
        log_f.write(f"EV distance traveled: {ev_distance_traveled}\n")
        log_f.write(f"Total time taken: {total_time_taken}\n")
        log_f.write(f"Total energy consumed: {total_energy_consumed}\n")

        log_f.write("===================\n")

    env.close()
    return {
        'frames': frame_num,
        'total_reward': total_reward,
        'tasks_completed': num_tasks_completed,
        'distance_traveled': ev_distance_traveled
    }

def startup(config_file):
    """
    Entry point that runs the simulation and returns metrics
    """
    return run_simulation(config_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python server.py <config_file>")
        sys.exit(1)
        
    config_file = sys.argv[1]
    metrics = startup(config_file)
    print("\nFinal Metrics:", metrics)