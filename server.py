import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import time # Added time for potential delay

import envs
from agents.sarsa import SARSAAgent
from agents.dijkstra import DijkstraAgent
from agents.dqn import DQNAgent
from agents.a2c import A2CAgent
from agents.sarsa_l import SARSALambdaAgent

def run_simulation(config_file, render_mode='human'): # Default to 'human' for visualization
    """
    Runs the simulation, optionally with GUI elements, and returns metrics
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
    # Pass the desired render_mode to the environment
    env = gym.make("LMDEnv-v0", config=config, render_mode=render_mode)
    # Run simulation
    observation, info = env.reset(seed=42)
    if algo == 'sarsa':
        agent = SARSAAgent(env, config=config, policy_path=policy_path)
    elif algo == 'dqn':
        agent = DQNAgent(env, config=config, policy_path=policy_path)
    elif algo == 'a2c':
        agent = A2CAgent(env, config=config, policy_path=policy_path)
    elif algo == 'sarsa_l':
        agent = SARSALambdaAgent(env, config=config, policy_path=policy_path)
    else:
        agent = DijkstraAgent(env, info)

    if render_mode == 'human':
        env.render()
    acts = ["Up", "Right", "Down", "Left", "Stay"]

    terminated = False
    truncated = False
    frame_num = 0
    total_reward = 0.0

    with open(log_file, 'a') as log_f:
        print("Starting Simulation")
        log_f.write("Starting Simulation\n")

        # The main loop now uses terminated and truncated flags
        while not (terminated or truncated):
            print(f"Frame Number: {frame_num+1}")
            log_f.write(f"Frame Number: {frame_num+1}\n")

            # Get action from policy
            action = agent.predict(observation)
            # Ensure action format matches env.action_space (MultiDiscrete)
            if isinstance(action, np.ndarray):
                action = action.tolist() # Convert numpy array to list if needed

            # Step the simulation - returns 5 values now
            observation, reward, terminated, truncated, world_state = env.step(action)
            total_reward += reward
            frame_num += 1

            # Render the environment state if in human mode
            if render_mode == 'human':
                env.render()
                # Optional: Add a small delay to control speed if needed
                # time.sleep(0.05)

            # Log action and results
            action_str = acts[action] # Format action list for logging
            print(f"Action taken: [{action_str}]")
            print(f"Observation: {observation}") # Observation can be large, maybe omit from console
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            log_f.write(f"Action taken: [{action_str}]\n")
            # log_f.write(f"Observation: {observation}\n") # Avoid logging large observations
            log_f.write(f"Reward: {reward}\n")
            log_f.write(f"Terminated: {terminated}, Truncated: {truncated}\n")

            # Check if the render window was closed (pygame event handling is inside env.render)
            # If env.render() needs to signal closure, it might need modification,
            # or we check a flag set by it. For now, assume loop breaks on terminated/truncated.


        # Log final metrics (use the last world_state)
        num_tasks_completed = world_state["num_tasks_completed"]
        ev_distance_traveled = world_state["ev_distance_traveled"]
        total_time_taken = world_state["time_elapsed"]
        total_energy_consumed = world_state["total_energy_consumed"]

        print("\nSimulation completed.")
        print(f"Termination reason: {'Terminated' if terminated else 'Truncated' if truncated else 'Unknown'}")
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

    env.close() # Ensure pygame resources are cleaned up
    return {
        'frames': frame_num,
        'total_reward': total_reward,
        'tasks_completed': num_tasks_completed,
        'distance_traveled': ev_distance_traveled
    }

def startup(config_file, render_mode='human'): # Pass render_mode through
    """
    Entry point that runs the simulation and returns metrics
    """
    return run_simulation(config_file, render_mode)

if __name__ == "__main__":
    if len(sys.argv) < 2: # Allow optional render mode argument
        print("Usage: python server.py <config_file> [render_mode]")
        print("  render_mode (optional): 'human', 'print', 'rgb_array', or None (default: human)")
        sys.exit(1)

    config_file = sys.argv[1]
    render_arg = 'human' # Default render mode
    if len(sys.argv) > 2:
        render_arg = sys.argv[2]
        if render_arg.lower() == 'none':
             render_arg = None # Handle 'none' string case

    metrics = startup(config_file, render_mode=render_arg)
    print("\nFinal Metrics:", metrics)