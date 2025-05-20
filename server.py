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

def run_simulation(config_file, render_mode='human', env_type='multi'): # Default to 'human' for visualization
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
    num_agents = config['world']['num_ugvs']

    # # Setup logging
    # log_file = f"{policy_dir}/server_log.txt"
    # with open(log_file, 'w') as f:
    #     f.write("Server Simulation Log\n")
    #     f.write("===================\n")

    if env_type == 'multi':
        env = gym.make("MultiAgentLMDEnv-v0", config=config, render_mode=render_mode)
    else:
        if num_agents > 1:
            print("Warning: Using a single agent environment with multiple agents. This may not work as expected.")
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

    if render_mode == 'human':
        env.render()
    acts = ["Up", "Right", "Down", "Left", "Stay"]

    terminated = False
    truncated = False
    frame_num = 0
    total_reward = 0.0

    # with open(log_file, 'a') as log_f:
    if True:
        print("Starting Simulation")

        # The main loop now uses terminated and truncated flags
        while not (terminated or truncated):
            print(f"Frame Number: {frame_num+1}")

            action_list = None
            if env_type == 'multi':
                action_list = []
                all_obs = extract_observations(observation, num_agents)
                for i, obs in enumerate(all_obs):
                    agent = agents[i]
                    if not obs['ugv_status']:
                        print(f"UGV {i} is not active. Skipping.")
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

            # Log action and results
            print(f"Observation: {observation}") # Observation can be large, maybe omit from console
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print("World State")
            print(f"  Tasks completed: {world_state['num_tasks_completed']}")
            print(f"  EV distance traveled: {world_state['ev_distance_traveled']}")
            print(f"  Total time taken: {world_state['time_elapsed']}")
            print(f"  Total energy consumed: {world_state['total_energy_consumed']}")
            if env_type == 'multi':
                for ugv_id in range(num_agents):
                    print(f"UGV {ugv_id} Info...")
                    action_str = acts[action_list[ugv_id]] # Format action list for logging
                    print(f"Action taken: {action_str}")
                    print(world_state['ugv_status'][ugv_id])
                    print("Active Task:", world_state['active_task'][ugv_id])
                    print(world_state['ugv_states'][ugv_id])
            else:
                print("Action taken:", acts[action_list])

            # Render the environment state if in human mode
            if render_mode == 'human':
                env.render()

            print("-------------------------------------------\n")

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


    env.close() # Ensure pygame resources are cleaned up
    return {
        'frames': frame_num,
        'total_reward': total_reward,
        'tasks_completed': num_tasks_completed,
        'distance_traveled': ev_distance_traveled
    }

def startup(config_file, render_mode='human', env_type='multi'): # Pass render_mode through
    """
    Entry point that runs the simulation and returns metrics
    """
    return run_simulation(config_file, render_mode, env_type)

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