import time
# from console import startup
import argparse
from server import startup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")
    parser.add_argument('-a', '--algo', type=str, required=False)
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    parser.add_argument('-rm', '--render_mode', type=str, required=False, default='human', help="Render mode (e.g., 'human', 'rgb_array')")

    args = parser.parse_args()

    if args.policy:
        if args.algo == 'dqn':
            config_file = f"runs/dqn/{args.policy}/config.yaml"
        elif args.algo == 'sarsa':
            config_file = f"runs/sarsa/{args.policy}/config.yaml"
        elif args.algo == 'a2c':
            config_file = f"runs/a2c/{args.policy}/config.yaml"
        elif args.algo == 'sarsa_l':
            config_file = f"runs/sarsa_l/{args.policy}/config.yaml"
        else:
            raise ValueError("Invalid algorithm specified. Use 'dqn' or 'sarsa'.")
    else:
        config_file = f"runs/dijkstras/config.yaml"

    render_mode = args.render_mode

    startup(config_file, render_mode=render_mode)