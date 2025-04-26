import time
from console import startup
import argparse
# from server import startup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")

    args = parser.parse_args()

    if args.policy:
        config_file = f"runs/sarsa/{args.policy}/config.yaml"   
    else:
        config_file = f"runs/dijkstras/config.yaml"

    startup(config_file)