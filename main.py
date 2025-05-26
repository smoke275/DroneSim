import argparse
from server import lmd_simulator
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drone Simulation Configuration")
    parser.add_argument('-c', '--config', type=str, help="Configuration file path")
    parser.add_argument('-p', '--policy', type=str, required=False, help="Policy name (e.g., '0d318b44')")
    args = parser.parse_args()


    if args.config:
        config_path = args.config
    else:
        config_path = f"runs/{args.policy}/config.yaml"

    config = yaml.safe_load(open(config_path, 'r'))

    metrics = lmd_simulator(config)
    print("Metrics:", metrics)