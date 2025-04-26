import time
from console import startup
# from server import startup

if __name__ == '__main__':
    # policy_name = "0d318b44"

    # config_file = f"runs/sarsa/{policy_name}/config.yaml"

    config_file = "runs/dijkstras/config.yaml"

    startup(config_file)
