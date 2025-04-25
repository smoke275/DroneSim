import time
# from console import startup
from server import startup

if __name__ == '__main__':
    # policy_name = "7a52c0b8" # No Osc, nearest task
    # policy_name = "e1d9a053" # No Osc, direction to nearest task
    # policy_name = "cdad2483" # Osc, nearest task
    # policy_name = "feed2726" # Osc, direction to nearest task

    # policy_name = "a2879162" 
    policy_name = "fcf242b7"

    config_file = f"runs/sarsa/{policy_name}/config.yaml"

    # config_file = "runs/dijkstras/config.yaml"

    startup(config_file)
