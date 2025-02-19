import time
from server import startup

if __name__ == '__main__':
    config = {
        "world": {
            "maze": "data/maze.csv",
            "num_evs": 6,
            "num_tasks": 35,
            "num_uavs_per_bs": 2,
            "num_base_stations": 4,
        },
        "ev": {
            "range": 1700
        },
        "uav": {
            "range": 2000,
            "speed": 3
        }
    }

    evs_range = list(range(6, 9))  # 5 to 10 inclusive
    tasks_range = list(range(35, 40))  # 35 to 50 inclusive
    R_range = list(range(1500, 2001, 100))  # 1500 to 3000 inclusive, step 100

    total_iterations = len(evs_range) * len(tasks_range) * len(R_range)
    current = 1

    for num_evs in evs_range:
        for num_tasks in tasks_range:
            for ev_range in R_range:
                print(f"Iteration {current}/{total_iterations}: Num EVs: {num_evs}, Num Tasks: {num_tasks}, EV Range: {ev_range}")
                config["world"]["num_evs"] = num_evs
                config["world"]["num_tasks"] = num_tasks
                config["ev"]["range"] = ev_range
                startup(config)
                current += 1
