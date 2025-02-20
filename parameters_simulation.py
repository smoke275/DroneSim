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

    evs_range = range(4, 21)  # 5 to 10 inclusive
    tasks_range = range(20,101)  # 35 to 50 inclusive
    R_range = range(100,3001,10)  # 1500 to 3000 inclusive, step 100

    epochs_per_param = 100

    output_file = f"runs/randomized_simulatation/results_evs_tasks_range.csv"
    with open(output_file, 'w') as f:
        f.write("num_evs,num_tasks,ev_range,timesteps,num_tasks_finished\n")


    total_iterations = len(evs_range) * len(tasks_range) * len(R_range)
    current = 1

    for num_evs in evs_range:
        for num_tasks in tasks_range:
            for ev_range in R_range:
                print(f"Iteration {current}/{total_iterations}: Num EVs: {num_evs}, Num Tasks: {num_tasks}, EV Range: {ev_range}")
                for epoch in range(epochs_per_param):
                    print("Epoch:", epoch)
                    config["world"]["num_evs"] = num_evs
                    config["world"]["num_tasks"] = num_tasks
                    config["ev"]["range"] = ev_range
                    startup(config, output_file)
                    current += 1
