import random
import time
from math import floor
from server import startup

if __name__ == '__main__':
    config = {
        "world": {
            "maze_size": 35,
            "maze_loop_percentage": 80,
            "num_evs": 6,
            "num_tasks": 35,
            "num_uavs_per_bs": 2,
            "num_base_stations": 4,
            "bms": False
        },
        "ev": {
            "range": 1700
        },
        "uav": {
            "range": 2000,
            "speed": 3
        }
    }

    # Example parameter ranges:
    evs_range = range(6, 11)               # e.g., 4..20
    tasks_range = range(35, 41)           # e.g., 20..100
    R_range = range(1500, 3001, 100)         # e.g., 100..3000 step 10
    # Maze parameters (will be randomized each epoch)
    MAZE_SIZE_OPTIONS         = list(range(35, 51))
    MAZE_LOOP_PERCENT_OPTIONS = list(range(60, 91, 5))

    epochs_per_param = 20

    # 1) Write all the param names and metrics to the results file.
    #    Add anything else you want to track (distance metrics, etc.).
    output_file = f"runs/scenario_1/results.csv"
    with open(output_file, 'w') as f:
        f.write("num_evs,num_tasks,ev_range,maze_loop_percentage,maze_size,bms,timesteps,num_tasks_finished,ev_distance,drone_distance\n")

    # 2) Calculate how many total parameter combinations (and total iterations with epochs).
    total_param_combos = (
        len(evs_range)
        * len(tasks_range)
        * len(R_range)
    )
    total_iterations = total_param_combos * epochs_per_param

    # For the progress bar:
    current_iter = 0
    start_time = time.time()

    # 3) Nested loops:
    for num_evs in evs_range:
        for num_tasks in tasks_range:
            for ev_range in R_range:
                maze_size = random.choice(MAZE_SIZE_OPTIONS)
                loop_per = random.choice(MAZE_LOOP_PERCENT_OPTIONS)

                # Start the clock for this param combo (all epochs).
                param_combo_start = time.time()

                # Print which parameters we're about to run
                print(f"[Param Combo] Maze={maze_size}, Loop%={loop_per}, EVs={num_evs}, "
                        f"Tasks={num_tasks}, EV Range={ev_range}")

                # Run all epochs for the current combination
                for epoch in range(epochs_per_param):
                    # Update config
                    config["world"]["maze_size"] = maze_size
                    config["world"]["maze_loop_percentage"] = loop_per
                    config["world"]["num_evs"] = num_evs
                    config["world"]["num_tasks"] = num_tasks
                    config["ev"]["range"] = ev_range

                    # Do the simulation run
                    startup(config, output_file)

                    # Update iteration count
                    current_iter += 1

                    # --- Progress bar & ETA ---
                    elapsed = time.time() - start_time
                    avg_sec_per_iter = elapsed / current_iter
                    remaining = total_iterations - current_iter
                    eta_sec = remaining * avg_sec_per_iter

                    # Build a simple bar of length 50
                    progress = current_iter / total_iterations
                    bar_len = 50
                    filled_len = int(progress * bar_len)
                    bar = '=' * filled_len + '-' * (bar_len - filled_len)

                    # Format ETA into H:MM:SS
                    hrs = floor(eta_sec / 3600)
                    mins = floor((eta_sec % 3600) / 60)
                    secs = floor(eta_sec % 60)
                    eta_str = f"{hrs}h {mins}m {secs}s"

                    print(f"\r[{bar}] {progress*100:.2f}% "
                            f"(Iteration {current_iter}/{total_iterations}) "
                            f"ETA: {eta_str}     ", end="")
                    # The 'end=""' plus '\r' does an in-place update of the same line.
                
                # Once all epochs are done for this param combo:
                param_combo_time = time.time() - param_combo_start
                print(f"\nAll {epochs_per_param} epochs done for Maze={maze_size}, "
                        f"Loop%={loop_per}, EVs={num_evs}, Tasks={num_tasks}, "
                        f"EV Range={ev_range}.  "
                        f"(Took {param_combo_time:.2f} sec)\n")
                exit()
