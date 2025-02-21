import random
import time
import psutil
from math import floor
from concurrent.futures import ProcessPoolExecutor, as_completed

from server import startup  # Your existing function that runs one simulation

# ------------------ CONFIGURATION ------------------

# Ranges for the 'fixed' parameters
NUM_EVS_RANGE       = range(4, 21)        # e.g., 4 to 20
NUM_TASKS_RANGE     = range(20, 101)        # e.g., 20 to 100
EV_RANGE_RANGE      = range(500, 3001, 10)  # e.g., 100, 110, 120, ..., 3000

# Maze parameters (will be randomized each epoch)
MAZE_SIZE_OPTIONS         = list(range(35, 51))
MAZE_LOOP_PERCENT_OPTIONS = list(range(60, 91, 5))

EPOCHS_PER_COMBO = 50

# Dynamic window thresholds
MIN_WORKERS = 2
MAX_WORKERS = 8
CPU_LOW_THRESHOLD  = 40.0
CPU_HIGH_THRESHOLD = 85.0
CHECK_INTERVAL     = 5  # seconds between resource checks

# Output file for results
OUTPUT_FILE = "runs/randomized_simulation/results_dynamic_window.csv"

# ------------------ DYNAMIC WINDOW HELPER ------------------

def dynamic_window_size(current_size):
    """Return an updated pool size based on current CPU usage."""
    cpu_usage = psutil.cpu_percent(interval=0.5)
    if cpu_usage < CPU_LOW_THRESHOLD and current_size < MAX_WORKERS:
        return min(current_size + 1, MAX_WORKERS)
    elif cpu_usage > CPU_HIGH_THRESHOLD and current_size > MIN_WORKERS:
        return max(current_size - 1, MIN_WORKERS)
    return current_size

# ------------------ JOB WRAPPER ------------------

def run_simulation_job(params):
    """
    Each job's parameters include:
      - Fixed parameters: num_evs, num_tasks, ev_range
      - Randomized maze parameters: maze_size, maze_loop_percentage
      - And the new 'bms' switch (True/False)
    Calls `startup(config, OUTPUT_FILE)` once.
    """
    config = {
        "world": {
            "maze_size": params['maze_size'],
            "maze_loop_percentage": params['maze_loop_percentage'],
            "num_evs": params['num_evs'],
            "num_tasks": params['num_tasks'],
            "num_uavs_per_bs": 2,
            "num_base_stations": 4,
            "bms": params['bms']
        },
        "ev": {
            "range": params['ev_range']
        },
        "uav": {
            "range": 2000,
            "speed": 3
        }
    }
    # Run one simulation; startup writes results to OUTPUT_FILE
    startup(config, OUTPUT_FILE)
    return True

# ------------------ MAIN SCRIPT ------------------

def main():
    # Build a list of jobs. For each combination of num_evs, num_tasks, ev_range,
    # create EPOCHS_PER_COMBO jobs with randomized maze parameters.
    # For each combo, include two jobs: one with bms False and one with bms True.
    jobs = []
    for evs in NUM_EVS_RANGE:
        for tasks in NUM_TASKS_RANGE:
            for ev_range in EV_RANGE_RANGE:
                for _ in range(EPOCHS_PER_COMBO):
                    job_params = {
                        'num_evs': evs,
                        'num_tasks': tasks,
                        'ev_range': ev_range,
                        'maze_size': random.choice(MAZE_SIZE_OPTIONS),
                        'maze_loop_percentage': random.choice(MAZE_LOOP_PERCENT_OPTIONS),
                        'bms': False
                    }
                    jobs.append(job_params)
                    job_params = {
                        'num_evs': evs,
                        'num_tasks': tasks,
                        'ev_range': ev_range,
                        'maze_size': random.choice(MAZE_SIZE_OPTIONS),
                        'maze_loop_percentage': random.choice(MAZE_LOOP_PERCENT_OPTIONS),
                        'bms': True
                    }
                    jobs.append(job_params)

    total_jobs = len(jobs)
    completed_jobs = 0
    start_time = time.time()

    # Start a ProcessPoolExecutor with an initial number of workers
    current_workers = MIN_WORKERS
    executor = ProcessPoolExecutor(max_workers=current_workers)

    # Dictionary to track futures -> job parameters
    futures = {}

    # Submit an initial batch of tasks up to the current worker count
    for _ in range(min(current_workers, total_jobs)):
        job_params = jobs.pop()
        fut = executor.submit(run_simulation_job, job_params)
        futures[fut] = job_params

    last_check_time = time.time()

    # Main loop: as tasks complete, submit new ones and adjust pool size dynamically
    while futures:
        for completed_future in as_completed(futures, timeout=1):
            completed_jobs += 1
            del futures[completed_future]

            # Calculate progress and ETA
            elapsed = time.time() - start_time
            progress = completed_jobs / total_jobs
            avg_time = elapsed / completed_jobs
            remaining = total_jobs - completed_jobs
            eta_sec = remaining * avg_time

            bar_len = 40
            filled_len = int(progress * bar_len)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)

            hrs = floor(eta_sec / 3600)
            mins = floor((eta_sec % 3600) / 60)
            secs = floor(eta_sec % 60)
            eta_str = f"{hrs}h {mins}m {secs}s"

            # Updated progress printout now includes the current pool size.
            print(
                f"\r[{bar}] {progress * 100:.1f}%  "
                f"({completed_jobs}/{total_jobs})  ETA: {eta_str}  "
                f"Pool Size: {current_workers} ",
                end=''
            )

            # Submit a new job if available
            if jobs:
                next_job_params = jobs.pop()
                fut = executor.submit(run_simulation_job, next_job_params)
                futures[fut] = next_job_params

        # Adjust pool size every CHECK_INTERVAL seconds based on CPU usage.
        now = time.time()
        if now - last_check_time > CHECK_INTERVAL:
            new_size = dynamic_window_size(current_workers)
            if new_size != current_workers:
                print(f"\n[Adjusting pool size {current_workers} -> {new_size}]")
                executor._max_workers = new_size  # Quick hack to update on the fly
                current_workers = new_size
            last_check_time = now

    total_time = time.time() - start_time
    print(f"\nAll {total_jobs} jobs done in {total_time:.2f} seconds.")
    executor.shutdown()

if __name__ == '__main__':
    main()
