import argparse
import sys
import threading

from PyQt5.QtWidgets import QApplication

from dronesim.config import Strategy, DEFAULT_STRATEGY
from dronesim.render import Window
from dronesim.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description='Drone logistics simulation')
    parser.add_argument('--maze', default='maze.csv', help='maze CSV to load')
    parser.add_argument('--strategy', default=DEFAULT_STRATEGY.value,
                        choices=[s.value for s in Strategy])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--trucks', type=int, default=None, help='fleet size override')
    parser.add_argument('--truck-range', type=float, default=None, help='battery range override')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = Window()
    kwargs = {}
    if args.trucks is not None:
        kwargs['num_trucks'] = args.trucks
    if args.truck_range is not None:
        kwargs['truck_range'] = args.truck_range
    sim = Simulation(window, strategy=Strategy(args.strategy),
                     seed=args.seed, maze_path=args.maze, **kwargs)
    threading.Thread(target=sim.run, daemon=True).start()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
