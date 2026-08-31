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
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = Window()
    sim = Simulation(window, strategy=Strategy(args.strategy),
                     seed=args.seed, maze_path=args.maze)
    threading.Thread(target=sim.run, daemon=True).start()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
