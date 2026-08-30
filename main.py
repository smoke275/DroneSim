import sys
import threading

from PyQt5.QtWidgets import QApplication

from dronesim.render import Window
from dronesim.simulation import Simulation


def main():
    app = QApplication(sys.argv)
    window = Window()
    sim = Simulation(window)
    threading.Thread(target=sim.run, daemon=True).start()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
