"""Sprite loading and a rotation cache (quantized to 5 degrees)."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QTransform


class SpriteStore:

    def __init__(self):
        self.pixmaps = {
            'drone': QPixmap('transparent_drone.png').scaled(50, 50, Qt.KeepAspectRatio),
            'truck': QPixmap('truck.png').scaled(40, 40, Qt.KeepAspectRatio),
        }
        self._cache = {}

    def rotated(self, key, angle):
        a = int(round(angle / 5.0) * 5) % 360
        if (key, a) not in self._cache:
            self._cache[(key, a)] = self.pixmaps[key].transformed(
                QTransform().rotate(a), Qt.SmoothTransformation)
        return self._cache[(key, a)]
