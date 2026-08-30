"""Qt window and draw-operation renderer with zoom/pan support.

The simulation thread pushes draw operations via Window.draw() and commits a
frame with Window.execute(); paintEvent replays the committed stack.

World coordinates: x right, y up, spanning [-BOUNDARY, +BOUNDARY].
OPERATION.text takes window-frame coords (pass -world_y for the y value).
OPERATION.hud_* ops are drawn in screen pixels and ignore zoom/pan.
"""

import threading
from enum import Enum, auto

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QFont
from PyQt5.QtWidgets import QMainWindow

from . import config

sem = threading.Semaphore()


class OPERATION(Enum):
    point = auto()
    line = auto()
    dotted_line = auto()
    circle = auto()
    filled_circle = auto()
    border_circle = auto()
    polygon = auto()
    filled_polygon = auto()
    dotted_polygon = auto()
    border_polygon = auto()
    text = auto()
    image = auto()
    wall = auto()       # line with drop shadow and rounded caps
    hud_panel = auto()  # screen-space translucent panel: [op, x, y, w, h]
    hud_text = auto()   # screen-space text: [op, x, y, color, string]


GROUND_COLOR = QColor('#f2f4f1')
OUTSIDE_COLOR = QColor('#d5dbe0')
SITE_BORDER_COLOR = QColor('#90a4ae')
WALL_COLOR = QColor('#37474f')
WALL_SHADOW = QColor(38, 50, 56, 70)


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.action_stack = []
        self.main_stack = []

        # Static background layer (maze walls/grid), cached per view
        self._static_ops = []
        self._static_version = 0
        self._static_pixmap = None
        self._static_key = None

        # View state (world coords)
        self.zoom = 1.0
        self.view_cx = 0.0
        self.view_cy = 0.0
        self._drag_start = None

        self.setWindowTitle('Drone Logistics Simulation')
        self.setGeometry(300, 120, config.WINDOW_SIZE, config.WINDOW_SIZE)
        self.show()

    # ------------------------------------------------------------------ frame
    def draw(self, value):
        self.action_stack.append(value)

    def set_static_ops(self, ops):
        """Register draw ops that never change (walls, grid). They are
        rendered into a cached layer and re-rasterized only when the view
        changes, instead of being stroked every frame."""
        self._static_ops = list(ops)
        self._static_version += 1

    def execute(self):
        sem.acquire()
        self.main_stack = self.action_stack
        self.action_stack = []
        sem.release()
        self.update()

    # ------------------------------------------------------------------- view
    def _view_rect(self):
        """Visible window-frame rect (window y = -world y)."""
        vw = 2 * config.BOUNDARY_X / self.zoom
        vh = 2 * config.BOUNDARY_Y / self.zoom
        return QRectF(self.view_cx - vw / 2, -self.view_cy - vh / 2, vw, vh)

    def _pixel_to_world(self, px, py):
        rect = self._view_rect()
        u = rect.left() + px / max(1, self.width()) * rect.width()
        v = rect.top() + py / max(1, self.height()) * rect.height()
        return u, -v

    def _clamp_view(self):
        self.zoom = max(config.MIN_ZOOM, min(config.MAX_ZOOM, self.zoom))
        self.view_cx = max(-config.BOUNDARY_X, min(config.BOUNDARY_X, self.view_cx))
        self.view_cy = max(-config.BOUNDARY_Y, min(config.BOUNDARY_Y, self.view_cy))

    def wheelEvent(self, event):
        # Zoom about the cursor so the world point under the mouse stays put
        px, py = event.pos().x(), event.pos().y()
        wx, wy = self._pixel_to_world(px, py)
        notches = event.angleDelta().y() / 120.0
        self.zoom *= config.ZOOM_STEP ** notches
        self._clamp_view()
        vw = 2 * config.BOUNDARY_X / self.zoom
        vh = 2 * config.BOUNDARY_Y / self.zoom
        self.view_cx = wx - (px / max(1, self.width())) * vw + vw / 2
        self.view_cy = wy + (py / max(1, self.height())) * vh - vh / 2
        self._clamp_view()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = (event.pos().x(), event.pos().y(), self.view_cx, self.view_cy)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            sx, sy, cx0, cy0 = self._drag_start
            rect = self._view_rect()
            dpx = event.pos().x() - sx
            dpy = event.pos().y() - sy
            self.view_cx = cx0 - dpx * rect.width() / max(1, self.width())
            self.view_cy = cy0 + dpy * rect.height() / max(1, self.height())
            self._clamp_view()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag_start = None

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_R:
            self.zoom = 1.0
            self.view_cx = 0.0
            self.view_cy = 0.0
            self.update()
        elif e.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom *= config.ZOOM_STEP
            self._clamp_view()
            self.update()
        elif e.key() == Qt.Key_Minus:
            self.zoom /= config.ZOOM_STEP
            self._clamp_view()
            self.update()

    # ------------------------------------------------------------------ paint
    def _apply_world_transform(self, painter):
        rect = self._view_rect()
        painter.setWindow(QRect(int(rect.left()), int(rect.top()),
                                max(1, int(rect.width())), max(1, int(rect.height()))))
        painter.setViewport(QRect(0, 0, self.width(), self.height()))
        painter.scale(1, -1)

    def _render_static_layer(self, key):
        """Rasterize ground + static ops into a pixmap for the current view."""
        pixmap = QtGui.QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(pixmap.rect(), OUTSIDE_COLOR)
        self._apply_world_transform(painter)

        site = QRectF(-config.BOUNDARY_X, -config.BOUNDARY_Y,
                      2 * config.BOUNDARY_X, 2 * config.BOUNDARY_Y)
        painter.fillRect(site, GROUND_COLOR)
        painter.setPen(QPen(SITE_BORDER_COLOR, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(site)

        self._replay_ops(painter, self._static_ops)
        painter.end()
        self._static_pixmap = pixmap
        self._static_key = key

    def paintEvent(self, event):
        painter = QPainter(self)

        key = (round(self.zoom, 4), round(self.view_cx, 2), round(self.view_cy, 2),
               self.width(), self.height(), self._static_version)
        if self._static_pixmap is None or key != self._static_key:
            self._render_static_layer(key)
        painter.drawPixmap(0, 0, self._static_pixmap)

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.TextAntialiasing)
        self._apply_world_transform(painter)
        painter.setFont(QFont('Sans', 13))

        sem.acquire()
        my_stack = self.main_stack
        sem.release()

        hud_ops = self._replay_ops(painter, my_stack)

        # Screen-space HUD pass: unaffected by zoom/pan
        painter.resetTransform()
        painter.setFont(QFont('Sans', 10))
        for i in hud_ops:
            if i[0] == OPERATION.hud_panel:
                painter.setPen(QPen(QColor(96, 125, 139, 120), 1))
                painter.setBrush(QBrush(QColor(255, 255, 255, 210)))
                painter.drawRoundedRect(QRectF(i[1], i[2], i[3], i[4]), 6, 6)
            else:
                painter.setPen(QPen(QColor(i[3]), 1))
                painter.drawText(QPointF(float(i[1]), float(i[2])), i[4])

        # View hint in the corner
        painter.setPen(QPen(QColor(69, 90, 100, 180), 1))
        painter.setFont(QFont('Sans', 8))
        painter.drawText(QPointF(10, self.height() - 10),
                         f'zoom {self.zoom:.1f}x  ·  scroll: zoom  ·  drag: pan  ·  R: reset  ·  Esc: quit')

    def _replay_ops(self, painter, ops):
        """Draw world-space ops; returns any HUD ops encountered for a later
        screen-space pass."""
        hud_ops = []
        for i in ops:
            if i[0] == OPERATION.point:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.drawPoint(QPointF(i[1], i[2]))
            elif i[0] == OPERATION.line:
                painter.setPen(QPen(i[6], i[5], Qt.SolidLine))
                painter.drawLine(QPointF(i[1], i[2]), QPointF(i[3], i[4]))
            elif i[0] == OPERATION.dotted_line:
                painter.setPen(QPen(i[6], i[5], Qt.DotLine))
                painter.drawLine(QPointF(i[1], i[2]), QPointF(i[3], i[4]))
            elif i[0] == OPERATION.wall:
                pen = QPen(WALL_SHADOW, 5.5, Qt.SolidLine)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(i[1] + 2.2, i[2] - 2.2), QPointF(i[3] + 2.2, i[4] - 2.2))
                pen = QPen(WALL_COLOR, 3.5, Qt.SolidLine)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(i[1], i[2]), QPointF(i[3], i[4]))
            elif i[0] == OPERATION.circle:
                painter.setPen(QPen(i[5], i[4], Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.filled_circle:
                painter.setPen(QPen(i[5], i[4], Qt.SolidLine))
                painter.setBrush(QBrush(i[5], Qt.SolidPattern))
                painter.drawEllipse(QPointF(i[1], i[2]), i[3], i[3])
            elif i[0] == OPERATION.border_circle:
                painter.setPen(QPen(Qt.black, i[4], Qt.SolidLine))
                painter.setBrush(QBrush(i[5], Qt.SolidPattern))
                painter.drawEllipse(QPointF(i[1], i[2]), i[3], i[3])
            elif i[0] in (OPERATION.polygon, OPERATION.dotted_polygon,
                          OPERATION.filled_polygon, OPERATION.border_polygon):
                points = [QPointF(x, y) for x, y in zip(i[1], i[2])]
                poly = QPolygonF(points)
                if i[0] == OPERATION.polygon:
                    painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                    painter.setBrush(Qt.NoBrush)
                elif i[0] == OPERATION.dotted_polygon:
                    painter.setPen(QPen(i[4], i[3], Qt.DotLine))
                    painter.setBrush(Qt.NoBrush)
                elif i[0] == OPERATION.filled_polygon:
                    painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                    painter.setBrush(QBrush(i[4], Qt.SolidPattern))
                else:  # border_polygon
                    painter.setPen(QPen(Qt.black, i[3], Qt.SolidLine))
                    painter.setBrush(QBrush(i[4], Qt.SolidPattern))
                painter.drawPolygon(poly)
            elif i[0] == OPERATION.text:
                painter.setPen(QPen(i[4], i[3], Qt.SolidLine))
                painter.save()
                painter.scale(1, -1)
                painter.drawText(QPointF(float(i[1]), float(i[2])), i[5])
                painter.restore()
            elif i[0] == OPERATION.image:
                painter.drawPixmap(int(i[1]), int(i[2]), i[3])
            elif i[0] in (OPERATION.hud_panel, OPERATION.hud_text):
                hud_ops.append(i)
        return hud_ops
