"""FUEL rendezvous mission: quadcopter intercepts a driving UGV, the UGV holds,
the copter hovers overhead for T_swap (the battery hand-off), then returns to
its pad while the UGV resumes. Logs the flight-phase metrics used to calibrate
the fleet simulator (approach time, capture tolerance, rendezvous duration).
"""

import csv
import math
import time
from enum import Enum, auto

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


class Phase(Enum):
    ARM = auto()
    TAKEOFF = auto()
    INTERCEPT = auto()
    HOVER_SWAP = auto()
    RETURN = auto()
    LAND = auto()
    IDLE = auto()  # docked and recharging; next cycle starts afterwards


class Mission(Node):

    def __init__(self):
        super().__init__('fuel_rendezvous_mission')
        p = self.declare_parameter
        self.cruise_alt = p('cruise_alt', 4.0).value       # above the walls
        self.hover_alt = p('hover_alt', 2.5).value         # above the UGV deck
        self.hold_radius = p('hold_radius', 5.0).value     # UGV halts inside this
        self.delta = p('capture_delta', 0.6).value         # capture tolerance [m]
        self.t_swap = p('t_swap', 10.0).value              # hover/swap duration [s]
        self.t_recharge = p('t_recharge', 8.0).value       # dock time between cycles [s]
        self.ugv_speed = p('ugv_speed', 0.6).value
        self.vmax_xy = p('vmax_xy', 2.5).value
        self.vmax_z = p('vmax_z', 1.5).value
        self.metrics_path = p('metrics_path', '/tmp/rendezvous_metrics.csv').value
        # gz OdometryPublisher reports poses relative to each model's spawn
        # pose; these [x, y, z, yaw] spawn poses (matching swap_world.sdf)
        # let us transform both odometries into the world frame.
        self.x3_spawn = p('x3_spawn', [0.0, 0.0, 0.053, 0.0]).value
        self.ugv_spawn = p('ugv_spawn', [12.0, -10.0, 0.25, 1.5708]).value
        # UGV route through the maze as [x1, y1, x2, y2, ...] (world frame);
        # shorter than 2 points -> legacy straight-line drive.
        wp = p('waypoints', [0.0]).value
        self.route = [(wp[i], wp[i + 1]) for i in range(0, len(wp) - 1, 2)] \
            if len(wp) >= 4 else []
        self.route_idx = 1 if self.route else 0
        self.ugv_yaw = None

        self.x3_cmd = self.create_publisher(Twist, '/x3/cmd_vel', 10)
        self.x3_enable = self.create_publisher(Bool, '/x3/enable', 10)
        self.ugv_cmd = self.create_publisher(Twist, '/ugv/cmd_vel', 10)
        self.create_subscription(Odometry, '/x3/odometry', self._x3_odom, 20)
        self.create_subscription(Odometry, '/ugv/odometry', self._ugv_odom, 20)

        self.x3 = None   # (x, y, z)
        self.ugv = None
        # Pad position comes from parameters, never from the first odometry
        # sample (which can catch a spawn-settling transient).
        self.pad = (self.x3_spawn[0], self.x3_spawn[1])
        self.phase = Phase.ARM
        self.phase_t0 = None
        self.t_start = None
        self.hover_errors = []
        self.marks = {}
        self.cycle = 1

        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz
        self.get_logger().info('mission node up; waiting for odometry')

    # ------------------------------------------------------------- callbacks
    # gz-harmonic's OdometryPublisher reports absolute world-frame pose, so
    # both odometries are used directly (no spawn-pose transform).
    @staticmethod
    def _yaw_of(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _x3_odom(self, msg):
        pos = msg.pose.pose.position
        self.x3 = (pos.x, pos.y, pos.z)
        self.x3_yaw = self._yaw_of(msg.pose.pose.orientation)
        tw = msg.twist.twist.linear
        c, s = math.cos(self.x3_yaw), math.sin(self.x3_yaw)
        self.x3_vel = (c * tw.x - s * tw.y, s * tw.x + c * tw.y, tw.z)

    def _ugv_odom(self, msg):
        pos = msg.pose.pose.position
        self.ugv = (pos.x, pos.y, pos.z)
        self.ugv_yaw = self._yaw_of(msg.pose.pose.orientation)

    # --------------------------------------------------------------- helpers
    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _enter(self, phase):
        self.get_logger().info(f'phase -> {phase.name}')
        self.marks[phase.name] = self._now() - self.t_start
        self.phase = phase
        self.phase_t0 = self._now()

    def _fly_to(self, tx, ty, tz, gain=0.8):
        """P-controller on position; the X3 twist command is body-frame, so
        rotate the desired world velocity by the drone's yaw and regulate the
        yaw itself back to zero."""
        x, y, z = self.x3
        wvx, wvy, wvz = getattr(self, 'x3_vel', (0.0, 0.0, 0.0))
        kd = 0.5  # damping against the inner velocity loop's latency
        vx = max(-self.vmax_xy, min(self.vmax_xy, gain * (tx - x) - kd * wvx))
        vy = max(-self.vmax_xy, min(self.vmax_xy, gain * (ty - y) - kd * wvy))
        vz = max(-self.vmax_z, min(self.vmax_z, 1.0 * (tz - z) - kd * wvz))
        yaw = getattr(self, 'x3_yaw', 0.0)
        c, s = math.cos(-yaw), math.sin(-yaw)
        cmd = Twist()
        cmd.linear.x = c * vx - s * vy
        cmd.linear.y = s * vx + c * vy
        cmd.linear.z = vz
        cmd.angular.z = max(-0.5, min(0.5, -1.0 * yaw))
        self.x3_cmd.publish(cmd)
        return math.hypot(tx - x, ty - y), abs(tz - z)

    def _drive_ugv(self, speed):
        """Drive the UGV: follow the maze route waypoints when one is set,
        else legacy straight-line drive. speed==0 halts in place."""
        cmd = Twist()
        if speed == 0.0 or not self.route or self.ugv_yaw is None:
            cmd.linear.x = speed
            self.ugv_cmd.publish(cmd)
            return
        if self.route_idx >= len(self.route):
            # Patrol continuously: drive the route back the other way
            self.route.reverse()
            self.route_idx = 1
        tx, ty = self.route[self.route_idx]
        dx, dy = tx - self.ugv[0], ty - self.ugv[1]
        if math.hypot(dx, dy) < 1.2:
            self.route_idx += 1
            self.ugv_cmd.publish(cmd)
            return
        err = math.atan2(dy, dx) - self.ugv_yaw
        err = math.atan2(math.sin(err), math.cos(err))
        cmd.angular.z = max(-1.0, min(1.0, 1.5 * err))
        cmd.linear.x = speed if abs(err) < 0.5 else 0.15
        self.ugv_cmd.publish(cmd)

    # ------------------------------------------------------------------ FSM
    def _tick(self):
        if self.x3 is None or self.ugv is None:
            return
        if self.t_start is None:
            self.t_start = self._now()
            self.phase_t0 = self._now()

        if self.phase == Phase.ARM:
            self.t_start = self.phase_t0  # cycle clock starts at arming
            self.marks = {}
            self.hover_errors = []
            self.x3_enable.publish(Bool(data=True))
            if self._now() - self.phase_t0 > 1.0:
                self._enter(Phase.TAKEOFF)

        elif self.phase == Phase.TAKEOFF:
            # Keep re-arming: an enable message lost during startup must not
            # leave the copter grounded.
            self.x3_enable.publish(Bool(data=True))
            self._drive_ugv(self.ugv_speed)  # UGV is already out on its route
            _, dz = self._fly_to(self.pad[0], self.pad[1], self.cruise_alt)
            if dz < 0.3:
                self._enter(Phase.INTERCEPT)

        elif self.phase == Phase.INTERCEPT:
            ux, uy, _ = self.ugv
            dxy, _ = self._fly_to(ux, uy, self.cruise_alt)
            if dxy <= self.hold_radius:
                self._drive_ugv(0.0)  # truck HOLDING
            else:
                self._drive_ugv(self.ugv_speed)
            if dxy <= self.delta:
                self._enter(Phase.HOVER_SWAP)

        elif self.phase == Phase.HOVER_SWAP:
            self._drive_ugv(0.0)
            ux, uy, uz = self.ugv
            dxy, _ = self._fly_to(ux, uy, uz + self.hover_alt, gain=1.2)
            self.hover_errors.append(dxy)
            if self._now() - self.phase_t0 >= self.t_swap:
                self._enter(Phase.RETURN)

        elif self.phase == Phase.RETURN:
            self._drive_ugv(self.ugv_speed)  # truck resumes immediately
            dxy, dz = self._fly_to(self.pad[0], self.pad[1], self.cruise_alt)
            if dxy < 0.4 and dz < 0.3:
                self._enter(Phase.LAND)

        elif self.phase == Phase.LAND:
            self._drive_ugv(self.ugv_speed)
            dxy, dz = self._fly_to(self.pad[0], self.pad[1], 0.7, gain=0.5)
            if dz < 0.15:
                self.x3_enable.publish(Bool(data=False))
                self._enter(Phase.IDLE)
                self._report()
                self.cycle += 1

        elif self.phase == Phase.IDLE:
            # Docked and recharging while the UGV patrols; then go again,
            # mirroring the continuous dispatch loop of the fleet simulator.
            self._drive_ugv(self.ugv_speed)
            if self._now() - self.phase_t0 >= self.t_recharge:
                self._enter(Phase.ARM)

    # -------------------------------------------------------------- metrics
    def _report(self):
        m = self.marks
        mean_err = sum(self.hover_errors) / max(1, len(self.hover_errors))
        max_err = max(self.hover_errors) if self.hover_errors else 0.0
        row = {
            'cycle': self.cycle,
            'takeoff_end_s': round(m.get('INTERCEPT', 0.0), 2),
            'approach_time_s': round(m.get('HOVER_SWAP', 0.0) - m.get('INTERCEPT', 0.0), 2),
            'swap_hover_s': round(self.t_swap, 2),
            'hover_mean_err_m': round(mean_err, 3),
            'hover_max_err_m': round(max_err, 3),
            'return_land_s': round(m.get('IDLE', 0.0) - m.get('RETURN', 0.0), 2),
            'cycle_total_s': round(m.get('IDLE', 0.0), 2),
        }
        new_file = self.cycle == 1
        with open(self.metrics_path, 'a' if not new_file else 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self.get_logger().info('METRICS cycle %d: %s' % (
            self.cycle, '  '.join(f'{k}={v}' for k, v in row.items() if k != 'cycle')))


def main():
    rclpy.init()
    node = Mission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
