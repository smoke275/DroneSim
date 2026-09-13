"""Per-drone flight controller: waits docked until the BMS assigns it a truck,
then flies the rendezvous (intercept -> truck holds -> hover swap -> return ->
land -> recharge) and reports per-cycle metrics. One instance per drone."""

import csv
import math
import os
import random
from enum import Enum, auto

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


class Phase(Enum):
    IDLE = auto()
    ARM = auto()
    TAKEOFF = auto()
    INTERCEPT = auto()
    HOVER_SWAP = auto()
    RETURN = auto()
    LAND = auto()
    RECHARGE = auto()


class DroneAgent(Node):

    def __init__(self):
        super().__init__('drone_agent')
        p = self.declare_parameter
        self.name = p('name', 'x3_0').value
        self.pad = tuple(p('pad', [0.0, 0.0]).value)
        self.truck_names = p('truck_names', ['ugv_0']).value
        self.cruise_alt = p('cruise_alt', 5.0).value
        self.hover_alt = p('hover_alt', 2.5).value
        self.hold_radius = p('hold_radius', 5.0).value
        self.delta = p('capture_delta', 0.6).value
        self.t_swap = p('t_swap', 10.0).value
        self.t_recharge = p('t_recharge', 8.0).value
        self.vmax_xy = p('vmax_xy', 2.5).value
        self.vmax_z = p('vmax_z', 1.5).value
        # Disturbance / logging parameters (nominal = 0)
        self.pos_noise_std = float(p('pos_noise_std', 0.0).value)   # GNSS noise on estimates [m]
        self.drop_prob = float(p('drop_prob', 0.0).value)           # telemetry loss probability
        self.tag = p('run_tag', '').value
        self.wind_speed = float(p('wind_speed', 0.0).value)         # recorded for the CSV only
        self.wind_gust = float(p('wind_gust', 0.0).value)
        self.rng = random.Random(int(p('seed', 0).value) * 1000 + sum(map(ord, self.name)))
        self.capture_tol = float(p('capture_tol', 0.10).value)       # mechanism tolerance (RTK +-10 cm)
        # Integral gain of the position loop (0 = proportional-only guidance,
        # the paper's nominal controller). A steady wind leaves a P-only loop
        # with a constant offset ~ F_wind/(m*k_p); integral action removes it.
        self.ki = float(p('pos_ki', 0.0).value)
        self.i_max = float(p('pos_i_max', 2.0).value)                 # anti-windup [m/s]
        self.i_err = [0.0, 0.0, 0.0]
        self.last_tick = None

        self.cmd_pub = self.create_publisher(Twist, f'/{self.name}/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool, f'/{self.name}/enable', 10)
        self.status_pub = self.create_publisher(String, f'/{self.name}/status', 10)
        self.hold_pubs = {t: self.create_publisher(Bool, f'/{t}/hold', 10)
                          for t in self.truck_names}
        self.swap_pubs = {t: self.create_publisher(Bool, f'/{t}/swap', 10)
                          for t in self.truck_names}

        self.create_subscription(Odometry, f'/{self.name}/odometry', self._odom, 20)
        self.create_subscription(String, f'/{self.name}/assign', self._assign, 10)
        self.truck_pos = {}        # estimated (noisy, possibly stale) truck positions
        self.truck_true = {}       # ground truth, metrics only
        for t in self.truck_names:
            self.create_subscription(Odometry, f'/{t}/odometry',
                                     lambda msg, t=t: self._truck_odom(t, msg), 20)

        self.pos = None            # estimated own position (control)
        self.true_pos = None       # ground truth (metrics)
        self.yaw = 0.0
        self.vel = (0.0, 0.0, 0.0)
        self.phase = Phase.IDLE
        self.phase_t0 = None
        self.target = None
        self.t_start = None
        self.marks = {}
        self.hover_errors = []
        self.capture_losses = 0
        self.captured = False
        self.dropped = 0
        self.received = 0
        self.cycle = 0
        suffix = f'{self.tag}_{self.name}' if self.tag else self.name
        self.metrics_path = f'/tmp/rendezvous_metrics_{suffix}.csv'

        self.create_timer(0.05, self._tick)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(f'{self.name}: pad {self.pad}, serving {self.truck_names}')

    # ------------------------------------------------------------- callbacks
    def _noisy(self, x, y):
        if self.pos_noise_std > 0.0:
            return x + self.rng.gauss(0.0, self.pos_noise_std), y + self.rng.gauss(0.0, self.pos_noise_std)
        return x, y

    def _dropped(self):
        self.received += 1
        if self.drop_prob > 0.0 and self.rng.random() < self.drop_prob:
            self.dropped += 1
            return True
        return False

    def _truck_odom(self, t, msg):
        pos = msg.pose.pose.position
        self.truck_true[t] = (pos.x, pos.y, pos.z)
        if self._dropped():
            return  # stale estimate persists
        x, y = self._noisy(pos.x, pos.y)
        self.truck_pos[t] = (x, y, pos.z)

    def _odom(self, msg):
        pos = msg.pose.pose.position
        self.true_pos = (pos.x, pos.y, pos.z)
        if self._dropped():
            return
        x, y = self._noisy(pos.x, pos.y)
        self.pos = (x, y, pos.z)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        tw = msg.twist.twist.linear
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.vel = (c * tw.x - s * tw.y, s * tw.x + c * tw.y, tw.z)

    def _assign(self, msg):
        if self.phase == Phase.IDLE and msg.data in self.truck_names:
            self.target = msg.data
            self.cycle += 1
            self.t_start = self._now()
            self.marks = {}
            self.hover_errors = []
            self.capture_losses = 0
            self.captured = False
            self._enter(Phase.ARM)
            self.get_logger().info(f'{self.name}: dispatched to {self.target}')

    def _publish_status(self):
        self.status_pub.publish(String(data=self.phase.name))

    # --------------------------------------------------------------- helpers
    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _enter(self, phase):
        if self.t_start is not None:
            self.marks[phase.name] = self._now() - self.t_start
        self.get_logger().info(f'{self.name}: phase -> {phase.name}')
        self.phase = phase
        self.phase_t0 = self._now()
        self.i_err = [0.0, 0.0, 0.0]  # fresh integrator per phase / setpoint

    def _fly_to(self, tx, ty, tz, gain=0.8):
        x, y, z = self.pos
        kd = 0.5
        now = self._now()
        dt = 0.05 if self.last_tick is None else max(0.0, min(0.2, now - self.last_tick))
        self.last_tick = now
        if self.ki > 0.0:
            for i, e in enumerate((tx - x, ty - y, tz - z)):
                self.i_err[i] = max(-self.i_max, min(self.i_max, self.i_err[i] + self.ki * e * dt))
        ix, iy, iz = self.i_err
        vx = max(-self.vmax_xy, min(self.vmax_xy, gain * (tx - x) + ix - kd * self.vel[0]))
        vy = max(-self.vmax_xy, min(self.vmax_xy, gain * (ty - y) + iy - kd * self.vel[1]))
        vz = max(-self.vmax_z, min(self.vmax_z, 1.0 * (tz - z) + iz - kd * self.vel[2]))
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        cmd = Twist()
        cmd.linear.x = c * vx - s * vy
        cmd.linear.y = s * vx + c * vy
        cmd.linear.z = vz
        cmd.angular.z = max(-0.5, min(0.5, -1.0 * self.yaw))
        self.cmd_pub.publish(cmd)
        return math.hypot(tx - x, ty - y), abs(tz - z)

    def _set_hold(self, value):
        if self.target is not None:
            self.hold_pubs[self.target].publish(Bool(data=value))

    # ------------------------------------------------------------------ FSM
    def _tick(self):
        if self.pos is None:
            return

        if self.phase == Phase.IDLE:
            return

        if self.phase == Phase.ARM:
            self.enable_pub.publish(Bool(data=True))
            if self._now() - self.phase_t0 > 1.0:
                self._enter(Phase.TAKEOFF)

        elif self.phase == Phase.TAKEOFF:
            self.enable_pub.publish(Bool(data=True))  # re-arm against lost msgs
            _, dz = self._fly_to(self.pad[0], self.pad[1], self.cruise_alt)
            if dz < 0.3:
                self._enter(Phase.INTERCEPT)

        elif self.phase == Phase.INTERCEPT:
            if self.target not in self.truck_pos:
                return
            ux, uy, _ = self.truck_pos[self.target]
            dxy, _ = self._fly_to(ux, uy, self.cruise_alt)
            self._set_hold(dxy <= self.hold_radius)
            if dxy <= self.delta:
                self._enter(Phase.HOVER_SWAP)

        elif self.phase == Phase.HOVER_SWAP:
            self._set_hold(True)
            ux, uy, uz = self.truck_pos[self.target]
            self._fly_to(ux, uy, uz + self.hover_alt, gain=1.2)
            # Metrics use ground truth: the true deck-to-drone offset
            tx, ty, _ = self.truck_true.get(self.target, (ux, uy, uz))
            px, py, _ = self.true_pos if self.true_pos else self.pos
            err = math.hypot(tx - px, ty - py)
            self.hover_errors.append(err)
            if err <= self.delta:
                self.captured = True
            elif self.captured:
                self.captured = False
                self.capture_losses += 1  # drifted out of the capture cone
            if self._now() - self.phase_t0 >= self.t_swap:
                self.swap_pubs[self.target].publish(Bool(data=True))
                self._set_hold(False)
                self._enter(Phase.RETURN)

        elif self.phase == Phase.RETURN:
            dxy, dz = self._fly_to(self.pad[0], self.pad[1], self.cruise_alt)
            if dxy < 0.4 and dz < 0.3:
                self._enter(Phase.LAND)

        elif self.phase == Phase.LAND:
            dxy, dz = self._fly_to(self.pad[0], self.pad[1], 0.7, gain=0.5)
            if dz < 0.15:
                self.enable_pub.publish(Bool(data=False))
                self._enter(Phase.RECHARGE)
                self._report()
                self.target = None

        elif self.phase == Phase.RECHARGE:
            if self._now() - self.phase_t0 >= self.t_recharge:
                self._enter(Phase.IDLE)

    # -------------------------------------------------------------- metrics
    def _report(self):
        m = self.marks
        errs = sorted(self.hover_errors)
        n = max(1, len(errs))
        mean_err = sum(errs) / n
        p95 = errs[min(len(errs) - 1, int(0.95 * len(errs)))] if errs else 0.0
        within_tol = sum(1 for e in errs if e <= self.capture_tol) / n
        within_delta = sum(1 for e in errs if e <= self.delta) / n
        row = {
            'cycle': self.cycle,
            'truck': self.target,
            'takeoff_end_s': round(m.get('INTERCEPT', 0.0), 2),
            'approach_time_s': round(m.get('HOVER_SWAP', 0.0) - m.get('INTERCEPT', 0.0), 2),
            'swap_hover_s': round(self.t_swap, 2),
            'hover_mean_err_m': round(mean_err, 3),
            'hover_p95_err_m': round(p95, 3),
            'hover_max_err_m': round(max(errs), 3) if errs else 0.0,
            'frac_within_tol': round(within_tol, 3),
            'frac_within_delta': round(within_delta, 3),
            'capture_losses': self.capture_losses,
            'return_land_s': round(m.get('RECHARGE', 0.0) - m.get('RETURN', 0.0), 2),
            'cycle_total_s': round(m.get('RECHARGE', 0.0), 2),
            'msg_drop_frac': round(self.dropped / max(1, self.received), 3),
            'wind_speed': self.wind_speed,
            'wind_gust': self.wind_gust,
            'pos_noise': self.pos_noise_std,
            'drop_prob': self.drop_prob,
            'pos_ki': self.ki,
            'run_tag': self.tag,
        }
        new_file = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self.get_logger().info('%s METRICS cycle %d: %s' % (
            self.name, self.cycle,
            '  '.join(f'{k}={v}' for k, v in row.items() if k not in ('cycle',))))


def main():
    rclpy.init()
    node = DroneAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
