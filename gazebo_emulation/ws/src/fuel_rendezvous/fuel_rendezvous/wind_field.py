"""Wind field driver: sets the world wind velocity at runtime through Gazebo's
WindEffects topic (/world/<world>/wind, gz.msgs.Wind). The base wind is a
mean vector plus Ornstein-Uhlenbeck gusts (per-axis, correlation time
``gust_period``, standard deviation ``gust_std``). The realised wind is
logged to /tmp/wind_<tag>.csv so a run's actual disturbance is recorded.

Nothing is published when both mean speed and gust std are zero, so the
nominal experiment is unchanged."""

import csv
import math
import random
import subprocess

import rclpy
from rclpy.node import Node


class WindField(Node):

    def __init__(self):
        super().__init__('wind_field')
        p = self.declare_parameter
        self.world = p('world', 'swap_world').value
        self.mean_speed = float(p('mean_speed', 0.0).value)
        self.dir_deg = float(p('dir_deg', 0.0).value)
        self.gust_std = float(p('gust_std', 0.0).value)
        self.gust_period = max(0.5, float(p('gust_period', 5.0).value))
        self.rate = float(p('rate_hz', 2.0).value)
        self.tag = p('run_tag', '').value
        seed = int(p('seed', 0).value)
        self.rng = random.Random(seed)

        d = math.radians(self.dir_deg)
        self.mean = (self.mean_speed * math.cos(d), self.mean_speed * math.sin(d))
        self.gust = [0.0, 0.0]
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.samples = 0
        self.log_path = f'/tmp/wind_{self.tag or "run"}.csv'
        self.log = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.log)
        self.writer.writerow(['t_s', 'vx', 'vy', 'speed'])

        if self.mean_speed == 0.0 and self.gust_std == 0.0:
            self.get_logger().info('wind_field: calm (no wind published)')
            self._publish(0.0, 0.0, enable=False)
            return
        self.get_logger().info(
            f'wind_field: mean {self.mean_speed:.1f} m/s @ {self.dir_deg:.0f} deg, '
            f'gust std {self.gust_std:.1f} m/s, period {self.gust_period:.1f} s')
        self.create_timer(1.0 / self.rate, self._tick)

    def _tick(self):
        dt = 1.0 / self.rate
        tau = self.gust_period
        for i in range(2):
            # Ornstein-Uhlenbeck: mean-reverting Gaussian gusts
            self.gust[i] += (-self.gust[i] * dt / tau
                             + self.gust_std * math.sqrt(2.0 * dt / tau) * self.rng.gauss(0.0, 1.0))
        vx = self.mean[0] + self.gust[0]
        vy = self.mean[1] + self.gust[1]
        self._publish(vx, vy, enable=True)
        t = self.get_clock().now().nanoseconds / 1e9 - self.t0
        self.writer.writerow([round(t, 2), round(vx, 3), round(vy, 3), round(math.hypot(vx, vy), 3)])
        self.samples += 1
        if self.samples % (int(self.rate) * 15) == 0:
            self.log.flush()
            self.get_logger().info(f'wind now {math.hypot(vx, vy):.1f} m/s')

    def _publish(self, vx, vy, enable):
        msg = f'linear_velocity: {{x: {vx:.3f}, y: {vy:.3f}, z: 0.0}}, enable_wind: {"true" if enable else "false"}'
        try:
            subprocess.run(['gz', 'topic', '-t', f'/world/{self.world}/wind', '-m', 'gz.msgs.Wind', '-p', msg],
                           check=False, timeout=2.0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            self.get_logger().warn('gz topic publish timed out')


def main():
    rclpy.init()
    node = WindField()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.log.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
