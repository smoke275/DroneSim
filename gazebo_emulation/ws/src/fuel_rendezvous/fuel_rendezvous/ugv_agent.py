"""Per-truck controller: follows its maze route (ping-pong patrol), integrates
a distance-based battery model, halts while a drone holds it, and resets its
battery when a swap completes. One instance per ground robot."""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class UgvAgent(Node):

    def __init__(self):
        super().__init__('ugv_agent')
        self.name = self.declare_parameter('name', 'ugv_0').value
        wp = self.declare_parameter('route', [0.0]).value
        self.route = [(wp[i], wp[i + 1]) for i in range(0, len(wp) - 1, 2)]
        self.speed = self.declare_parameter('speed', 1.0).value
        # full battery lasts this many meters of driving
        self.battery_range = self.declare_parameter('battery_range_m', 120.0).value

        self.pos = None
        self.yaw = None
        self.prev_pos = None
        self.soc = 1.0
        self.hold = False
        self.route_idx = 1

        self.cmd_pub = self.create_publisher(Twist, f'/{self.name}/cmd_vel', 10)
        self.soc_pub = self.create_publisher(Float32, f'/{self.name}/soc', 10)
        self.create_subscription(Odometry, f'/{self.name}/odometry', self._odom, 20)
        self.create_subscription(Bool, f'/{self.name}/hold', self._hold, 10)
        self.create_subscription(Bool, f'/{self.name}/swap', self._swap, 10)
        self.create_timer(0.05, self._tick)
        self.create_timer(0.5, lambda: self.soc_pub.publish(Float32(data=self.soc)))
        self.get_logger().info(f'{self.name}: route of {len(self.route)} waypoints')

    def _odom(self, msg):
        pos = msg.pose.pose.position
        if self.pos is not None:
            self.soc = max(0.0, self.soc - math.hypot(pos.x - self.pos[0],
                                                      pos.y - self.pos[1]) / self.battery_range)
        self.pos = (pos.x, pos.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _hold(self, msg):
        self.hold = msg.data

    def _swap(self, msg):
        if msg.data:
            self.soc = 1.0
            self.get_logger().info(f'{self.name}: battery swapped, SOC 100%')

    def _tick(self):
        cmd = Twist()
        if self.pos is None or self.yaw is None or self.hold or len(self.route) < 2:
            self.cmd_pub.publish(cmd)
            return
        if self.route_idx >= len(self.route):
            self.route.reverse()  # patrol back the other way
            self.route_idx = 1
        tx, ty = self.route[self.route_idx]
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        if math.hypot(dx, dy) < 1.2:
            self.route_idx += 1
            self.cmd_pub.publish(cmd)
            return
        err = math.atan2(dy, dx) - self.yaw
        err = math.atan2(math.sin(err), math.cos(err))
        cmd.angular.z = max(-1.0, min(1.0, 1.5 * err))
        cmd.linear.x = self.speed if abs(err) < 0.5 else 0.15
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = UgvAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
