"""Central BMS dispatcher: monitors every truck's state of charge and assigns
the nearest idle drone when a truck needs a swap — the ROS counterpart of the
fleet simulator's predictive dispatch (Algorithm 2)."""

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String


class BmsDispatcher(Node):

    def __init__(self):
        super().__init__('bms_dispatcher')
        p = self.declare_parameter
        self.drone_names = p('drone_names', ['x3_0']).value
        pads_flat = p('drone_pads', [0.0, 0.0]).value
        self.pads = {n: (pads_flat[2 * i], pads_flat[2 * i + 1])
                     for i, n in enumerate(self.drone_names)}
        self.truck_names = p('truck_names', ['ugv_0']).value
        self.soc_threshold = p('soc_threshold', 0.55).value  # predictive margin

        self.soc = {}
        self.truck_pos = {}
        self.drone_status = {}
        self.assigned = {}  # truck -> drone

        self.assign_pubs = {d: self.create_publisher(String, f'/{d}/assign', 10)
                            for d in self.drone_names}
        for t in self.truck_names:
            self.create_subscription(Float32, f'/{t}/soc',
                                     lambda m, t=t: self.soc.__setitem__(t, m.data), 10)
            self.create_subscription(
                Odometry, f'/{t}/odometry',
                lambda m, t=t: self.truck_pos.__setitem__(
                    t, (m.pose.pose.position.x, m.pose.pose.position.y)), 20)
        for d in self.drone_names:
            self.create_subscription(String, f'/{d}/status',
                                     lambda m, d=d: self._status(d, m.data), 10)

        self.create_timer(0.5, self._tick)
        self.get_logger().info(
            f'BMS up: drones {self.drone_names}, trucks {self.truck_names}, '
            f'dispatch below SOC {self.soc_threshold:.0%}')

    def _status(self, drone, status):
        self.drone_status[drone] = status
        if status == 'IDLE':
            # mission finished: free any truck this drone was serving
            for truck, d in list(self.assigned.items()):
                if d == drone and self._acked(drone):
                    del self.assigned[truck]

    def _acked(self, drone):
        """An assignment is acked once the drone has left IDLE at least once;
        track via a simple flag set when we saw it non-idle."""
        return getattr(self, f'_flew_{drone}', False)

    def _tick(self):
        for drone, status in self.drone_status.items():
            if status != 'IDLE':
                setattr(self, f'_flew_{drone}', True)

        for truck in self.truck_names:
            soc = self.soc.get(truck)
            if soc is None or soc >= self.soc_threshold or truck in self.assigned:
                continue
            drone = self._nearest_idle_drone(truck)
            if drone is None:
                continue
            self.assigned[truck] = drone
            setattr(self, f'_flew_{drone}', False)
            self.get_logger().info(
                f'DISPATCH {drone} -> {truck} (SOC {soc:.0%})')

        # Re-publish unacked assignments so a lost message cannot strand a truck
        for truck, drone in self.assigned.items():
            if self.drone_status.get(drone) == 'IDLE' and not self._acked(drone):
                self.assign_pubs[drone].publish(String(data=truck))

    def _nearest_idle_drone(self, truck):
        if truck not in self.truck_pos:
            return None
        tx, ty = self.truck_pos[truck]
        busy = set(self.assigned.values())
        best, best_d = None, float('inf')
        for d in self.drone_names:
            if d in busy or self.drone_status.get(d) != 'IDLE':
                continue
            px, py = self.pads[d]
            dist = math.hypot(px - tx, py - ty)
            if dist < best_d:
                best, best_d = d, dist
        return best


def main():
    rclpy.init()
    node = BmsDispatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
