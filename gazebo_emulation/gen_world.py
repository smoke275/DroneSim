"""Generate the multi-robot Gazebo world + bridge + fleet config from a maze
CSV of the fleet simulator.

Recreates the experiment architecture in ROS 2: each truck and each drone gets
its own controller node (ugv_agent / drone_agent) and a central BMS dispatcher
assigns drones to low-battery trucks. This script emits:

  worlds/swap_world.sdf   maze walls, N trucks, M drones with full dynamics
  config/bridge.yaml      per-robot ros_gz topic bridge
  config/fleet.yaml       pads, routes, spawns for the launch file

    python3 gen_world.py --maze ../maze.csv --trucks 2 --drones 2
Then rebuild the image (./start.sh).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dronesim.world import World  # noqa: E402

PKG = os.path.join(os.path.dirname(__file__), 'ws', 'src', 'fuel_rendezvous')

WALL_H = 2.5
WALL_T = 0.3

MOTOR_TPL = '''      <plugin filename="gz-sim-multicopter-motor-model-system"
              name="gz::sim::systems::MulticopterMotorModel">
        <robotNamespace>{ns}</robotNamespace>
        <jointName>X3/rotor_{i}_joint</jointName>
        <linkName>X3/rotor_{i}</linkName>
        <turningDirection>{dirn}</turningDirection>
        <timeConstantUp>0.0125</timeConstantUp>
        <timeConstantDown>0.025</timeConstantDown>
        <maxRotVelocity>800.0</maxRotVelocity>
        <motorConstant>8.54858e-06</motorConstant>
        <momentConstant>0.016</momentConstant>
        <commandSubTopic>gazebo/command/motor_speed</commandSubTopic>
        <actuator_number>{i}</actuator_number>
        <rotorDragCoefficient>8.06428e-05</rotorDragCoefficient>
        <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
        <motorSpeedPubTopic>motor_speed/{i}</motorSpeedPubTopic>
        <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
        <motorType>velocity</motorType>
      </plugin>'''


def drone_sdf(name, ns, x, y):
    motors = '\n'.join(MOTOR_TPL.format(ns=ns, i=i, dirn=d)
                       for i, d in enumerate(['ccw', 'ccw', 'cw', 'cw']))
    return f'''
    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/X3 UAV/4</uri>
      <name>{name}</name>
      <pose>{x:.2f} {y:.2f} 0.35 0 0 0</pose>
{motors}
      <plugin filename="gz-sim-multicopter-control-system"
              name="gz::sim::systems::MulticopterVelocityControl">
        <robotNamespace>{ns}</robotNamespace>
        <commandSubTopic>gazebo/command/twist</commandSubTopic>
        <enableSubTopic>enable</enableSubTopic>
        <comLinkName>X3/base_link</comLinkName>
        <velocityGain>2.7 2.7 2.7</velocityGain>
        <attitudeGain>2 3 0.15</attitudeGain>
        <angularRateGain>0.4 0.52 0.18</angularRateGain>
        <maximumLinearAcceleration>2 2 2</maximumLinearAcceleration>
        <maximumLinearVelocity>4 4 2.5</maximumLinearVelocity>
        <maximumAngularVelocity>3 3 3</maximumAngularVelocity>
        <linearVelocityNoiseMean>0 0 0</linearVelocityNoiseMean>
        <linearVelocityNoiseStdDev>0.1105 0.1261 0.0947</linearVelocityNoiseStdDev>
        <angularVelocityNoiseMean>0 0 0</angularVelocityNoiseMean>
        <angularVelocityNoiseStdDev>0.004 0.004 0.004</angularVelocityNoiseStdDev>
        <rotorConfiguration>
          <rotor><jointName>X3/rotor_0_joint</jointName><forceConstant>8.54858e-06</forceConstant><momentConstant>0.016</momentConstant><direction>1</direction></rotor>
          <rotor><jointName>X3/rotor_1_joint</jointName><forceConstant>8.54858e-06</forceConstant><momentConstant>0.016</momentConstant><direction>1</direction></rotor>
          <rotor><jointName>X3/rotor_2_joint</jointName><forceConstant>8.54858e-06</forceConstant><momentConstant>0.016</momentConstant><direction>-1</direction></rotor>
          <rotor><jointName>X3/rotor_3_joint</jointName><forceConstant>8.54858e-06</forceConstant><momentConstant>0.016</momentConstant><direction>-1</direction></rotor>
        </rotorConfiguration>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system"
              name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
      </plugin>
    </include>
'''


def truck_sdf(name, x, y, yaw, color):
    return f'''
    <model name="{name}">
      <pose>{x:.2f} {y:.2f} 0.25 0 0 {yaw:.4f}</pose>
      <link name="chassis">
        <pose>0 0 0.2 0 0 0</pose>
        <inertial>
          <mass>20</mass>
          <inertia><ixx>0.8</ixx><iyy>1.5</iyy><izz>1.8</izz></inertia>
        </inertial>
        <collision name="c"><geometry><box><size>1.2 0.8 0.35</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>1.2 0.8 0.35</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
        </visual>
        <visual name="deck">
          <pose>0 0 0.19 0 0 0</pose>
          <geometry><box><size>0.6 0.6 0.03</size></box></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
        </visual>
      </link>
      <link name="left_wheel">
        <pose>0.35 0.45 0.0 -1.5708 0 0</pose>
        <inertial><mass>2</mass><inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.05</izz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.22</radius><length>0.12</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.22</radius><length>0.12</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
      </link>
      <link name="right_wheel">
        <pose>0.35 -0.45 0.0 -1.5708 0 0</pose>
        <inertial><mass>2</mass><inertia><ixx>0.03</ixx><iyy>0.03</iyy><izz>0.05</izz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.22</radius><length>0.12</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.22</radius><length>0.12</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
      </link>
      <link name="caster">
        <pose>-0.45 0 -0.05 0 0 0</pose>
        <inertial><mass>1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
        <collision name="c"><geometry><sphere><radius>0.17</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><sphere><radius>0.17</radius></sphere></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material></visual>
      </link>
      <joint name="left_wheel_joint" type="revolute">
        <parent>chassis</parent><child>left_wheel</child><axis><xyz>0 0 1</xyz></axis>
      </joint>
      <joint name="right_wheel_joint" type="revolute">
        <parent>chassis</parent><child>right_wheel</child><axis><xyz>0 0 1</xyz></axis>
      </joint>
      <joint name="caster_joint" type="ball"><parent>chassis</parent><child>caster</child></joint>
      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.9</wheel_separation>
        <wheel_radius>0.22</wheel_radius>
        <topic>/model/{name}/cmd_vel</topic>
        <odom_topic>/model/{name}/wheel_odometry</odom_topic>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system"
              name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
      </plugin>
    </model>
'''


TRUCK_COLORS = ['0.85 0.55 0.1 1', '0.2 0.6 0.85 1', '0.75 0.2 0.25 1', '0.3 0.7 0.35 1']


def simplify(cells):
    out = [cells[0]]
    for a, b, c in zip(cells, cells[1:], cells[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    out.append(cells[-1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze', default=os.path.join(os.path.dirname(__file__), '..', 'maze.csv'))
    parser.add_argument('--cell', type=float, default=3.0)
    parser.add_argument('--trucks', type=int, default=6)
    parser.add_argument('--drones', type=int, default=4)
    parser.add_argument('--route-cells', type=int, default=22)
    args = parser.parse_args()

    w = World(args.maze)
    s = args.cell / w.cell_size

    def cell_m(r, c):
        x, y = w.cell_to_canvas(r, c)
        return x * s, y * s

    # Like the fleet simulator: every truck starts from the central warehouse
    # and fans out toward its own task region. Spawns are staggered along each
    # truck's own exit corridor so the fleet doesn't pile up on one cell.
    mr, mc = w.max_row, w.max_col
    goals = [(2, 2), (2, mc - 2), (mr - 2, 2), (mr - 2, mc - 2),
             (2, mc // 2), (mr - 2, mc // 2), (mr // 2, 2), (mr // 2, mc - 2)]
    trucks = []
    used_spawn_cells = set()
    for i in range(args.trucks):
        path = w.find_shortest_path(w.warehouse_cell, goals[i % len(goals)])
        if not path:
            raise SystemExit(f'no route for truck {i}')
        path = path[:args.route_cells]
        start = min(i, len(path) - 2)
        while start < len(path) - 2 and tuple(path[start]) in used_spawn_cells:
            start += 1
        used_spawn_cells.add(tuple(path[start]))
        route = [cell_m(r, c) for r, c in simplify(path[start:])]
        yaw = math.atan2(route[1][1] - route[0][1], route[1][0] - route[0][0])
        trucks.append({'name': f'ugv_{i}', 'route': route, 'yaw': yaw,
                       'speed': round(0.85 + 0.06 * i, 2)})

    # Drone base stations spread across the four quadrants, like the sim's
    # base stations; drones cycle through them.
    pad_cells = [(mr // 4, mc // 4), (mr // 4, 3 * mc // 4),
                 (3 * mr // 4, mc // 4), (3 * mr // 4, 3 * mc // 4)]
    drones = []
    for i in range(args.drones):
        pr, pc = pad_cells[i % 4]
        pr = max(1, min(w.max_row, pr))
        pc = max(1, min(w.max_col, pc))
        drones.append({'name': f'x3_{i}', 'ns': f'X3_{i}', 'pad': cell_m(pr, pc)})

    walls = []
    for i, (x1, y1, x2, y2) in enumerate(w.wall_segments):
        x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.hypot(x2 - x1, y2 - y1) + WALL_T
        yaw = math.atan2(y2 - y1, x2 - x1)
        walls.append(
            f'    <model name="wall_{i}"><static>true</static><link name="l">'
            f'<pose>{cx:.2f} {cy:.2f} {WALL_H / 2} 0 0 {yaw:.4f}</pose>'
            f'<collision name="c"><geometry><box><size>{length:.2f} {WALL_T} {WALL_H}</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>{length:.2f} {WALL_T} {WALL_H}</size></box></geometry>'
            f'<material><ambient>0.45 0.42 0.4 1</ambient><diffuse>0.45 0.42 0.4 1</diffuse></material>'
            f'</visual></link></model>')

    pads_sdf = ''.join(
        f'''    <model name="pad_{i}"><static>true</static><link name="link">
      <pose>{d['pad'][0]:.2f} {d['pad'][1]:.2f} 0.01 0 0 0</pose>
      <visual name="visual"><geometry><cylinder><radius>1.0</radius><length>0.02</length></cylinder></geometry>
        <material><ambient>0.2 0.45 0.75 1</ambient><diffuse>0.2 0.45 0.75 1</diffuse></material></visual>
    </link></model>\n''' for i, d in enumerate(drones))

    extent = max(w.max_row, w.max_col) * args.cell + 20
    sdf = f'''<?xml version="1.0" ?>
<!-- GENERATED by gen_world.py from {os.path.basename(args.maze)} — do not edit by hand -->
<sdf version="1.9">
  <world name="swap_world">
    <physics name="4ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>{extent:.0f} {extent:.0f}</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{extent:.0f} {extent:.0f}</size></plane></geometry>
          <material><ambient>0.75 0.78 0.72 1</ambient><diffuse>0.75 0.78 0.72 1</diffuse></material>
        </visual>
      </link>
    </model>

    <!-- Central warehouse, matching the fleet simulator -->
    <model name="warehouse">
      <static>true</static>
      <link name="link">
        <pose>{cell_m(*w.warehouse_cell)[0]:.2f} {cell_m(*w.warehouse_cell)[1]:.2f} 0.25 0 0 0</pose>
        <visual name="v">
          <geometry><cylinder><radius>1.3</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.2 0.29 0.37 1</ambient><diffuse>0.2 0.29 0.37 1</diffuse></material>
        </visual>
      </link>
    </model>

{pads_sdf}
{chr(10).join(walls)}
{''.join(drone_sdf(d['name'], d['ns'], *d['pad']) for d in drones)}
{''.join(truck_sdf(t['name'], t['route'][0][0], t['route'][0][1], t['yaw'], TRUCK_COLORS[i % 4]) for i, t in enumerate(trucks))}
  </world>
</sdf>
'''
    with open(os.path.join(PKG, 'worlds', 'swap_world.sdf'), 'w') as f:
        f.write(sdf)

    # ros_gz bridge config
    bridge = []
    for d in drones:
        bridge.append(f'''- ros_topic_name: /{d['name']}/cmd_vel
  gz_topic_name: /{d['ns']}/gazebo/command/twist
  ros_type_name: geometry_msgs/msg/Twist
  gz_type_name: gz.msgs.Twist
  direction: ROS_TO_GZ
- ros_topic_name: /{d['name']}/enable
  gz_topic_name: /{d['ns']}/enable
  ros_type_name: std_msgs/msg/Bool
  gz_type_name: gz.msgs.Boolean
  direction: ROS_TO_GZ
- ros_topic_name: /{d['name']}/odometry
  gz_topic_name: /model/{d['name']}/odometry
  ros_type_name: nav_msgs/msg/Odometry
  gz_type_name: gz.msgs.Odometry
  direction: GZ_TO_ROS
''')
    for t in trucks:
        bridge.append(f'''- ros_topic_name: /{t['name']}/cmd_vel
  gz_topic_name: /model/{t['name']}/cmd_vel
  ros_type_name: geometry_msgs/msg/Twist
  gz_type_name: gz.msgs.Twist
  direction: ROS_TO_GZ
- ros_topic_name: /{t['name']}/odometry
  gz_topic_name: /model/{t['name']}/odometry
  ros_type_name: nav_msgs/msg/Odometry
  gz_type_name: gz.msgs.Odometry
  direction: GZ_TO_ROS
''')
    with open(os.path.join(PKG, 'config', 'bridge.yaml'), 'w') as f:
        f.write('# GENERATED by gen_world.py\n' + '\n'.join(bridge))

    # Fleet description for the launch file / agents / dispatcher
    lines = ['# GENERATED by gen_world.py', 'drones:']
    for d in drones:
        lines.append(f"  - name: {d['name']}")
        lines.append(f"    pad: [{d['pad'][0]:.2f}, {d['pad'][1]:.2f}]")
    lines.append('trucks:')
    for t in trucks:
        flat = ', '.join(f'{v:.2f}' for xy in t['route'] for v in xy)
        lines.append(f"  - name: {t['name']}")
        lines.append(f"    speed: {t['speed']}")
        lines.append(f'    route: [{flat}]')
    with open(os.path.join(PKG, 'config', 'fleet.yaml'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Remove stale single-robot params from previous generator versions
    stale = os.path.join(PKG, 'config', 'mission_params.yaml')
    if os.path.exists(stale):
        os.remove(stale)

    print(f'world: {len(walls)} walls, {len(trucks)} trucks, {len(drones)} drones')
    for t in trucks:
        print(f"  {t['name']}: {len(t['route'])} waypoints from ({t['route'][0][0]:.0f},{t['route'][0][1]:.0f})")
    for d in drones:
        print(f"  {d['name']}: pad ({d['pad'][0]:.0f},{d['pad'][1]:.0f})")
    print('rebuild the image (./start.sh)')


if __name__ == '__main__':
    main()
