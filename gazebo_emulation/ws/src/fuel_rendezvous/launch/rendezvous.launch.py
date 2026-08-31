import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('fuel_rendezvous')
    world = os.path.join(share, 'worlds', 'swap_world.sdf')
    bridge_cfg = os.path.join(share, 'config', 'bridge.yaml')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    with open(os.path.join(share, 'config', 'fleet.yaml')) as f:
        fleet = yaml.safe_load(f)
    drones = fleet['drones']
    trucks = fleet['trucks']
    truck_names = [t['name'] for t in trucks]

    headless = LaunchConfiguration('headless')

    nodes = [
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             parameters=[{'config_file': bridge_cfg}], output='screen'),
        Node(package='fuel_rendezvous', executable='bms_dispatcher', output='screen',
             parameters=[{
                 'drone_names': [d['name'] for d in drones],
                 'drone_pads': [v for d in drones for v in d['pad']],
                 'truck_names': truck_names,
             }]),
    ]
    for d in drones:
        nodes.append(Node(
            package='fuel_rendezvous', executable='drone_agent', output='screen',
            name=f"drone_agent_{d['name']}",
            parameters=[{'name': d['name'], 'pad': d['pad'],
                         'truck_names': truck_names}]))
    for t in trucks:
        nodes.append(Node(
            package='fuel_rendezvous', executable='ugv_agent', output='screen',
            name=f"ugv_agent_{t['name']}",
            parameters=[{'name': t['name'], 'route': t['route'],
                         'speed': float(t.get('speed', 1.0))}]))

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='run gz sim without the GUI'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': f'-r {world}'}.items(),
            condition=UnlessCondition(headless)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': f'-r -s {world}'}.items(),
            condition=IfCondition(headless)),
    ] + nodes)
