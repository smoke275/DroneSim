import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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

    def fparam(name):
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    # Disturbance / run-condition arguments (all default to the nominal,
    # undisturbed experiment of the paper's case study).
    args = [
        DeclareLaunchArgument('headless', default_value='false',
                              description='run gz sim without the GUI'),
        DeclareLaunchArgument('run_tag', default_value='',
                              description='label written into the metrics CSV names and rows'),
        DeclareLaunchArgument('wind_speed', default_value='0.0',
                              description='mean horizontal wind speed [m/s]'),
        DeclareLaunchArgument('wind_dir_deg', default_value='0.0',
                              description='mean wind direction [deg, world frame]'),
        DeclareLaunchArgument('wind_gust', default_value='0.0',
                              description='gust standard deviation [m/s] (Ornstein-Uhlenbeck around the mean)'),
        DeclareLaunchArgument('wind_gust_period', default_value='5.0',
                              description='gust correlation time [s]'),
        DeclareLaunchArgument('pos_noise', default_value='0.0',
                              description='GNSS position noise std applied to drone and truck estimates [m]'),
        DeclareLaunchArgument('drop_prob', default_value='0.0',
                              description='probability that an odometry/telemetry message is lost'),
        DeclareLaunchArgument('seed', default_value='0',
                              description='random seed for noise and drops'),
        DeclareLaunchArgument('pos_ki', default_value='0.0',
                              description='integral gain of the drone position loop (0 = P-only nominal guidance)'),
    ]

    nodes = [
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             parameters=[{'config_file': bridge_cfg}], output='screen'),
        Node(package='fuel_rendezvous', executable='bms_dispatcher', output='screen',
             parameters=[{
                 'drone_names': [d['name'] for d in drones],
                 'drone_pads': [v for d in drones for v in d['pad']],
                 'truck_names': truck_names,
             }]),
        Node(package='fuel_rendezvous', executable='wind_field', output='screen',
             parameters=[{
                 'world': 'swap_world',
                 'mean_speed': fparam('wind_speed'),
                 'dir_deg': fparam('wind_dir_deg'),
                 'gust_std': fparam('wind_gust'),
                 'gust_period': fparam('wind_gust_period'),
                 'seed': ParameterValue(LaunchConfiguration('seed'), value_type=int),
                 'run_tag': ParameterValue(LaunchConfiguration('run_tag'), value_type=str),
             }]),
    ]
    for d in drones:
        nodes.append(Node(
            package='fuel_rendezvous', executable='drone_agent', output='screen',
            name=f"drone_agent_{d['name']}",
            parameters=[{'name': d['name'], 'pad': d['pad'],
                         'truck_names': truck_names,
                         'pos_noise_std': fparam('pos_noise'),
                         'drop_prob': fparam('drop_prob'),
                         'seed': ParameterValue(LaunchConfiguration('seed'), value_type=int),
                         'run_tag': ParameterValue(LaunchConfiguration('run_tag'), value_type=str),
                         'wind_speed': fparam('wind_speed'),
                         'wind_gust': fparam('wind_gust'),
                         'pos_ki': fparam('pos_ki')}]))
    for t in trucks:
        nodes.append(Node(
            package='fuel_rendezvous', executable='ugv_agent', output='screen',
            name=f"ugv_agent_{t['name']}",
            parameters=[{'name': t['name'], 'route': t['route'],
                         'speed': float(t.get('speed', 1.0))}]))

    return LaunchDescription(args + [
        # Local models (wind-enabled X3) must be resolvable before gz sim starts
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(share, 'models')),
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
