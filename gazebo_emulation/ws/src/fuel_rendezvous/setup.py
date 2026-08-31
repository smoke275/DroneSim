import os
from glob import glob

from setuptools import setup

package_name = 'fuel_rendezvous'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Aerial battery-swap rendezvous emulation in Gazebo (gz-sim)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mission = fuel_rendezvous.mission:main',
            'drone_agent = fuel_rendezvous.drone_agent:main',
            'ugv_agent = fuel_rendezvous.ugv_agent:main',
            'bms_dispatcher = fuel_rendezvous.bms_dispatcher:main',
        ],
    },
)
