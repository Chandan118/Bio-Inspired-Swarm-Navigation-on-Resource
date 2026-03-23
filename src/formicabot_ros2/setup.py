"""
setup.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Module for Setup
"""

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'formicabot_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.xacro'))),
        (os.path.join('share', package_name, 'worlds'), glob(os.path.join('worlds', '*.world'))),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'scikit-learn'],
    zip_safe=True,
    maintainer='FormicaBot',
    maintainer_email='author@example.com',
    description='ROS 2 Humble implementation of the FormicaBot Swarm Robotics system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_environment_node = formicabot_ros2.sim_environment_node:main',
            'robot_node = formicabot_ros2.robot_node:main',
        ],
    },
)
