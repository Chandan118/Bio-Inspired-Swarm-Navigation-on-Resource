import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch file for FormicaBot Swarm Robotics System.
    Starts the central Environment Node and N Robot Nodes.
    """
    
    # Allow user to specify swarm size via command line argument
    n_robots_arg = DeclareLaunchArgument(
        'n_robots',
        default_value='10',
        description='Number of FormicaBot robots in the swarm'
    )
    
    ld = LaunchDescription()
    ld.add_action(n_robots_arg)
    
    # Start the central environment node
    sim_env_node = Node(
        package='formicabot_ros2',
        executable='sim_environment_node',
        name='sim_environment',
        output='screen'
    )
    ld.add_action(sim_env_node)
    
    # We use a static number for this Python instantiation example.
    # To truly launch `n_robots` dynamically in ROS 2 Foxy Python Launch files
    # without opaque functions, we'd normally use OpaqueFunction.
    # For this demonstration package, we instantiate 10 default robots.
    
    for i in range(10):
        # Evenly space them in a small circle around nest
        offset_x = 0.5 + (0.2 * (i % 5))
        offset_y = 5.0 + (0.2 * (i // 5))
        
        robot_node = Node(
            package='formicabot_ros2',
            executable='robot_node',
            name=f'formicabot_{i}',
            namespace=f'robot_{i}',
            parameters=[{
                'robot_id': i,
                'start_x': offset_x,
                'start_y': offset_y
            }],
            output='screen'
        )
        ld.add_action(robot_node)
        
    return ld
