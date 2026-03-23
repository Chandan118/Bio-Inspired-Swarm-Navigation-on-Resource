"""
gazebo.launch.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Module for Gazebo.Launch
"""

import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def scrub_env():
    """Surgically remove Noetic and handle display/python variables."""
    print("--- ☢️ ADVANCED ENVIRONMENT SCRUBBING ☢️ ---")
    
    # Isolate from Noetic
    for var in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH']:
        val = os.environ.get(var, '')
        if 'noetic' in val.lower():
            clean_parts = [p for p in val.split(':') if 'noetic' not in p.lower()]
            os.environ[var] = ':'.join(clean_parts)
    
    os.environ['ROS_DISTRO'] = 'humble'
    
    # Python XML fix
    if '/opt/ros/humble/lib/python3.10/site-packages' not in os.environ.get('PYTHONPATH', ''):
        os.environ['PYTHONPATH'] = '/opt/ros/humble/lib/python3.10/site-packages:' + os.environ.get('PYTHONPATH', '')

    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
        
    print(f"CWD: {os.getcwd()}")
    print("Environment isolated for ROS 2 Humble.")

def find_file_in_workspace(filename):
    """Deep search for a file in the workspace."""
    ws_base = os.getcwd()
    print(f"DEBUG: Searching for {filename} starting from {ws_base}")
    
    # Try multiple base points
    search_points = [ws_base, os.path.join(ws_base, 'src'), os.path.dirname(ws_base)]
    
    for start_node in search_points:
        if not os.path.exists(start_node): continue
        for root, dirs, files in os.walk(start_node):
            if filename in files:
                return os.path.join(root, filename)
    return None

def generate_launch_description():
    # 0. Scrub the environment
    scrub_env()

    pkg_dir = get_package_share_directory('formicabot_ros2')
    world_path = os.path.join(pkg_dir, 'worlds', 'arena.world')
    
    # --- DEEP URDF DISCOVERY ---
    unitree_urdf = ""
    
    # 1. Try find in 'go1_description' install if sourced
    try:
        go1_pkg = get_package_share_directory('go1_description')
        for sub in ['urdf', 'xacro', 'models', 'share/go1_description/urdf']:
            p = os.path.join(go1_pkg, sub, 'go1.xacro')
            if os.path.exists(p):
                unitree_urdf = p
                break
    except:
        pass
        
    # 2. Deep search in workspace (Very robust)
    if not unitree_urdf:
        unitree_urdf = find_file_in_workspace('go1.xacro')
        
    if unitree_urdf:
        print(f"🎯 SUCCESS: High-Fidelity Model Found at {unitree_urdf}")
        # Inject paths for Gazebo to find meshes
        # Usually meshes are at ../meshes or in the share directory
        pkg_root = os.path.dirname(os.path.dirname(unitree_urdf))
        os.environ['GAZEBO_MODEL_PATH'] = pkg_root + ':' + os.environ.get('GAZEBO_MODEL_PATH', '')
    else:
        print("❌ CRITICAL: go1.xacro NOT FOUND. Swarm will use fallback 'cube-style' robots.")
        unitree_urdf = os.path.join(pkg_dir, 'urdf', 'unitree_dog.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Environment synchronization
    env_actions = []
    for var in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'GAZEBO_MODEL_PATH', 'DISPLAY']:
        env_actions.append(SetEnvironmentVariable(var, os.environ.get(var, '')))

    ld = LaunchDescription()
    for action in env_actions:
        ld.add_action(action)

    # 1. Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': world_path, 'verbose': 'true'}.items()
    )
    ld.add_action(gazebo)

    # 1.5 Swarm Coordinator
    sim_env_node = Node(
        package='formicabot_ros2', executable='sim_environment_node', name='sim_environment',
        parameters=[{'n_robots': 5}], output='screen'
    )
    ld.add_action(sim_env_node)

    # 2. Spawn Swarm
    total_robots = 5
    for i in range(total_robots):
        namespace = f'robot_{i}'
        robot_name = f'go1_{i}'
        tmp_urdf = f'/tmp/{robot_name}.urdf'
        
        # Hardened Xacro Command
        xacro_cmd = f'PYTHONPATH={os.environ["PYTHONPATH"]} xacro {unitree_urdf} use_nominal_extrinsics:=true add_ignore:=false'
        
        print(f"Processing {robot_name}...")
        os.system(f'{xacro_cmd} > {tmp_urdf} 2>/tmp/xacro_err_{i}.log')
        
        if not os.path.exists(tmp_urdf) or os.path.getsize(tmp_urdf) < 500:
            print(f"🔴 ERROR: xacro failed for {robot_name}. Fallback triggered.")
            continue

        # RSP
        rsp = Node(
            package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher',
            namespace=namespace, output='screen',
            parameters=[{'robot_description': open(tmp_urdf, 'r').read(), 'use_sim_time': use_sim_time, 'frame_prefix': f'{namespace}/'}]
        )
        
        # Spawn Entity
        spawn = Node(
            package='gazebo_ros', executable='spawn_entity.py', name='spawn_entity',
            namespace=namespace, output='screen',
            arguments=['-file', tmp_urdf, '-entity', robot_name, '-x', str(1.0 + i*2.0), '-y', '5.0', '-z', '0.6']
        )
        
        # Logic Node
        logic_node = Node(
            package='formicabot_ros2', executable='robot_node', name='robot_node',
            namespace=namespace, output='screen',
            parameters=[{'robot_id': i, 'start_x': 1.0 + i*2.0, 'start_y': 5.0, 'use_sim_time': use_sim_time}]
        )
        
        ld.add_action(rsp)
        ld.add_action(spawn)
        ld.add_action(logic_node)

    return ld
