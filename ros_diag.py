"""
ros_diag.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Module for Ros Diag
"""

import sys
import os
import rclpy

print("--- ROS 2 Environment Diagnostic ---")
print(f"Python Version: {sys.version}")
print(f"Path: {sys.path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")

try:
    import geometry_msgs.msg
    print(f"geometry_msgs location: {geometry_msgs.msg.__file__}")
    from geometry_msgs.msg import Twist
    print(f"Twist type: {type(Twist)}")
    if hasattr(Twist, '_TYPE_SUPPORT'):
        print("Twist has _TYPE_SUPPORT: YES (Correct for ROS 2)")
    else:
        print("Twist has _TYPE_SUPPORT: NO (Likely ROS 1 or broken)")
except Exception as e:
    print(f"Failed to import geometry_msgs: {e}")

try:
    import rclpy.type_support
    import inspect
    lines = inspect.getsourcelines(rclpy.type_support.check_for_type_support)
    print("\n--- rclpy.type_support.check_for_type_support source snippet ---")
    for line in lines[0][:10]:
        print(line.strip())
except Exception as e:
    print(f"Failed to inspect rclpy: {e}")
