import sys
import os

# --- ROS 2 Environment Sanitizer ---
# Force-remove any ROS 1 (Noetic) paths from sys.path to prevent 'AttributeError'
sys.path = [p for p in sys.path if 'noetic' not in p.lower()]

# Ensure Humble paths and workspace paths are at the very front
humble_paths = [
    '/opt/ros/humble/lib/python3.10/site-packages',
    '/opt/ros/humble/local/lib/python3.10/dist-packages'
]
for p in reversed(humble_paths):
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import Marker

import numpy as np

# Import core algorithms
from formicabot_ros2.core.config import Config
from formicabot_ros2.core.aco_foraging import ACOForaging
from formicabot_ros2.core.mobilenet_recognition import TargetRecognitionEngine
from formicabot_ros2.core.localization import LocalizationSystem
from formicabot_ros2.core.power_manager import PowerManager

class FormicaBotNode(Node):
    """
    Independent ROS 2 Node for a single FormicaBot.
    Designed to run onboard the Jetson Orin Nano, hooking into physical sensors
    (or simulation topics) and publishing motor commands.
    """

    def __init__(self):
        super().__init__('robot_node')
        
        # ROS Parameters (default values for simulation)
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('start_x', 0.5)
        self.declare_parameter('start_y', 5.0)
        
        self.robot_id = self.get_parameter('robot_id').value
        start_x = self.get_parameter('start_x').value
        start_y = self.get_parameter('start_y').value
        
        self.cfg = Config()
        self.rng = np.random.default_rng(self.cfg.random_seed + self.robot_id)
        
        # Initialize internal state machine
        self.pos = np.array([start_x, start_y])
        self.heading = self.rng.uniform(-np.pi, np.pi)
        
        self.aco = ACOForaging(self.cfg.aco, self.cfg.env, self.rng)
        self.power = PowerManager(self.cfg.swarm.battery_wh, self.cfg.power)
        self.slam = LocalizationSystem(self.cfg.env.width, self.cfg.env.height, self.pos, self.heading, self.cfg.slam)
        self.cnn = TargetRecognitionEngine(self.cfg.cnn, self.rng, use_mock=True) # Switch to False for actual jetson
        
        # State
        self.role = "default"
        self.state = "explore"
        self.carrying = False
        self.target_quality = 0.0
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.role_pub = self.create_publisher(String, 'role', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'pose_estimated', 10)
        self.target_pub = self.create_publisher(String, 'target_detected', 10)
        self.behavior_pub = self.create_publisher(String, 'behavior_summary', 10)
        self.marker_pub = self.create_publisher(Marker, 'role_marker', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self._scan_callback, 10)
        self.image_sub = self.create_subscription(Image, 'camera/color/image_raw', self._image_callback, 10)
        self.role_sub = self.create_subscription(String, '/swarm/roles', self._role_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self._odom_callback, 10)

        # Timer: 10Hz Control Loop
        self.use_sim_time = self.get_parameter('use_sim_time').value if self.has_parameter('use_sim_time') else True
        self.timer_period = self.cfg.pheromone.dt
        self.timer = self.create_timer(self.timer_period, self._control_loop)
        
        self.get_logger().info(f"FormicaBot Node {self.robot_id} Initialized at ({start_x}, {start_y}).")

    def _scan_callback(self, msg: LaserScan):
        """Handle incoming LiDAR scans for the occupancy grid."""
        ranges = np.array(msg.ranges)
        # Create angles array matching ranges
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Pass to SLAM module
        est_pos, est_heading = self.slam.get_pose()
        self.slam.process_lidar_scan(est_pos, est_heading, ranges, angles, msg.range_max)

    def _odom_callback(self, msg: Odometry):
        """Update internal position and heading from Gazebo Odometry."""
        if self.use_sim_time:
            self.pos[0] = msg.pose.pose.position.x
            self.pos[1] = msg.pose.pose.position.y
            # Simplified heading extraction from quaternion
            q = msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.heading = np.arctan2(siny_cosp, cosy_cosp)

    def _role_callback(self, msg: String):
        """Update robot role based on central swarm coordination."""
        # msg format: "robot_id:role" (e.g. "0:scout")
        try:
            rid, role = msg.data.split(':')
            if int(rid) == self.robot_id:
                if role != self.role:
                    self.get_logger().info(f"[{self.robot_id}] Role updated to: {role}")
                    self.role = role
        except ValueError:
            pass

    def _image_callback(self, msg: Image):
        """Handle incoming RGB-D imagery for MobileNetV3 Target Recognition."""
        # Note: In a real system, use cv_bridge to convert msg to numpy array
        # Here we rely on the mock CNN internal timing.
        pass

    def _euler_to_quaternion(self, roll, pitch, yaw):
        """Helper to convert euler angles to quaternion [x, y, z, w]."""
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    def _control_loop(self):
        """Main control loop executing ACO Foraging and sending cmd_vel."""
        # 1. Power State Check
        if self.power.require_return() and self.state != "return":
            self.state = "return"
            self.get_logger().warn(f"[{self.robot_id}] Low battery, returning to nest.")

        # 2. Target Inference
        # In a real system, 'is_near_target' is driven by high pheromone levels
        near_target = self.role == "worker" and self.state == "exploit"
        detected, cls_id, conf = self.cnn.infer(
            is_near_target=near_target, 
            deprioritise_camera=self.power.require_return(),
            sim_timestep=self.get_clock().now().nanoseconds / 1e9
        )
        
        if self.state in ["explore", "exploit"] and detected and not self.carrying:
            self.carrying = True
            self.state = "return"
            self.get_logger().info(f"[{self.robot_id}] Target Detected! Returning.")
            
            # Publish detection event
            tmsg = String()
            tmsg.data = f"Robot {self.robot_id} found target (Class {cls_id})"
            self.target_pub.publish(tmsg)

        # 3. ACO Navigation
        self.aco.set_role(self.role)
        
        # Pheromone readings would come from topics/sensors in a fully distributed ROS setup.
        p_readings = np.ones(self.aco.N_DIRS) * self.cfg.aco.tau_min
        grad = np.array([0.0, 0.0])
        obs_mask = np.zeros(self.aco.N_DIRS)
        goal = np.array([0.5, 5.0]) if self.state == "return" else np.array([5.0, 5.0])
        
        chosen_dir, prob = self.aco.select_direction(
            self.pos, self.heading, p_readings, grad, obs_mask, goal, self.carrying, self.target_quality
        )
        
        # Determine if we should deposit (stigmergy)
        deposited = False
        if self.aco.should_deposit(p_readings[chosen_dir], self.target_quality):
            deposited = True
        
        # 4. Generate & Send Twist (Motor Command)
        target_vel = self.aco._dir_to_vec(chosen_dir)
        speed = self.cfg.swarm.fast_speed if self.state == "explore" else self.cfg.swarm.max_speed
        
        # Control Refinement: Reduce linear speed if angular error is high
        target_heading = np.arctan2(target_vel[1], target_vel[0])
        heading_err = target_heading - self.heading
        heading_err = (heading_err + np.pi) % (2*np.pi) - np.pi
        
        actual_speed = float(speed) * np.cos(heading_err)
        if actual_speed < 0: actual_speed = 0.0 # Don't go backwards in this mode

        twist = Twist()
        twist.linear.x = actual_speed
        twist.angular.z = float(heading_err * 2.0) 
        
        self.cmd_vel_pub.publish(twist)
        
        # Integration step (Only if not using Gazebo physics)
        if not self.use_sim_time:
            self.heading += twist.angular.z * self.timer_period
            self.pos[0] += twist.linear.x * np.cos(self.heading) * self.timer_period
            self.pos[1] += twist.linear.x * np.sin(self.heading) * self.timer_period

        # 4.5 Publish 3D Role Marker
        self._publish_role_marker()

        # 5. Publish Role Info & Telemetry
        msg = String()
        msg.data = self.role
        self.role_pub.publish(msg)
        
        # Publish Pose
        pmsg = PoseStamped()
        pmsg.header.stamp = self.get_clock().now().to_msg()
        pmsg.header.frame_id = "map"
        pmsg.pose.position.x = float(self.pos[0])
        pmsg.pose.position.y = float(self.pos[1])
        
        # Correct Quaternion
        q = self._euler_to_quaternion(0, 0, self.heading)
        pmsg.pose.orientation.x = q[0]
        pmsg.pose.orientation.y = q[1]
        pmsg.pose.orientation.z = q[2]
        pmsg.pose.orientation.w = q[3]
        self.pose_pub.publish(pmsg)

        # 6. Publish Behavioral Summary (for OPTICS coordination)
        bmsg = String()
        max_s = self.cfg.swarm.max_speed if self.cfg.swarm.max_speed > 0 else 1.0
        b_speed = speed / max_s
        b_turn = abs(twist.angular.z) / 2.0
        b_dep = 1 if deposited else 0
        b_state = self.state
        b_inter = 0 
        bmsg.data = f"{b_speed}:{b_turn}:{b_dep}:{b_state}:{b_inter}"
        self.behavior_pub.publish(bmsg)

    def _publish_role_marker(self):
        """Publishes a floating text marker above the robot for 3D visualization."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"robot_{self.robot_id}"
        marker.id = self.robot_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position floating above the robot
        marker.pose.position.x = float(self.pos[0])
        marker.pose.position.y = float(self.pos[1])
        marker.pose.position.z = 0.6  # Height above robot
        
        marker.scale.z = 0.2  # Text size
        
        role_upper = self.role.upper()
        marker.text = f"[{self.robot_id}] {role_upper}"
        
        # Color code: Scout=Blue, Worker=Green, Noise/Default=Gray
        marker.color.a = 1.0
        if self.role == "scout":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 0.5, 1.0
        elif self.role == "worker":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.7, 0.7, 0.7
            
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = FormicaBotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
