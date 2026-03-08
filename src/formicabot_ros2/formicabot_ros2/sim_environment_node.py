import sys
import os

# --- ROS 2 Environment Sanitizer ---
sys.path = [p for p in sys.path if 'noetic' not in p.lower()]
humble_paths = [
    '/opt/ros/humble/lib/python3.10/site-packages',
    '/opt/ros/humble/local/lib/python3.10/dist-packages'
]
for p in reversed(humble_paths):
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import os
import json
from datetime import datetime

# Import the core physics components
from formicabot_ros2.core.config import Config
from formicabot_ros2.core.environment import Environment
from formicabot_ros2.core.pheromone import PheromoneGrid
from formicabot_ros2.core.optics_clustering import OPTICSRoleDifferentiator

class SimEnvironmentNode(Node):
    """
    Central environment node bridging physical simulation and ROS 2 topics.
    Maintains the pheromone grid and obstacles/targets, and broadcasts them
    for visualization via RViz.
    """

    def __init__(self):
        super().__init__('sim_environment_node')
        
        # Load configuration
        self.cfg = Config()
        
        # ROS Parameters
        self.declare_parameter('n_robots', 10)
        self.n_robots = self.get_parameter('n_robots').value
        
        self.env = Environment(self.cfg)
        self.pheromone = PheromoneGrid(self.cfg.env.width, self.cfg.env.height, self.cfg.env.cell_size, self.cfg.pheromone)
        
        # Swarm Coordination (OPTICS)
        self.optics = OPTICSRoleDifferentiator(self.cfg.optics, self.n_robots)
        
        # ROS 2 Timers (Sim rate: 10 Hz)
        self.timer_period = self.cfg.pheromone.dt
        self.timer = self.create_timer(self.timer_period, self._sim_step_callback)
        
        # Publishers for visualization
        self.pub_opt_map = self.create_publisher(OccupancyGrid, '/pheromone/optical/map', 10)
        self.pub_chem_map = self.create_publisher(OccupancyGrid, '/pheromone/chemical/map', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/environment_markers', 10)
        self.pub_roles = self.create_publisher(String, '/swarm/roles', 10)

        # Dynamic Subscribers for Swarm Telemetry
        self.subs_behavior = []
        for i in range(self.n_robots):
            topic = f'/robot_{i}/behavior_summary'
            sub = self.create_subscription(
                String, 
                topic, 
                lambda msg, rid=i: self._behavior_callback(msg, rid), 
                10
            )
            self.subs_behavior.append(sub)

        # Simulation Time
        self.sim_time = 0.0
        self.step_count = 0
        
        # Results Path (parameter driven for flexibility)
        self.declare_parameter('results_dir', '~/Documents/formicabot_ws/simulation_results')
        res_dir = self.get_parameter('results_dir').value
        self.results_dir = os.path.expanduser(res_dir)
        
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            self.get_logger().info(f"Results will be saved to: {self.results_dir}")
        except Exception as e:
            self.get_logger().error(f"Failed to create results dir: {str(e)}")

    def _behavior_callback(self, msg: String, robot_id: int):
        """Parse behavioral data and update OPTICS profiler."""
        try:
            # Format: "speed:turn_rate:deposited:state:interactions"
            parts = msg.data.split(':')
            speed = float(parts[0])
            turn_rate = float(parts[1])
            deposited = bool(int(parts[2]))
            state = parts[3]
            interactions = int(parts[4])
            
            self.optics.update(robot_id, speed, turn_rate, deposited, state, interactions)
        except (ValueError, IndexError):
            pass

    def _sim_step_callback(self):
        """Advances physical environment physics: Pheromone Evaporation"""
        self.sim_time += self.timer_period
        self.step_count += 1
        
        # Evaporate pheromones
        self.pheromone.evaporate(self.timer_period)
        self.pheromone.update_power_accounting(self.timer_period)
        
        # OPTICS Clustering Interval (Emergent Role Assignment)
        if self.step_count % self.cfg.optics.cluster_interval == 0:
            self.get_logger().info("Running Swarm OPTICS Clustering...")
            role_assignments = self.optics.run_clustering(self.step_count)
            
            # Broadcast new roles to individual robots
            for rid, role in role_assignments.items():
                rmsg = String()
                rmsg.data = f"{rid}:{role}"
                self.pub_roles.publish(rmsg)
            
            # Save periodic snapshot of swarm health
            self._save_results()

        # Broadcast visualisations every 1.0 seconds (1 Hz) to save bandwidth
        if int(self.sim_time * 10) % 10 == 0:
            self._publish_pheromone_grid()
            self._publish_environment_markers()

    def _publish_pheromone_grid(self):
        """Converts the internal Numpy 2D grids to ROS OccupancyGrid msgs"""
        stamp = self.get_clock().now().to_msg()
        opt, chem = self.pheromone.to_array()
        
        for name, grid_data, pub in [("optical", opt, self.pub_opt_map), ("chemical", chem, self.pub_chem_map)]:
            msg = OccupancyGrid()
            msg.header.stamp = stamp
            msg.header.frame_id = "map"
            
            msg.info.resolution = self.cfg.env.cell_size
            msg.info.width = self.pheromone.nx
            msg.info.height = self.pheromone.ny
            msg.info.origin.position.x = 0.0
            msg.info.origin.position.y = 0.0
            msg.info.origin.position.z = 0.0
            msg.info.origin.orientation.w = 1.0
            
            # Map concentration to 0-100 scale for RViz
            max_val = self.cfg.pheromone.optical_max if name == "optical" else self.cfg.pheromone.chemical_max
            scaled = np.clip((grid_data / max_val) * 100, 0, 100).astype(np.int8)
            msg.data = scaled.flatten('F').tolist() # Fortran order to match (x,y)
            
            pub.publish(msg)

    def _publish_environment_markers(self):
        """Publishes static obstacles, targets, and the nest as RViz markers"""
        msg = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        
        # Nest
        nest_marker = Marker()
        nest_marker.header.frame_id = "map"
        nest_marker.header.stamp = stamp
        nest_marker.ns = "nest"
        nest_marker.id = 0
        nest_marker.type = Marker.CYLINDER
        nest_marker.action = Marker.ADD
        nest_marker.pose.position.x = float(self.cfg.env.nest_pos[0])
        nest_marker.pose.position.y = float(self.cfg.env.nest_pos[1])
        nest_marker.pose.position.z = 0.05
        nest_marker.pose.orientation.w = 1.0
        nest_marker.scale.x = 0.5
        nest_marker.scale.y = 0.5
        nest_marker.scale.z = 0.1
        nest_marker.color.r = 0.0
        nest_marker.color.g = 1.0
        nest_marker.color.b = 0.0
        nest_marker.color.a = 0.8
        msg.markers.append(nest_marker)
        
        self.pub_markers.publish(msg)


    def _save_results(self):
        """Saves clustering and swarm statistics to a JSON file."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "sim_time": self.sim_time,
            "step_count": self.step_count,
            "swarm_stats": self.optics.get_stats(),
            "cluster_history": self.optics.get_cluster_history()
        }
        
        filename = os.path.join(self.results_dir, "swarm_results_latest.json")
        try:
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=4)
            self.get_logger().info(f"Results saved to: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save results: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = SimEnvironmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
