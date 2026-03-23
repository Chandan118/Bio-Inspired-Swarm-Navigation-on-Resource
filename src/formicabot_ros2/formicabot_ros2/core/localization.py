"""
localization.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Multi-Modal Localization and Mapping (SLAM)
"""

import numpy as np
from typing import Tuple, List, Optional
import math

from formicabot_ros2.core.config import SLAMConfig


class ExtendedKalmanFilter:
    """
    2D EKF fusing wheel odometry and IMU heading.
    State: x = [x_pos, y_pos, theta]^T
    """

    def __init__(self, init_pos: np.ndarray, init_heading: float, cfg: SLAMConfig):
        self.cfg = cfg
        self.x = np.array([init_pos[0], init_pos[1], init_heading], dtype=np.float64)
        # Covariance matrix P
        self.P = np.eye(3) * 0.01

        # Process noise Q
        self.Q = np.diag([
            self.cfg.ekf_process_noise_pos,
            self.cfg.ekf_process_noise_pos,
            self.cfg.ekf_process_noise_yaw
        ])

        # Measurement noise R (IMU heading)
        self.R = np.array([[self.cfg.ekf_obs_noise_imu]])

    def predict(self, dr: float, dtheta_odom: float):
        """
        Predict step based on wheel odometry.
        dr: forward displacement [m]
        dtheta_odom: heading change from wheels [rad]
        """
        theta = self.x[2]
        
        # State transition function F(x, u)
        # x_new = x + dr * cos(theta)
        # y_new = y + dr * sin(theta)
        # theta_new = theta + dtheta_odom
        
        self.x[0] += dr * np.cos(theta)
        self.x[1] += dr * np.sin(theta)
        self.x[2] += dtheta_odom
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        # Jacobian of F wrt x (F_x)
        F_x = np.array([
            [1, 0, -dr * np.sin(theta)],
            [0, 1,  dr * np.cos(theta)],
            [0, 0, 1]
        ])

        # Update covariance
        self.P = F_x @ self.P @ F_x.T + self.Q

    def update_imu(self, theta_imu: float):
        """
        Update step using IMU absolute heading.
        """
        # Measurement residual
        y = theta_imu - self.x[2]
        y = (y + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Measurement matrix H = [0, 0, 1]
        H = np.array([[0, 0, 1]])

        # Innovation covariance S = H P H^T + R
        S = H @ self.P @ H.T + self.R

        # Kalman gain K = P H^T S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + (K @ np.array([y])).flatten()
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def get_state(self) -> Tuple[np.ndarray, float]:
        return self.x[0:2].copy(), self.x[2]

    def get_position_uncertainty(self) -> float:
        """Returns standard deviation of position estimation."""
        return np.sqrt(self.P[0, 0] + self.P[1, 1])


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Standard Bresenham integer ray-casting algorithm.
    Returns list of cells traversed from (x0, y0) to (x1, y1) inclusive.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


class OccupancyGrid:
    """
    Probabilistic occupancy grid using log-odds representation.
    """

    def __init__(self, width: float, height: float, cfg: SLAMConfig):
        self.cfg = cfg
        self.nx = int(np.ceil(width / self.cfg.grid_resolution))
        self.ny = int(np.ceil(height / self.cfg.grid_resolution))
        
        # Initialize grid with prior log-odds
        prior_log_odds = np.log(self.cfg.p_occ_prior / (1.0 - self.cfg.p_occ_prior))
        self.log_odds = np.full((self.nx, self.ny), prior_log_odds, dtype=np.float32)

        # Precompute sensor model log-odds
        self.l_occ = np.log(self.cfg.p_occ_hit / (1.0 - self.cfg.p_occ_hit))
        self.l_free = np.log(self.cfg.p_occ_miss / (1.0 - self.cfg.p_occ_miss))

    def _xy_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(x / self.cfg.grid_resolution)
        cy = int(y / self.cfg.grid_resolution)
        return cx, cy

    def update_ray(self, origin_x: float, origin_y: float, hit_x: float, hit_y: float, max_range: bool = False):
        """
        Update grid cells along a single LiDAR ray cast.
        """
        cx0, cy0 = self._xy_to_cell(origin_x, origin_y)
        cx1, cy1 = self._xy_to_cell(hit_x, hit_y)
        
        # Clamp endpoints to grid boundaries
        cx0 = np.clip(cx0, 0, self.nx - 1)
        cy0 = np.clip(cy0, 0, self.ny - 1)
        cx1 = np.clip(cx1, 0, self.nx - 1)
        cy1 = np.clip(cy1, 0, self.ny - 1)

        ray_cells = bresenham_line(cx0, cy0, cx1, cy1)
        
        for i, (cx, cy) in enumerate(ray_cells):
            # Check bounds just in case
            if 0 <= cx < self.nx and 0 <= cy < self.ny:
                is_endpoint = (i == len(ray_cells) - 1)
                
                if is_endpoint and not max_range:
                    # Obstacle hit
                    self.log_odds[cx, cy] += self.l_occ
                else:
                    # Free space along the ray
                    self.log_odds[cx, cy] += self.l_free
                
                # Clamp to prevent overconfidence
                self.log_odds[cx, cy] = np.clip(
                    self.log_odds[cx, cy], 
                    self.cfg.log_odds_min, 
                    self.cfg.log_odds_max
                )

    def get_probabilities(self) -> np.ndarray:
        """Convert log-odds back to probabilities [0, 1]."""
        return 1.0 - (1.0 / (1.0 + np.exp(self.log_odds)))

    def fuse_map(self, other_log_odds: np.ndarray):
        """
        Bayesian fusion of another robot's map.
        In log-odds form, this is simple addition (assuming conditional independence).
        """
        self.log_odds += other_log_odds
        self.log_odds = np.clip(self.log_odds, self.cfg.log_odds_min, self.cfg.log_odds_max)


class LocalizationSystem:
    """
    High-level manager for the robot's localization stack.
    """

    def __init__(self, width: float, height: float, init_pos: np.ndarray, init_heading: float, cfg: SLAMConfig = None):
        self.cfg = cfg or SLAMConfig()
        self.ekf = ExtendedKalmanFilter(init_pos, init_heading, self.cfg)
        self.grid = OccupancyGrid(width, height, self.cfg)
        
        # Ground truth tracking for error calculation
        self.true_pos = init_pos.copy()
        
        # Active localizer flag (for simulated ICP corrections)
        self.is_lost = False

    def step_odometry(self, dr: float, dtheta: float, imu_heading: float):
        """Standard high-rate odometry and IMU update."""
        self.ekf.predict(dr, dtheta)
        self.ekf.update_imu(imu_heading)
        
        # Check uncertainty threshold as described in paper
        uncert = self.ekf.get_position_uncertainty()
        self.is_lost = uncert > self.cfg.uncertainty_threshold

    def process_lidar_scan(self, 
                           pos_est: np.ndarray, 
                           heading_est: float, 
                           ranges: np.ndarray, 
                           angles: np.ndarray, 
                           max_range: float):
        """
        Process a 360-degree LiDAR scan into the occupancy grid.
        ranges: array of distances
        angles: array of angles relative to robot forward vector
        """
        for r, a in zip(ranges, angles):
            world_angle = heading_est + a
            hit_x = pos_est[0] + r * np.cos(world_angle)
            hit_y = pos_est[1] + r * np.sin(world_angle)
            
            # If ray hit max range, we didn't see an obstacle, just free space
            is_max = (r >= max_range * 0.99)
            
            self.grid.update_ray(
                origin_x=pos_est[0],
                origin_y=pos_est[1],
                hit_x=hit_x,
                hit_y=hit_y,
                max_range=is_max
            )

    def apply_icp_correction(self, true_pos: np.ndarray, rse_error_m: float = 0.05):
        """
        Simulate a successful ICP scan match against global coordinate frame.
        In the real system, ICP alignes current scan to the occupancy grid.
        Here we inject a low-variance noisy measurement of true position.
        """
        # Simulated ICP alignment provides a noisy global position measurement
        meas_x = true_pos[0] + np.random.normal(0, rse_error_m)
        meas_y = true_pos[1] + np.random.normal(0, rse_error_m)
        
        # Update EKF state directly (simplified for simulation)
        self.ekf.x[0] = meas_x
        self.ekf.x[1] = meas_y
        
        # Reset covariance indicating high confidence
        self.ekf.P[0,0] = rse_error_m**2
        self.ekf.P[1,1] = rse_error_m**2
        self.is_lost = False

    def get_pose(self) -> Tuple[np.ndarray, float]:
        return self.ekf.get_state()
