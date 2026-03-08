"""
Hierarchical Power Management System
======================================
Simulates FormicaBot's power architecture (Section 4.6).

Features:
  - DVFS (Dynamic Voltage & Frequency Scaling) for CPU/GPU.
  - Sensor power gating (Kinect, MQ-135, LiDAR).
  - Battery state-of-charge tracking.
  - Return-to-nest safety override.
"""

from typing import Dict, Tuple

from formicabot_ros2.core.config import PowerConfig


class PowerManager:
    def __init__(self, init_energy_wh: float, cfg: PowerConfig):
        self.cfg = cfg
        self.capacity_j = init_energy_wh * 3600.0
        self.energy_j = self.capacity_j
        
        # Subsystem states
        self.cpu_state = "nominal"  # idle, nominal, max
        self.gpu_state = "idle"     # idle, active
        self.kinect_active = False
        self.gas_active = False
        self.lidar_active = False
        self.motors_active = True
        self.speed_factor = 1.0     # 1.0 = nominal, >1.0 = fast
        
        # History
        self.power_history = []
        self.current_draw_w = 0.0
        self.is_returning_home = False

    def update(self, dt: float, target_quality: float = 0.0, target_proximity: bool = False):
        """
        Update power states and consume energy over delta-time dt.
        """
        # Predictive DVFS policy
        if target_proximity or target_quality > 0.5:
            # Anticipate CNN load
            self.cpu_state = "max"
            self.gpu_state = "active"
            self.kinect_active = True
        else:
            self.cpu_state = "nominal"
            self.gpu_state = "idle"
            self.kinect_active = False
            
        # Modulate motor power based on speed factor
        # Higher speed -> higher power curve (assume quadratic for drag)
        motor_pwr = self.cfg.motor_nominal_w * (self.speed_factor ** 2)
        if motor_pwr < 0.1:
            self.motors_active = False
            motor_pwr = 0.0
        else:
            self.motors_active = True
            
        # Sensor power calculation
        kinect_pwr = self.cfg.kinect_active_w if self.kinect_active else self.cfg.kinect_idle_w
        gas_pwr = self.cfg.gas_sensor_gated_w # Handled more precisely by global grid, this is baseline
        if self.gas_active:
             gas_pwr = self.cfg.gas_sensor_always_on_w
             
        lidar_pwr = self.cfg.lidar_active_w if self.lidar_active else self.cfg.lidar_standby_w
        
        # CPU/GPU logic
        cpu_pwr = {
            "idle": self.cfg.cpu_idle_w,
            "nominal": self.cfg.cpu_nominal_w,
            "max": self.cfg.cpu_max_w
        }[self.cpu_state]
        
        gpu_pwr = self.cfg.gpu_active_w if self.gpu_state == "active" else self.cfg.gpu_idle_w
        
        # Total draw
        self.current_draw_w = (
            motor_pwr + kinect_pwr + gas_pwr + lidar_pwr + 
            cpu_pwr + gpu_pwr + self.cfg.imu_w + self.cfg.wifi_w
        )
        
        # Consume energy
        consumed_j = self.current_draw_w * dt
        self.energy_j = max(0.0, self.energy_j - consumed_j)
        
        self.power_history.append(self.current_draw_w)
        
        # Safety check: remaining runtime
        if self.current_draw_w > 0:
            remaining_sec = self.energy_j / self.current_draw_w
            if remaining_sec < (self.cfg.battery_safety_min_runtime_min * 60.0):
                self.is_returning_home = True

    def get_soc(self) -> float:
        """State of Charge [0.0 - 1.0]"""
        return self.energy_j / self.capacity_j

    def require_return(self) -> bool:
        """Returns True if battery is critically low."""
        return self.is_returning_home
        
    def get_stats(self) -> Dict:
        avg_w = sum(self.power_history) / max(1, len(self.power_history))
        return {
            "soc": self.get_soc(),
            "avg_power_w": avg_w,
            "estimated_runtime_hr": (self.energy_j / max(1e-6, avg_w)) / 3600.0,
            "is_returning": self.is_returning_home
        }
