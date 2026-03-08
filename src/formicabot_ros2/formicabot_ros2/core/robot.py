"""
Single Robot State Machine
==========================
Defines the FormicaBot agent behaviour (Section 4.5).
States: 'explore', 'exploit', 'return', 'recharge'.
"""

import numpy as np
from typing import Tuple, Dict, Optional

from formicabot_ros2.core.config import Config
from formicabot_ros2.core.aco_foraging import ACOForaging
from formicabot_ros2.core.power_manager import PowerManager
from formicabot_ros2.core.localization import LocalizationSystem
from formicabot_ros2.core.pheromone import PheromoneGrid
from formicabot_ros2.core.mobilenet_recognition import TargetRecognitionEngine


class FormicaBot:
    """Intelligent agent simulating one physical ant robot."""

    def __init__(
        self,
        robot_id: int,
        start_pos: np.ndarray,
        cfg: Config,
        rng: np.random.Generator,
    ):
        self.id = robot_id
        self.cfg = cfg
        self.rng = rng

        # Physics/State
        self.pos = start_pos.copy()
        self.heading = rng.uniform(-np.pi, np.pi)
        self.state = "explore"      # 'explore', 'exploit', 'return'
        self.role = "default"       # Set by OPTICS
        self.carrying = False       # Carrying virtual payload
        self.target_quality = 0.0

        # Subsystems
        self.aco = ACOForaging(cfg.aco, cfg.env, rng)
        self.power = PowerManager(cfg.swarm.battery_wh, cfg.power)
        self.slam = LocalizationSystem(cfg.env.width, cfg.env.height, start_pos, self.heading, cfg.slam)
        self.cnn = TargetRecognitionEngine(cfg.cnn, rng, use_mock=True)

        # Action history for OPTICS
        self.last_speed = 0.0
        self.last_turn = 0.0
        self.deposited_this_step = False
        self.n_interactions = 0

    def step(
        self,
        time_sec: float,
        dt: float,
        pheromone: PheromoneGrid,
        obstacles: np.ndarray,
        targets: np.ndarray,  # shape [N, 3]: x, y, quality
    ):
        """Execute one simulation timestep for this robot."""
        # 1. State Transitions based on battery
        if self.power.require_return() and self.state != "return":
            self.state = "return"

        # 2. Sensing & Localization
        true_pos = self.pos.copy() # (simulated ground truth)
        
        # Local Pheromone neighborhood & gradient
        local_phero, dom_mode = pheromone.read_neighbourhood(self.pos[0], self.pos[1])
        gas_active = (dom_mode == "chemical")
        self.power.gas_active = gas_active
        grad = pheromone.gradient_at(self.pos[0], self.pos[1])

        # 3. Target Detection (CNN)
        # Check if we are physically near any target (ground truth for mock)
        near_tgt = False
        quality_found = 0.0
        for tx, ty, tq in targets:
            if np.linalg.norm(self.pos - [tx, ty]) < self.cfg.swarm.sensor_range:
                near_tgt = True
                quality_found = tq
                break

        self.power.kinect_active = near_tgt
        detected, cls_id, conf = self.cnn.infer(
            is_near_target=near_tgt,
            deprioritise_camera=self.power.require_return(),
            sim_timestep=time_sec
        )

        # State transition: explore -> exploit / return
        if self.state in ["explore", "exploit"] and detected and not self.carrying:
            self.carrying = True
            self.target_quality = quality_found
            self.state = "return"

        # Check if returned to nest
        dist_to_nest = np.linalg.norm(self.pos - np.array(self.cfg.env.nest_pos))
        if dist_to_nest < 0.5:
            if self.carrying:
                self.carrying = False
                self.target_quality = 0.0
                # If we just dropped off, exploit the trail we just built
                self.state = "exploit"
            elif self.power.require_return():
                # Recharging (simplified)
                self.power.energy_j = self.power.capacity_j
                self.power.is_returning_home = False
                self.state = "explore"

        # 4. ACO Planning & Movement
        self.aco.set_role(self.role)
        
        goal = np.array(self.cfg.env.nest_pos) if self.state == "return" else self.pos + grad
        
        # Discretise local obstacles (simulated 16-ray LIDAR)
        obs_mask = np.zeros(self.aco.N_DIRS)
        # (Simplified: assume boundaries are obstacles; internal dynamic obstacles handled via swarm.py)
        if self.pos[0] < 0.5:  obs_mask[8]  = 1.0 # West
        if self.pos[0] > self.cfg.env.width - 0.5: obs_mask[0]  = 1.0 # East
        if self.pos[1] < 0.5:  obs_mask[12] = 1.0 # South
        if self.pos[1] > self.cfg.env.height - 0.5: obs_mask[4] = 1.0 # North

        # Read surrounding pheromone
        p_readings = np.ones(self.aco.N_DIRS)
        for i in range(self.aco.N_DIRS):
            dvec = self.aco._dir_to_vec(i)
            tx = self.pos[0] + dvec[0] * self.cfg.swarm.sensor_range
            ty = self.pos[1] + dvec[1] * self.cfg.swarm.sensor_range
            val, _ = pheromone.read_effective(tx, ty, self.rng)
            p_readings[i] = max(0.01, val)

        chosen_dir, prob = self.aco.select_direction(
            self.pos, self.heading, p_readings, grad, obs_mask, goal, self.carrying, self.target_quality
        )

        # Execute move
        target_vel = self.aco._dir_to_vec(chosen_dir)
        speed = self.cfg.swarm.fast_speed if self.state == "explore" and dom_mode == "optical" else self.cfg.swarm.max_speed
        move_vec = target_vel * speed * dt
        
        new_pos = self.pos + move_vec
        new_heading = np.arctan2(target_vel[1], target_vel[0])
        
        # Update physics
        dtheta = new_heading - self.heading
        dr = np.linalg.norm(move_vec)
        self.pos = new_pos
        self.heading = new_heading
        self.last_speed = speed
        self.last_turn = dtheta

        # 5. Localisation / SLAM step
        self.slam.step_odometry(dr, dtheta, new_heading)
        
        # 6. Pheromone Deposition (Stigmergy)
        self.deposited_this_step = False
        if self.aco.should_deposit(local_phero, self.target_quality):
            # Scouts only deposit when returning. Workers deposit both ways.
            if self.role == "worker" or (self.role == "scout" and self.carrying):
                pheromone.deposit_reinforcement(self.pos[0], self.pos[1], quality=max(1.0, self.target_quality))
                self.deposited_this_step = True

        # 7. Energy update
        self.power.lidar_active = True # (Active while moving)
        self.power.update(dt, self.target_quality, near_tgt)

    def get_diagnostics(self) -> Dict:
        """Return full internal state for logging."""
        soc = self.power.get_soc()
        return {
            "id": self.id,
            "x": self.pos[0],
            "y": self.pos[1],
            "state": self.state,
            "role": self.role,
            "soc": soc,
            "carrying": int(self.carrying),
            "explore_rate": self.aco.exploration_count / max(1, self.aco.total_steps),
        }
