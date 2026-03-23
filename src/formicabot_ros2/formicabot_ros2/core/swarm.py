"""
swarm.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Swarm Coordinator
"""

import numpy as np
from typing import List, Dict

from formicabot_ros2.core.config import Config
from formicabot_ros2.core.environment import Environment
from formicabot_ros2.core.pheromone import PheromoneGrid
from formicabot_ros2.core.robot import FormicaBot
from formicabot_ros2.core.optics_clustering import OPTICSRoleDifferentiator


class SwarmSimulation:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        
        self.env = Environment(cfg)
        self.pheromone = PheromoneGrid(cfg.env.width, cfg.env.height, cfg.env.cell_size, cfg.pheromone)
        self.optics = OPTICSRoleDifferentiator(cfg.optics, cfg.swarm.n_robots, self.rng)
        
        # Initialize robots around nest
        self.robots: List[FormicaBot] = []
        for i in range(cfg.swarm.n_robots):
            offset_x = self.rng.uniform(-0.5, 0.5)
            offset_y = self.rng.uniform(-0.5, 0.5)
            pos = np.array([cfg.env.nest_pos[0] + offset_x, cfg.env.nest_pos[1] + offset_y])
            self.robots.append(FormicaBot(i, pos, cfg, self.rng))
            
        self.time_sec = 0.0
        self.timestep = 0
        
        # Metrics
        self.total_food_collected = 0

    def step(self):
        """Advance simulation by one timestep."""
        dt = self.cfg.pheromone.dt
        self.time_sec += dt
        
        # 1. Pheromone evaporation
        self.pheromone.evaporate(dt)
        self.pheromone.update_power_accounting(dt)
        
        # 2. Update all robots
        for robot in self.robots:
            # Check dropoff before step to count food
            was_carrying = robot.carrying
            
            robot.step(
                time_sec=self.time_sec,
                dt=dt,
                pheromone=self.pheromone,
                obstacles=self.env.get_obstacles(),
                targets=self.env.get_targets()
            )
            
            # Count dropoffs
            if was_carrying and not robot.carrying:
                self.total_food_collected += 1
                
            # Inter-robot repulsion (simple kinematic)
            for other_robot in self.robots:
                if other_robot.id == robot.id: continue
                dist = np.linalg.norm(robot.pos - other_robot.pos)
                if dist < self.cfg.swarm.robot_radius * 2:
                    repulse = (robot.pos - other_robot.pos) / (dist + 1e-9)
                    robot.pos += repulse * 0.05
            
            # Physics boundary clamping
            robot.pos[0] = np.clip(robot.pos[0], self.cfg.swarm.robot_radius, self.env.w - self.cfg.swarm.robot_radius)
            robot.pos[1] = np.clip(robot.pos[1], self.cfg.swarm.robot_radius, self.env.h - self.cfg.swarm.robot_radius)
            
            # 3. OPTICS Profiling
            if robot.power.current_draw_w > 0: # Only profile active robots
                # Count neighbors within interaction radius
                n_inter = sum(1 for r in self.robots 
                              if r.id != robot.id and np.linalg.norm(robot.pos - r.pos) < self.cfg.swarm.interaction_radius)
                
                self.optics.update(
                    robot_id=robot.id,
                    speed=robot.last_speed,
                    turn_rate=robot.last_turn,
                    deposited=robot.deposited_this_step,
                    state=robot.state,
                    n_interactions=n_inter
                )
                
        # 4. OPTICS Clustering Execution
        if self.optics.should_recluster(self.timestep):
            roles_dict = self.optics.run_clustering(self.timestep)
            for rid, role in roles_dict.items():
                self.robots[rid].role = role
                
        self.timestep += 1

    def run(self, max_steps: int = None, progress_bar: bool = True):
        """Run simulation entirely."""
        steps = max_steps or self.cfg.total_timesteps
        
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(steps), desc="Simulating FormicaBot Swarm")
            except ImportError:
                iterator = range(steps)
        else:
            iterator = range(steps)
            
        for _ in iterator:
            self.step()
            
        return self.get_summary()

    def get_summary(self) -> Dict:
        """Return end-of-run statistics."""
        avg_soc = np.mean([r.power.get_soc() for r in self.robots])
        avg_power = np.mean([sum(r.power.power_history)/max(1, len(r.power.power_history)) for r in self.robots])
        return {
            "time_sec": self.time_sec,
            "timesteps": self.timestep,
            "total_food": self.total_food_collected,
            "collection_rate": self.total_food_collected / max(1.0, self.time_sec),
            "avg_soc": float(avg_soc),
            "avg_power_w": float(avg_power),
            "pheromone_stats": self.pheromone.get_stats(),
            "optics_stats": self.optics.get_stats()
        }
