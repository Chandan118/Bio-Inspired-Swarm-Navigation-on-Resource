"""
2D Swarm Robotics Environment
===============================
Simulates the physical arena, targets, and static/dynamic obstacles.
"""

import numpy as np
from typing import List, Tuple, Dict

from formicabot_ros2.core.config import Config


class Environment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        
        # Dimensions
        self.w = cfg.env.width
        self.h = cfg.env.height
        
        # Target generator (Quality uniform between 1.0 and 5.0)
        self.targets = []
        for _ in range(cfg.env.n_targets):
            tx = self.rng.uniform(1.0, self.w - 1.0)
            ty = self.rng.uniform(1.0, self.h - 1.0)
            # Cannot spawn over nest
            while np.linalg.norm([tx - cfg.env.nest_pos[0], ty - cfg.env.nest_pos[1]]) < 2.0:
                tx = self.rng.uniform(1.0, self.w - 1.0)
                ty = self.rng.uniform(1.0, self.h - 1.0)
            tq = self.rng.uniform(1.0, 5.0)
            self.targets.append([tx, ty, tq])
        self.targets = np.array(self.targets)
        
        # Obstacles (cylindrical)
        self.obstacles = []
        for _ in range(cfg.env.n_obstacles):
            ox = self.rng.uniform(0.5, self.w - 0.5)
            oy = self.rng.uniform(0.5, self.h - 0.5)
            # Avoid nest and targets
            valid = True
            if np.linalg.norm([ox - cfg.env.nest_pos[0], oy - cfg.env.nest_pos[1]]) < 1.0:
                valid = False
            for tx, ty, tq in self.targets:
                if np.linalg.norm([ox - tx, oy - ty]) < 1.0:
                    valid = False
            if valid:
                self.obstacles.append([ox, oy])
                
        self.obstacles = np.array(self.obstacles)
        if len(self.obstacles) == 0:
            self.obstacles = np.zeros((0, 2))

    def get_targets(self) -> np.ndarray:
        return self.targets
        
    def get_obstacles(self) -> np.ndarray:
        return self.obstacles
        
    def check_collision(self, x: float, y: float, r: float) -> bool:
        """Simple circle-circle collision check vs static obstacles and walls."""
        if x < r or x > self.w - r or y < r or y > self.h - r:
            return True
        if len(self.obstacles) == 0: return False
        
        dists = np.linalg.norm(self.obstacles - [x, y], axis=1)
        return bool(np.any(dists < (r + self.cfg.env.obstacle_radius)))
