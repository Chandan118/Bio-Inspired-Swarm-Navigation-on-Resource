"""
aco_foraging.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Ant Colony Optimisation (ACO) Foraging Algorithm
"""

import numpy as np
from typing import List, Tuple, Optional

from formicabot_ros2.core.config import ACOConfig, EnvConfig


class ACOForaging:
    """
    ACO-based trajectory planner for a single robot.

    The robot maintains a candidate set of move directions (discretised into
    N_dirs directions), scores them via ACO, and selects the next heading.

    Parameters
    ----------
    cfg_aco  : ACOConfig
    cfg_env  : EnvConfig
    rng      : np.random.Generator  (seeded externally for reproducibility)
    """

    N_DIRS = 16  # Directional resolution for neighbour selection

    def __init__(
        self,
        cfg_aco: ACOConfig = None,
        cfg_env: EnvConfig = None,
        rng: np.random.Generator = None,
    ):
        self.cfg = cfg_aco or ACOConfig()
        self.env = cfg_env or EnvConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        # Role-specific weights (overridden by OPTICS clustering)
        self.w_chemo = self.cfg.w_chemo
        self.w_random = self.cfg.w_random
        self.w_wall = self.cfg.w_wall

        # Local ACO trail strengths (unit-normalised per-iteration)
        self._local_tau = np.ones(self.N_DIRS, dtype=np.float64)

        # Statistics
        self.total_deposits = 0
        self.total_steps = 0
        self.exploitation_count = 0
        self.exploration_count = 0

    # ------------------------------------------------------------------
    # Role parameter injection
    # ------------------------------------------------------------------
    def set_role(self, role: str):
        """
        Apply role-specific ACO weights from OPTICS clustering.
        role ∈ {'scout', 'worker', 'default'}
        """
        if role == "scout":
            self.w_chemo = self.cfg.scout_w_chemo
            self.w_random = self.cfg.scout_w_random
        elif role == "worker":
            self.w_chemo = self.cfg.worker_w_chemo
            self.w_random = self.cfg.worker_w_random
        else:
            self.w_chemo = self.cfg.w_chemo
            self.w_random = self.cfg.w_random
        # Wall avoidance always fills the remainder
        self.w_wall = max(0.0, 1.0 - self.w_chemo - self.w_random)

    # ------------------------------------------------------------------
    # Direction utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _dir_to_vec(dir_idx: int, n_dirs: int = N_DIRS) -> np.ndarray:
        angle = 2 * np.pi * dir_idx / n_dirs
        return np.array([np.cos(angle), np.sin(angle)])

    @staticmethod
    def _vec_to_dir(vec: np.ndarray, n_dirs: int = N_DIRS) -> int:
        angle = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
        return int(round(angle / (2 * np.pi) * n_dirs)) % n_dirs

    # ------------------------------------------------------------------
    # Heuristic (inverse distance to nest or target)
    # ------------------------------------------------------------------
    def _heuristic(
        self,
        pos: np.ndarray,
        heading: np.ndarray,
        goal: np.ndarray,
        carrying: bool,
    ) -> np.ndarray:
        """
        η_ij = cosine similarity between direction d and (goal - pos).
        Returning robots use nest as goal; searching robots use pheromone gradient.
        """
        to_goal = goal - pos
        dist = np.linalg.norm(to_goal) + 1e-9
        to_goal_unit = to_goal / dist

        eta = np.zeros(self.N_DIRS)
        for d in range(self.N_DIRS):
            dvec = self._dir_to_vec(d)
            # Cosine similarity: higher = better aligned with goal
            eta[d] = max(0.0, float(np.dot(dvec, to_goal_unit)))
        # Normalise
        s = eta.sum()
        if s > 1e-9:
            eta /= s
        else:
            eta[:] = 1.0 / self.N_DIRS
        return eta

    # ------------------------------------------------------------------
    # Main step function
    # ------------------------------------------------------------------
    def select_direction(
        self,
        pos: np.ndarray,
        current_heading: float,
        pheromone_readings: np.ndarray,
        pheromone_gradient: np.ndarray,
        obstacles: np.ndarray,
        goal: np.ndarray,
        carrying: bool = False,
        quality: float = 1.0,
    ) -> Tuple[int, float]:
        """
        Select next movement direction using ACO probability rule.

        Parameters
        ----------
        pos               : Current [x, y] in metres.
        current_heading   : Current heading in radians.
        pheromone_readings: [N_DIRS] local pheromone strengths (already sensed).
        pheromone_gradient: [2] unit vector pointing toward maximum pheromone.
        obstacles         : [N_DIRS] binary obstacle flags (1 = blocked).
        goal              : [2] target position (nest when returning).
        carrying          : True when robot carries a payload.
        quality           : Target quality (for probabilistic deposit).

        Returns
        -------
        chosen_dir : int  direction index (0 .. N_DIRS-1)
        probability: float  selection probability of chosen direction
        """
        self.total_steps += 1

        # Build pheromone component τ^alpha
        tau = np.clip(pheromone_readings, self.cfg.tau_min, self.cfg.tau_max)
        tau_component = tau ** self.cfg.alpha

        # Build heuristic component η^beta
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)])
        eta = self._heuristic(pos, heading_vec, goal, carrying)
        eta_component = eta ** self.cfg.beta

        # Random exploration component (uniform)
        random_component = np.ones(self.N_DIRS) / self.N_DIRS

        # Combined score
        score = (
            self.w_chemo * tau_component
            + (1 - self.w_chemo - self.w_random) * eta_component
            + self.w_random * random_component
        )

        # Mask blocked directions
        score = score * (1 - obstacles)
        score_sum = score.sum()
        # Re-calculating score to explicitly use w_wall if intended
        # The role-setting defines: w_wall = max(0.0, 1.0 - self.w_chemo - self.w_random)
        score = (
            self.w_chemo * tau_component
            + self.w_random * random_component
            + self.w_wall * eta_component 
        )
        
        # Mask blocked directions
        score = score * (1 - obstacles)
        score_sum = score.sum()
        
        if score_sum < 1e-9:
            # All directions blocked: reverse
            rev_dir = self._vec_to_dir(-heading_vec)
            return rev_dir, 1.0 / self.N_DIRS

        probs = score / score_sum

        # ACS exploitation step: with probability q0, take argmax
        if self.rng.random() < self.cfg.q0:
            chosen = int(np.argmax(probs))
            self.exploitation_count += 1
        else:
            chosen = int(self.rng.choice(self.N_DIRS, p=probs))
            self.exploration_count += 1

        # Local trail update (online)
        delta_tau = quality * self.cfg.alpha
        self._local_tau[chosen] = np.clip(
            self._local_tau[chosen] + delta_tau, self.cfg.tau_min, self.cfg.tau_max
        )
        # Evaporate other directions
        evap = (1 - self.cfg.rho)
        self._local_tau = np.clip(
            self._local_tau * evap, self.cfg.tau_min, self.cfg.tau_max
        )

        return chosen, float(probs[chosen])

    # ------------------------------------------------------------------
    # Deposit decision
    # ------------------------------------------------------------------
    def should_deposit(self, pheromone_val: float, target_quality: float = 1.0) -> bool:
        """
        Returns True if the robot should deposit pheromone at current position.
        Workers deposit on both outward and return legs.
        Scouts only deposit when returning with payload.
        """
        # Probabilistic reinforcement proportional to quality
        deposit_prob = np.clip(target_quality * 0.9, 0.1, 1.0)
        should = self.rng.random() < deposit_prob
        if should:
            self.total_deposits += 1
        return should

    # ------------------------------------------------------------------
    # Rerouting decision (dynamic obstacle)
    # ------------------------------------------------------------------
    def compute_reroute_direction(
        self,
        pos: np.ndarray,
        blocked_dir: int,
        obstacle_pos: np.ndarray,
    ) -> int:
        """
        When a direction is suddenly blocked by a dynamic obstacle,
        compute a valid detour direction via stigmergic rerouting.
        Selects the highest-scoring unblocked adjacent direction.
        """
        left = (blocked_dir - 1) % self.N_DIRS
        right = (blocked_dir + 1) % self.N_DIRS
        # Prefer the direction with higher local trail strength
        if self._local_tau[left] >= self._local_tau[right]:
            return left
        return right

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        total = max(1, self.exploitation_count + self.exploration_count)
        return {
            "total_steps": self.total_steps,
            "total_deposits": self.total_deposits,
            "exploitation_rate": self.exploitation_count / total,
            "exploration_rate": self.exploration_count / total,
        }
