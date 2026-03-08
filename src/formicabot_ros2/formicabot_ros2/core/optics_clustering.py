"""
OPTICS-Based Behavioural Role Differentiation
===============================================
Implements the OPTICS (Ordering Points To Identify the Clustering Structure)
density-based algorithm used in FormicaBot for emergent role differentiation.

Feature vector per robot (6-D):
  [0] average_speed          — normalised 0..1
  [1] turn_rate              — normalised angular velocity
  [2] pheromone_deposit_freq — deposits per timestep
  [3] time_in_explore        — fraction of window in exploration state
  [4] time_in_exploit        — fraction of window in exploitation state
  [5] n_interactions         — neighbour interactions per timestep

Clustering runs every N_cluster = 1000 timesteps.
Scout cluster → high w_random, low w_chemo
Worker cluster → low w_random, high w_chemo
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import OPTICS

from formicabot_ros2.core.config import OPTICSConfig


class BehaviourProfiler:
    """
    Maintains a sliding-window behavioural history for one robot and
    computes the 6-D feature vector used by OPTICS.
    """

    FEATURE_NAMES = [
        "avg_speed",
        "turn_rate",
        "phero_deposit_freq",
        "time_explore",
        "time_exploit",
        "n_interactions_per_step",
    ]

    def __init__(self, window: int = 200):
        self.window = window
        # Ring buffers
        self._speeds: List[float] = []
        self._turn_rates: List[float] = []
        self._deposits: List[int] = []      # 1 = deposited this step
        self._states: List[str] = []        # 'explore' or 'exploit'
        self._interactions: List[int] = []

    def record(
        self,
        speed: float,
        turn_rate: float,
        deposited: bool,
        state: str,
        n_interactions: int,
    ):
        """Append one timestep of observation."""
        self._speeds.append(speed)
        self._turn_rates.append(abs(turn_rate))
        self._deposits.append(int(deposited))
        self._states.append(state)
        self._interactions.append(n_interactions)

        # Trim to window
        if len(self._speeds) > self.window:
            self._speeds.pop(0)
            self._turn_rates.pop(0)
            self._deposits.pop(0)
            self._states.pop(0)
            self._interactions.pop(0)

    def get_feature_vector(self) -> np.ndarray:
        """Compute 6-D feature vector from current window."""
        n = max(1, len(self._speeds))
        avg_speed = np.mean(self._speeds) if self._speeds else 0.0
        avg_turn = np.mean(self._turn_rates) if self._turn_rates else 0.0
        deposit_freq = np.mean(self._deposits) if self._deposits else 0.0
        t_explore = self._states.count("explore") / n
        t_exploit = self._states.count("exploit") / n
        n_inter = np.mean(self._interactions) if self._interactions else 0.0
        return np.array(
            [avg_speed, avg_turn, deposit_freq, t_explore, t_exploit, n_inter],
            dtype=np.float64,
        )


class OPTICSRoleDifferentiator:
    """
    Swarm-level OPTICS clustering that assigns emergent roles.

    Parameters
    ----------
    cfg : OPTICSConfig
    n_robots : int
    rng : np.random.Generator
    """

    ROLE_NOISE = "noise"       # OPTICS noise points
    ROLE_SCOUT = "scout"       # Exploratory cluster
    ROLE_WORKER = "worker"     # Exploitative cluster

    def __init__(
        self,
        cfg: OPTICSConfig = None,
        n_robots: int = 20,
        rng: np.random.Generator = None,
    ):
        self.cfg = cfg or OPTICSConfig()
        self.n_robots = n_robots
        self.rng = rng if rng is not None else np.random.default_rng()

        # Per-robot profilers
        self.profilers: List[BehaviourProfiler] = [
            BehaviourProfiler(self.cfg.feature_window) for _ in range(n_robots)
        ]

        # Current role assignments
        self.roles: List[str] = [self.ROLE_WORKER] * n_robots
        self.labels: np.ndarray = np.zeros(n_robots, dtype=int)

        # History for analysis
        self.cluster_history: List[Dict] = []
        self.timestep = 0

    # ------------------------------------------------------------------
    # Per-timestep update
    # ------------------------------------------------------------------
    def update(
        self,
        robot_id: int,
        speed: float,
        turn_rate: float,
        deposited: bool,
        state: str,
        n_interactions: int,
    ):
        """Record one timestep of observations for a single robot."""
        self.profilers[robot_id].record(speed, turn_rate, deposited, state, n_interactions)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def run_clustering(self, timestep: int = None) -> Dict[int, str]:
        """
        Execute OPTICS clustering on current feature matrix.
        Returns a dict mapping robot_id → role string.
        """
        self.timestep = timestep or self.timestep

        # Build feature matrix [n_robots, n_features]
        X = np.vstack([p.get_feature_vector() for p in self.profilers])

        # Normalise columns to [0, 1]
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range < 1e-9] = 1.0   # Avoid div-by-zero
        X_norm = (X - col_min) / col_range

        # Run OPTICS
        clust = OPTICS(
            min_samples=self.cfg.min_pts,
            xi=self.cfg.xi,
            min_cluster_size=self.cfg.min_cluster_size,
        )
        try:
            clust.fit(X_norm)
            labels = clust.labels_
        except Exception:
            # Fallback: keep previous roles
            labels = self.labels.copy()

        self.labels = labels

        # Assign semantic roles based on cluster statistics
        role_map = self._assign_semantic_roles(X_norm, labels)
        for rid in range(self.n_robots):
            lbl = labels[rid]
            self.roles[rid] = role_map.get(lbl, self.ROLE_WORKER)

        # Log
        self.cluster_history.append(
            {
                "timestep": self.timestep,
                "n_scouts": self.roles.count(self.ROLE_SCOUT),
                "n_workers": self.roles.count(self.ROLE_WORKER),
                "n_noise": self.roles.count(self.ROLE_NOISE),
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            }
        )

        return {i: self.roles[i] for i in range(self.n_robots)}

    def _assign_semantic_roles(
        self, X_norm: np.ndarray, labels: np.ndarray
    ) -> Dict[int, str]:
        """
        Map integer cluster labels to semantic roles.
        The scout cluster has higher average (avg_speed, time_explore) and
        lower time_exploit compared to the worker cluster.
        """
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        role_map: Dict[int, str] = {-1: self.ROLE_NOISE}

        if not unique_labels:
            return role_map

        # Score each cluster by exploration tendency
        # Features: [avg_speed, turn_rate, deposit_freq, t_explore, t_exploit, interactions]
        explore_feat_idx = 3   # time_explore
        exploit_feat_idx = 4   # time_exploit

        cluster_scores: Dict[int, float] = {}
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            cluster_data = X_norm[mask]
            # Scout tendency = explore - exploit
            explore_score = (
                cluster_data[:, explore_feat_idx].mean()
                - cluster_data[:, exploit_feat_idx].mean()
            )
            cluster_scores[lbl] = explore_score

        # Sort: highest explore_score → scout
        sorted_labels = sorted(cluster_scores, key=cluster_scores.get, reverse=True)
        if len(sorted_labels) >= 2:
            role_map[sorted_labels[0]] = self.ROLE_SCOUT
            for lbl in sorted_labels[1:]:
                role_map[lbl] = self.ROLE_WORKER
        elif len(sorted_labels) == 1:
            # Single cluster: split by median of explore score
            lbl = sorted_labels[0]
            score = cluster_scores[lbl]
            role_map[lbl] = self.ROLE_SCOUT if score > 0 else self.ROLE_WORKER

        return role_map

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_role(self, robot_id: int) -> str:
        return self.roles[robot_id]

    def should_recluster(self, timestep: int) -> bool:
        return timestep % self.cfg.cluster_interval == 0 and timestep > 0

    def get_cluster_history(self) -> List[Dict]:
        return self.cluster_history

    def get_feature_matrix(self) -> np.ndarray:
        return np.vstack([p.get_feature_vector() for p in self.profilers])

    def get_stats(self) -> Dict:
        return {
            "n_scouts": self.roles.count(self.ROLE_SCOUT),
            "n_workers": self.roles.count(self.ROLE_WORKER),
            "n_noise": self.roles.count(self.ROLE_NOISE),
            "scout_fraction": self.roles.count(self.ROLE_SCOUT) / max(1, self.n_robots),
        }
