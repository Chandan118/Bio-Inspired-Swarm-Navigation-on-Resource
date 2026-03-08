"""
FormicaBot Configuration
========================
All simulation hyperparameters, hardware constants, and algorithm settings
derived from Chapter 4 of the thesis and FormicaBot journal paper (Sensors MDPI).
"""

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Environment parameters
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    width: float = 10.0          # Arena width  [m]
    height: float = 10.0         # Arena height [m]
    cell_size: float = 0.10      # Occupancy grid cell size [m]
    nest_pos: Tuple = (0.5, 5.0) # Nest location [m]
    n_targets: int = 5           # Number of foraging targets
    n_obstacles: int = 15        # Static obstacle count
    obstacle_radius: float = 0.25


# ---------------------------------------------------------------------------
# Swarm / Robot parameters
# ---------------------------------------------------------------------------
@dataclass
class SwarmConfig:
    n_robots: int = 20           # Swarm size
    robot_radius: float = 0.12   # Physical robot radius [m]
    max_speed: float = 0.30      # m/s (nominal)
    fast_speed: float = 0.45     # m/s (open-area transit)
    wheel_radius: float = 0.033  # [m] matches paper
    wheelbase: float = 0.20      # L [m] matches paper
    interaction_radius: float = 0.50  # r_interact [m]
    sensor_range: float = 1.20   # LiDAR max range [m] (sim scale)
    comm_range: float = 3.00     # Wi-Fi mesh range [m]
    battery_wh: float = 55.08    # [Wh]
    target_power_w: float = 1.20 # Target average power [W]


# ---------------------------------------------------------------------------
# Pheromone parameters
# ---------------------------------------------------------------------------
@dataclass
class PheromoneConfig:
    # Optical trail (TCRT5000 simulation)
    optical_deposit: float = 1.0    # Pheromone units deposited per step
    optical_evap_rate: float = 0.005 # λ optical [1/s] → τ(t+1) = τ(t)*exp(-λΔt)
    optical_max: float = 10.0       # Saturation cap
    optical_noise_std: float = 0.05 # Sensor noise std dev

    # Chemical trail (MQ-135 simulation)
    chemical_deposit: float = 0.80
    chemical_evap_rate: float = 0.003  # Diffuses/evaporates slower
    chemical_max: float = 10.0
    chemical_noise_std: float = 0.08

    # SNR threshold for modality switching
    snr_threshold: float = 3.0      # Below → activate chemical fallback
    optical_power_mw: float = 5.0   # Active power [mW]
    chemical_power_mw: float = 150.0  # Always-on power [mW]
    chemical_gated_mw: float = 25.0   # SNR-gated average [mW]
    dt: float = 0.10                # Simulation timestep [s]


# ---------------------------------------------------------------------------
# ACO Foraging parameters
# ---------------------------------------------------------------------------
@dataclass
class ACOConfig:
    alpha: float = 1.0    # Pheromone weight in path selection
    beta: float = 2.0     # Heuristic (inverse distance) weight
    rho: float = 0.10     # Pheromone decay (ACO specific)
    q0: float = 0.80      # Exploitation probability (ACS variant)
    w_chemo: float = 0.70  # Chemical pheromone weight in movement
    w_random: float = 0.15 # Random exploration weight
    w_wall: float = 0.15   # Wall-following / obstacle avoidance weight
    scout_w_random: float = 0.35   # Scout role: high exploration
    scout_w_chemo: float = 0.50
    worker_w_random: float = 0.10  # Worker role: high exploitation
    worker_w_chemo: float = 0.80
    tau_min: float = 0.01  # Min pheromone (prevents stagnation)
    tau_max: float = 10.0


# ---------------------------------------------------------------------------
# OPTICS Clustering parameters
# ---------------------------------------------------------------------------
@dataclass
class OPTICSConfig:
    min_pts: int = 3              # MinPts for core distance
    xi: float = 0.05             # Cluster extraction parameter
    min_cluster_size: int = 3
    n_features: int = 6           # Feature vector dimensionality
    cluster_interval: int = 100   # Re-cluster every N timesteps (10 seconds)
    feature_window: int = 200     # Timesteps of history per feature
    # Feature indices:  0=avg_speed, 1=turn_rate, 2=phero_deposit_freq,
    #                   3=time_explore, 4=time_exploit, 5=n_interactions


# ---------------------------------------------------------------------------
# MobileNetV3 / CNN parameters
# ---------------------------------------------------------------------------
@dataclass
class CNNConfig:
    input_h: int = 224
    input_w: int = 224
    n_classes: int = 3          # e.g. victim / crop_anomaly / background
    confidence_threshold: float = 0.85  # τ_conf
    fps_nominal: float = 15.0
    fps_reduced: float = 10.0
    latency_threshold_ms: float = 500.0
    power_active_w: float = 1.80  # During inference
    power_idle_w: float = 0.20
    quantization_bits: int = 8    # INT8 quantisation
    quantization_speedup: float = 3.2
    accuracy_drop_pct: float = 1.8  # <2% from paper


# ---------------------------------------------------------------------------
# Localisation / SLAM parameters
# ---------------------------------------------------------------------------
@dataclass
class SLAMConfig:
    # EKF
    ekf_process_noise_pos: float = 0.001  # [m^2]
    ekf_process_noise_yaw: float = 0.0001  # [rad^2]
    ekf_obs_noise_lidar: float = 0.01
    ekf_obs_noise_imu: float = 0.005
    uncertainty_threshold: float = 0.30   # [m] std-dev to trigger exploration boost

    # Occupancy grid
    grid_resolution: float = 0.10         # [m] 10 cm cells
    p_occ_prior: float = 0.50
    p_occ_hit: float = 0.90
    p_occ_miss: float = 0.10
    log_odds_max: float = 5.0
    log_odds_min: float = -5.0

    # ICP scan matching
    icp_max_iterations: int = 20
    icp_tolerance: float = 0.001   # Convergence threshold [m]
    icp_max_correspondence: float = 0.50  # [m]
    n_lidar_beams: int = 360       # Angular resolution


# ---------------------------------------------------------------------------
# Power Management parameters
# ---------------------------------------------------------------------------
@dataclass
class PowerConfig:
    # Subsystem power [W]
    cpu_idle_w: float = 0.30
    cpu_nominal_w: float = 0.80
    cpu_max_w: float = 1.50
    gpu_idle_w: float = 0.10
    gpu_active_w: float = 0.50
    motor_nominal_w: float = 0.80    # Both motors combined @ 0.3 m/s
    motor_fast_w: float = 1.20
    imu_w: float = 0.015
    optical_sensor_w: float = 0.005
    gas_sensor_always_on_w: float = 0.800
    gas_sensor_gated_w: float = 0.025
    kinect_active_w: float = 2.80
    kinect_idle_w: float = 0.30
    lidar_active_w: float = 1.20
    lidar_standby_w: float = 0.20
    wifi_w: float = 0.15

    # Thresholds
    battery_safety_min_runtime_min: float = 30.0  # Return-to-nest threshold
    dvfs_high_threshold: float = 0.80    # CPU utilisation to trigger scale-up
    dvfs_low_threshold: float = 0.30


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    pheromone: PheromoneConfig = field(default_factory=PheromoneConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    optics: OPTICSConfig = field(default_factory=OPTICSConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    power: PowerConfig = field(default_factory=PowerConfig)

    # Simulation
    total_timesteps: int = 6000   # 10 minutes @ 0.1s step
    random_seed: int = 42
    verbose: bool = False


# Default global config instance
DEFAULT_CONFIG = Config()
