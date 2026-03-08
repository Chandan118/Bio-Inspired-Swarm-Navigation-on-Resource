"""
Dual-Modality Virtual Pheromone Grid
=====================================
Implements both the optical trail (TCRT5000 simulation) and chemical trail
(MQ-135 simulation) pheromone grids described in FormicaBot Chapter 4.

Pheromone evaporation follows:
    τ(t+Δt) = τ(t) · exp(−λ · Δt)

Modality arbitration is driven by the real-time optical SNR.
When SNR < snr_threshold, the chemical fallback channel is activated.
This reduces average MQ-135 power from 800 mW (always-on) to ~25 mW.
"""

import numpy as np
from typing import Tuple

from formicabot_ros2.core.config import PheromoneConfig


class PheromoneGrid:
    """
    2-D pheromone grid supporting both optical and chemical modalities.

    Parameters
    ----------
    width, height : float
        Arena dimensions in metres.
    cell_size : float
        Grid resolution in metres (default 0.10 m = 10 cm).
    cfg : PheromoneConfig
        All pheromone hyperparameters.
    """

    def __init__(
        self,
        width: float,
        height: float,
        cell_size: float = 0.10,
        cfg: PheromoneConfig = None,
    ):
        self.cfg = cfg or PheromoneConfig()
        self.cell_size = cell_size
        self.nx = int(np.ceil(width / cell_size))
        self.ny = int(np.ceil(height / cell_size))

        # Continuous pheromone concentration grids [n_x, n_y]
        self.optical = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.chemical = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Per-cell SNR estimate (moving average of recent optical readings)
        self._snr_map = np.ones((self.nx, self.ny), dtype=np.float32) * 10.0
        self._snr_alpha = 0.05  # EMA coefficient

        # Per-cell modality flag: True → optical active, False → chemical active
        self._use_optical = np.ones((self.nx, self.ny), dtype=bool)

        # Energy accounting
        self.total_energy_optical_j = 0.0    # Joules consumed
        self.total_energy_chemical_j = 0.0

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _xy_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(np.clip(x / self.cell_size, 0, self.nx - 1))
        cy = int(np.clip(y / self.cell_size, 0, self.ny - 1))
        return cx, cy

    def _cell_to_xy(self, cx: int, cy: int) -> Tuple[float, float]:
        return (cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size

    # ------------------------------------------------------------------
    # Deposition
    # ------------------------------------------------------------------
    def deposit_optical(self, x: float, y: float, amount: float = None):
        """Deposit optical (TCRT5000-style) pheromone at (x, y)."""
        cx, cy = self._xy_to_cell(x, y)
        q = amount if amount is not None else self.cfg.optical_deposit
        self.optical[cx, cy] = np.clip(
            self.optical[cx, cy] + q, 0, self.cfg.optical_max
        )

    def deposit_chemical(self, x: float, y: float, amount: float = None):
        """Deposit chemical (MQ-135-style) pheromone at (x, y)."""
        cx, cy = self._xy_to_cell(x, y)
        q = amount if amount is not None else self.cfg.chemical_deposit
        self.chemical[cx, cy] = np.clip(
            self.chemical[cx, cy] + q, 0, self.cfg.chemical_max
        )

    def deposit_reinforcement(self, x: float, y: float, quality: float = 1.0):
        """
        Reinforce trail at (x, y) proportional to target quality.
        Both modalities receive weighted reinforcement.
        """
        amt_opt = self.cfg.optical_deposit * quality
        amt_chem = self.cfg.chemical_deposit * quality
        self.deposit_optical(x, y, amt_opt)
        self.deposit_chemical(x, y, amt_chem)

    # ------------------------------------------------------------------
    # Evaporation
    # ------------------------------------------------------------------
    def evaporate(self, dt: float = None):
        """
        Apply exponential evaporation to all cells.
        τ' = τ · exp(−λ · dt)
        """
        dt = dt or self.cfg.dt
        self.optical *= np.exp(-self.cfg.optical_evap_rate * dt)
        self.chemical *= np.exp(-self.cfg.chemical_evap_rate * dt)
        # Zero out negligible concentrations to handle float precision
        self.optical[self.optical < 1e-6] = 0.0
        self.chemical[self.chemical < 1e-6] = 0.0

    # ------------------------------------------------------------------
    # Sensing / Reading
    # ------------------------------------------------------------------
    def read_optical(self, x: float, y: float, rng: np.random.Generator = None) -> float:
        """
        Read optical pheromone with simulated TCRT5000 noise.
        Returns noisy sensor reading (clipped to non-negative).
        """
        cx, cy = self._xy_to_cell(x, y)
        true_val = self.optical[cx, cy]
        noise = 0.0
        if rng is not None:
            noise = rng.normal(0, self.cfg.optical_noise_std)
        reading = max(0.0, true_val + noise)

        # Update SNR map: SNR = signal / noise_std (simplified)
        snr = true_val / self.cfg.optical_noise_std if self.cfg.optical_noise_std > 0 else 99.0
        self._snr_map[cx, cy] = (
            (1 - self._snr_alpha) * self._snr_map[cx, cy] + self._snr_alpha * snr
        )
        self._use_optical[cx, cy] = self._snr_map[cx, cy] >= self.cfg.snr_threshold

        return reading

    def read_chemical(self, x: float, y: float, rng: np.random.Generator = None) -> float:
        """
        Read chemical pheromone with simulated MQ-135 noise.
        """
        cx, cy = self._xy_to_cell(x, y)
        true_val = self.chemical[cx, cy]
        noise = 0.0
        if rng is not None:
            noise = rng.normal(0, self.cfg.chemical_noise_std)
        return max(0.0, true_val + noise)

    def read_effective(
        self, x: float, y: float, rng: np.random.Generator = None
    ) -> Tuple[float, str]:
        """
        SNR-arbitrated read: returns (concentration, active_modality).
        Primary = optical; fallback = chemical when SNR drops below threshold.
        """
        cx, cy = self._xy_to_cell(x, y)
        opt_reading = self.read_optical(x, y, rng)

        if self._use_optical[cx, cy]:
            return opt_reading, "optical"
        else:
            chem_reading = self.read_chemical(x, y, rng)
            return chem_reading, "chemical"

    def read_neighbourhood(
        self, x: float, y: float, radius_m: float = 0.30, rng: np.random.Generator = None
    ) -> Tuple[float, str]:
        """
        Average effective pheromone over a neighbourhood of radius_m.
        Models the spatial integration a real sensor would perform.
        """
        r_cells = max(1, int(radius_m / self.cell_size))
        cx, cy = self._xy_to_cell(x, y)
        vals = []
        mode_counts = {"optical": 0, "chemical": 0}
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx_, ny_ = cx + dx, cy + dy
                if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                    xi, yi = self._cell_to_xy(nx_, ny_)
                    v, mode = self.read_effective(xi, yi, rng)
                    vals.append(v)
                    mode_counts[mode] += 1
        dom_mode = "optical" if mode_counts["optical"] >= mode_counts["chemical"] else "chemical"
        return float(np.mean(vals)) if vals else 0.0, dom_mode

    # ------------------------------------------------------------------
    # Power accounting
    # ------------------------------------------------------------------
    def update_power_accounting(self, dt: float = None):
        """
        Accumulate energy usage based on active modality proportions.
        Called once per timestep.
        """
        dt = dt or self.cfg.dt
        # Proportion of cells using chemical sensing
        frac_chemical = 1.0 - self._use_optical.mean()
        frac_optical = 1.0 - frac_chemical

        # Optical sensors: always active (very low power)
        self.total_energy_optical_j += (
            self.cfg.optical_power_mw * 1e-3 * dt * frac_optical
        )
        # Chemical sensors: gated – only activated where SNR is low
        # If frac_chemical > 0, some MQ-135 heaters are on
        eff_chem_power_mw = (
            frac_chemical * self.cfg.chemical_power_mw
            + (1 - frac_chemical) * self.cfg.chemical_gated_mw
        )
        self.total_energy_chemical_j += eff_chem_power_mw * 1e-3 * dt

    # ------------------------------------------------------------------
    # Gradient helpers (for trail following)
    # ------------------------------------------------------------------
    def gradient_at(self, x: float, y: float) -> np.ndarray:
        """
        Compute the 2D pheromone gradient at (x, y) using finite differences.
        Returns a unit vector pointing in the direction of increasing pheromone.
        """
        cx, cy = self._xy_to_cell(x, y)
        # Clamp to valid range
        cx1, cx2 = max(0, cx - 1), min(self.nx - 1, cx + 1)
        cy1, cy2 = max(0, cy - 1), min(self.ny - 1, cy + 1)

        # Use whichever modality is dominant
        if self._use_optical[cx, cy]:
            grid = self.optical
        else:
            grid = self.chemical

        gx = (grid[cx2, cy] - grid[cx1, cy]) / (2 * self.cell_size + 1e-9)
        gy = (grid[cx, cy2] - grid[cx, cy1]) / (2 * self.cell_size + 1e-9)
        grad = np.array([gx, gy])
        norm = np.linalg.norm(grad)
        return grad / (norm + 1e-9)

    # ------------------------------------------------------------------
    # Statistics / reporting
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return summary statistics for logging and visualisation."""
        n_chemical_cells = int((~self._use_optical).sum())
        return {
            "optical_mean": float(self.optical.mean()),
            "optical_max": float(self.optical.max()),
            "chemical_mean": float(self.chemical.mean()),
            "chemical_max": float(self.chemical.max()),
            "cells_using_chemical": n_chemical_cells,
            "frac_chemical": n_chemical_cells / (self.nx * self.ny),
            "energy_optical_j": self.total_energy_optical_j,
            "energy_chemical_j": self.total_energy_chemical_j,
        }

    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return copies of optical and chemical grids for visualisation."""
        return self.optical.copy(), self.chemical.copy()
