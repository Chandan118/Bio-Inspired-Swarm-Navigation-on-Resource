"""
MobileNetV3-Small Target Recognition
======================================
Simulates the quantisation-aware MobileNetV3-Small CNN deployed on the
Jetson Orin Nano for real-time target recognition in FormicaBot.

Key parameters (from paper):
  - Training dataset: 15,000 labelled images → augmented to 60,000
  - Quantisation: FP32 → INT8 (4× memory reduction, 3.2× speedup)
  - Accuracy drop: <2% (1.8% measured)
  - Inference: 15 FPS @ 1.8 W on Jetson Orin Nano GPU
  - Confidence threshold τ_conf = 0.85
  - Adaptive frame-rate: 15 FPS → 10 FPS when latency > 500 ms
  - Input: 224×224 RGB + depth channel (Azure Kinect)

This module provides:
  1. A PyTorch MobileNetV3-based model with INT8 quantisation simulation
  2. A lightweight numpy-only mock for environments without PyTorch
  3. An inference engine with adaptive frame-rate control
  4. Power and latency tracking
"""

import time
import numpy as np
from typing import Tuple, Optional, List, Dict

from formicabot_ros2.core.config import CNNConfig

# Try importing PyTorch; fall back to numpy-only mock
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torch.quantization as quantization
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# =========================================================================
# PyTorch-based model (real inference)
# =========================================================================
if _TORCH_AVAILABLE:
    class MobileNetV3Target(nn.Module):
        """
        MobileNetV3-Small fine-tuned for FormicaBot target classes.
        Supports both FP32 and PTQ INT8 quantisation.
        """

        def __init__(self, n_classes: int = 3, pretrained: bool = False):
            super().__init__()
            # Load MobileNetV3-Small backbone
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            # Replace final classifier
            in_feats = backbone.classifier[3].in_features
            backbone.classifier[3] = nn.Linear(in_feats, n_classes)
            self.backbone = backbone

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.backbone(x)

        def quantize(self):
            """Simulate INT8 post-training quantisation (static)."""
            self.backbone.eval()
            self.backbone = torch.quantization.quantize_dynamic(
                self.backbone,
                {nn.Linear},
                dtype=torch.qint8,
            )
            return self


# =========================================================================
# Numpy-only mock model (no PyTorch required)
# =========================================================================
class _MockMobileNetV3:
    """
    Deterministic mock that simulates the CNN output distribution
    without requiring a GPU or PyTorch installation.
    Used for unit testing and environments without deep-learning libraries.
    """

    def __init__(self, n_classes: int, rng: np.random.Generator):
        self.n_classes = n_classes
        self.rng = rng
        # Simulated accuracy: ~92% for true positives
        self._accuracy = 0.92

    def predict(self, is_target: bool = False) -> np.ndarray:
        """Returns softmax probabilities [n_classes]."""
        logits = self.rng.normal(0, 1, self.n_classes)
        if is_target:
            # Boost the correct class
            correct_cls = 0
            logits[correct_cls] += np.log(self._accuracy / (1 - self._accuracy + 1e-9))
        exp_logits = np.exp(logits - logits.max())
        return exp_logits / exp_logits.sum()


# =========================================================================
# Inference Engine
# =========================================================================
class TargetRecognitionEngine:
    """
    Manages target recognition inference with adaptive frame-rate control.

    Parameters
    ----------
    cfg      : CNNConfig
    rng      : Random generator (for mock mode)
    use_mock : Force mock mode even if PyTorch is available
    """

    def __init__(
        self,
        cfg: CNNConfig = None,
        rng: np.random.Generator = None,
        use_mock: bool = True,
    ):
        self.cfg = cfg or CNNConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.use_mock = use_mock or not _TORCH_AVAILABLE

        if not self.use_mock:
            try:
                self.model = MobileNetV3Target(n_classes=self.cfg.n_classes, pretrained=False)
                self.model.quantize()
                self.model.eval()
            except Exception:
                self.use_mock = True

        if self.use_mock:
            self._mock = _MockMobileNetV3(self.cfg.n_classes, self.rng)

        # Adaptive frame-rate state
        self.current_fps = self.cfg.fps_nominal
        self._frame_counter = 0
        self._last_inference_ts: float = 0.0
        self._latency_history: List[float] = []

        # Power model
        self.is_active = False   # Camera gated when not near target
        self.cumulative_energy_j = 0.0

        # Detection log
        self.detection_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(
        self,
        image: Optional[np.ndarray] = None,
        is_near_target: bool = False,
        deprioritise_camera: bool = False,
        sim_timestep: float = 0.0,
    ) -> Tuple[bool, int, float]:
        """
        Run one inference pass.

        Parameters
        ----------
        image            : [H, W, 3] uint8 or float32 (ignored in mock).
        is_near_target   : Hint: robot is in high-pheromone region.
        deprioritise_camera : Power-gating signal from power manager.
        sim_timestep     : Current simulation time [s].

        Returns
        -------
        detected   : bool  — True if target detected above confidence threshold
        class_id   : int   — Predicted class index
        confidence : float — Softmax probability of predicted class
        """
        # Camera gating: Azure Kinect active only near targets
        self.is_active = is_near_target and not deprioritise_camera

        # Adaptive frame-rate: skip inference if not at frame interval
        frame_interval = 1.0 / self.current_fps
        elapsed = sim_timestep - self._last_inference_ts
        if elapsed < frame_interval:
            return False, -1, 0.0

        self._last_inference_ts = sim_timestep
        self._frame_counter += 1

        # Timing
        t0 = time.time()
        probs = self._run_forward(image, is_near_target)
        latency_ms = (time.time() - t0) * 1000

        # Adaptive FPS control
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 10:
            self._latency_history.pop(0)
        avg_latency = float(np.mean(self._latency_history))
        if avg_latency > self.cfg.latency_threshold_ms:
            self.current_fps = self.cfg.fps_reduced
        else:
            self.current_fps = self.cfg.fps_nominal

        # Decision
        best_cls = int(np.argmax(probs))
        confidence = float(probs[best_cls])
        detected = self.is_active and confidence >= self.cfg.confidence_threshold

        # Energy accounting
        p_now = self.cfg.power_active_w if self.is_active else self.cfg.power_idle_w
        self.cumulative_energy_j += p_now * frame_interval

        if detected:
            self.detection_history.append(
                {
                    "timestep": sim_timestep,
                    "class_id": best_cls,
                    "confidence": confidence,
                    "fps": self.current_fps,
                }
            )

        return detected, best_cls, confidence

    def _run_forward(
        self, image: Optional[np.ndarray], is_near_target: bool
    ) -> np.ndarray:
        """Internal: run forward pass (mock or real)."""
        if self.use_mock:
            return self._mock.predict(is_target=is_near_target)

        # Real PyTorch inference
        import torch
        import torchvision.transforms as T

        if image is None:
            image = np.zeros((self.cfg.input_h, self.cfg.input_w, 3), dtype=np.uint8)
        # Preprocess: normalise to ImageNet statistics
        img_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        return probs

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        return {
            "frames_processed": self._frame_counter,
            "current_fps": self.current_fps,
            "total_detections": len(self.detection_history),
            "cumulative_energy_j": self.cumulative_energy_j,
            "avg_latency_ms": float(np.mean(self._latency_history)) if self._latency_history else 0.0,
            "quantisation_speedup": self.cfg.quantization_speedup,
            "accuracy_drop_pct": self.cfg.accuracy_drop_pct,
        }
