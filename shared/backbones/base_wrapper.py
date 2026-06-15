"""
Component: Model Wrapper — Abstract Base Class
Location: shared/backbones/base_wrapper.py

Ref:
    - rules/STRUCTURE.md
    - rules/CONVENTIONS.md Section 4
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModelWrapper(ABC):
    """Abstract base for all method model wrappers.

    Contract:
        - forward_train(images, labels, **kwargs) → loss scalar
        - forward_inference(images) → pred tensor (B, H, W) integer class map
        - train_mode() → sets model to train, activates method-specific components
        - eval_mode() → sets model to eval, DEACTIVATES method-specific components
        - get_hook_stats() → dict of diagnostic stats (or empty dict)
        - time_inference(dummy_input) → float (ms per image, no method components)

    Backbone is ALWAYS a GeoSeg model loaded via geoseg_adapter.
    Method-specific logic is layered on top, not embedded inside.
    """

    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        self.backbone = backbone
        self.cfg = cfg
        self._is_train_mode = False

    def to(self, device: torch.device | str) -> BaseModelWrapper:
        """Move backbone and all nn.Module attributes to target device."""
        self.backbone = self.backbone.to(device)
        for name, attr in list(self.__dict__.items()):
            if isinstance(attr, nn.Module):
                setattr(self, name, attr.to(device))
        return self

    @abstractmethod
    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for training. Returns scalar loss."""
        ...

    @abstractmethod
    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference. Returns (B, H, W) argmax prediction."""
        ...

    def train_mode(self) -> None:
        """Set model to training mode. Activate method components."""
        self.backbone.train()
        self._is_train_mode = True

    def eval_mode(self) -> None:
        """Set model to eval mode. Deactivate method components.

        IMPORTANT: After calling eval_mode(), the model must behave
        IDENTICALLY to a plain backbone with no method overhead.
        This is how zero inference overhead is guaranteed.
        """
        self.backbone.eval()
        self._is_train_mode = False

    def parameters(self):
        """Return backbone parameters for optimizer."""
        return self.backbone.parameters()

    def get_hook_stats(self) -> dict[str, float]:
        """Return diagnostic stats from the current forward/backward pass.

        Returns empty dict if the method has no hooks or stats to report.
        Subclasses override this.
        """
        return {}

    def time_inference(
        self,
        dummy_input: torch.Tensor,
        n_warmup: int = 10,
        n_measure: int = 50,
    ) -> float:
        """Measure inference time per image (in milliseconds).

        This is called in EVAL mode — no hooks active.
        Used to verify zero inference overhead claim.

        Args:
            dummy_input: (1, C, H, W) tensor on the correct device.
            n_warmup: Number of warmup runs.
            n_measure: Number of timed runs.

        Returns:
            Mean inference time in milliseconds per image.
        """
        self.eval_mode()
        device = next(self.backbone.parameters()).device
        dummy_input = dummy_input.to(device)

        with torch.no_grad():
            # Warmup
            for _ in range(n_warmup):
                self.forward_inference(dummy_input)

            # Measure
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_measure):
                self.forward_inference(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000  # ms

        return elapsed / n_measure
