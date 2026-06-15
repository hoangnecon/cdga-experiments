"""
Component: GeoSeg Model Adapter
Location: shared/backbones/geoseg_adapter.py

Handles:
- Loading GeoSeg models (UNetFormer, FTUNetformer)
- Programmatic PYTHONPATH resolution for GeoSeg
- Capturing intermediate features before the segmentation head
"""
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# Resolve GeoSeg import path dynamically
# ──────────────────────────────────────────────────────────
current_file = Path(__file__).resolve()
# shared is at project_root/shared
project_root = current_file.parents[2]
geoseg_dir = project_root / "tmp" / "GeoSeg"

if geoseg_dir.exists() and str(geoseg_dir) not in sys.path:
    sys.path.insert(0, str(geoseg_dir))


def load_geoseg_backbone(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Load a backbone model from the GeoSeg package.

    Args:
        model_name: 'unetformer_r18' or 'ftunetformer_swinb'.
        num_classes: Number of semantic classes.
        pretrained: Whether to load pre-trained weights.

    Returns:
        nn.Module: The loaded GeoSeg model.
    """
    model_name = model_name.lower().replace("-", "_")

    if model_name == "unetformer_r18":
        try:
            from geoseg.models.UNetFormer import UNetFormer
        except ImportError as e:
            raise ImportError(
                f"Failed to import UNetFormer from geoseg. "
                f"Ensure tmp/GeoSeg is present and sys.path is correct.\nError: {e}"
            )
        # unetformer_r18 default backbone in GeoSeg is swsl_resnet18
        model = UNetFormer(num_classes=num_classes, pretrained=pretrained)
        return model

    elif model_name == "ftunetformer_swinb":
        try:
            from geoseg.models.FTUNetFormer import ft_unetformer
        except ImportError as e:
            raise ImportError(
                f"Failed to import FTUNetformer from geoseg. "
                f"Ensure tmp/GeoSeg is present and sys.path is correct.\nError: {e}"
            )
        # ft_unetformer helper function automatically sets Swin-B depths,heads,etc.
        # pretrained weights are loaded from pretrain_weights/stseg_base.pth by default in GeoSeg,
        # we can disable weight_path loading if it's not present or handled externally
        model = ft_unetformer(pretrained=pretrained, num_classes=num_classes, weight_path=None)
        return model

    else:
        raise ValueError(
            f"Unsupported backbone model: {model_name}. "
            f"Supported options: 'unetformer_r18', 'ftunetformer_swinb'."
        )


def get_feature_and_logits(
    model: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute forward pass and capture the feature map right before the segmentation head.

    Uses a temporary forward hook on model.decoder.segmentation_head to extract the input.

    Args:
        model: GeoSeg backbone model (UNetFormer or FTUNetformer).
        x: (B, C, H, W) input image tensor.

    Returns:
        Tuple of (feature_map, logits)
    """
    feature_map = None

    def hook_fn(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output_tensor: torch.Tensor) -> None:
        nonlocal feature_map
        # The input to the final conv layer of segmentation_head is the feature map we want
        feature_map = input_tensors[0]

    # Register temporary hook on the final Conv layer of segmentation_head
    if hasattr(model, "decoder") and hasattr(model.decoder, "segmentation_head"):
        handle = model.decoder.segmentation_head[-1].register_forward_hook(hook_fn)
    else:
        raise AttributeError(
            "Model does not have model.decoder.segmentation_head. "
            "Please check if this is a supported UNetFormer/FTUNetformer architecture."
        )

    try:
        out = model(x)
    finally:
        # Always remove hook to prevent memory leaks
        handle.remove()

    if feature_map is None:
        raise RuntimeError("Failed to capture feature map. Forward pass did not trigger hook.")

    # In training mode, GeoSeg models may return a tuple (logits, aux_logits)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

    return feature_map, logits
