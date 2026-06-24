"""
Component: GeoSeg Model Adapter
Location: shared/backbones/geoseg_adapter.py

Handles:
- Loading GeoSeg models (UNetFormer, FTUNetformer)
- Programmatic PYTHONPATH resolution for GeoSeg
- Capturing intermediate features before the segmentation head
"""
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# Resolve GeoSeg import path dynamically
# Priority:
#   1. GEOSEG_DIR environment variable (explicit override, works in subprocesses)
#   2. project_root/tmp/GeoSeg (local dev path)
#   3. project_root/geoseg (Colab/server path: GeoSeg cloned at WACV_EXP/geoseg/)
# ──────────────────────────────────────────────────────────
current_file = Path(__file__).resolve()
# shared is at project_root/shared → parents[2] = project_root
project_root = current_file.parents[2]

geoseg_dir: Optional[Path] = None

# 1. Explicit env var (highest priority — works for subprocess spawned by !python)
_env_geoseg = os.environ.get("GEOSEG_DIR")
if _env_geoseg:
    _candidate = Path(_env_geoseg)
    if _candidate.exists():
        geoseg_dir = _candidate

# 2. Local dev path: project_root/tmp/GeoSeg
if geoseg_dir is None:
    _candidate = project_root / "tmp" / "GeoSeg"
    if _candidate.exists():
        geoseg_dir = _candidate

# 3. Colab/server fallback: project_root/geoseg (GeoSeg repo cloned here)
if geoseg_dir is None:
    _candidate = project_root / "geoseg"
    if _candidate.exists():
        geoseg_dir = _candidate

if geoseg_dir is not None and str(geoseg_dir) not in sys.path:
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

    elif model_name == "unetformer_r50":
        model = UNetFormer(backbone_name="resnet50.a1_in1k", num_classes=num_classes, pretrained=pretrained)
        return model

    elif model_name == "unetformer_convnexts":
        from geoseg.models.UNetFormer import Decoder
        import timm
        # UNetFormer defaults to resnet18. Create with dummy backbone, then swap.
        model = UNetFormer(num_classes=num_classes, pretrained=False)
        model.backbone = timm.create_model(
            "convnext_small.fb_in22k_ft_in1k",
            features_only=True, output_stride=32,
            out_indices=(0, 1, 2, 3), pretrained=pretrained,
        )
        encoder_channels = model.backbone.feature_info.channels()
        model.decoder = Decoder(encoder_channels, 64, 0.1, 8, num_classes)
        return model

    elif model_name == "ftunetformer_swint":
        try:
            from geoseg.models.FTUNetFormer import FTUNetFormer
        except ImportError as e:
            raise ImportError(f"Failed to import FTUNetFormer. Error: {e}")
        model = FTUNetFormer(num_classes=num_classes, freeze_stages=-1)
        return model

    elif model_name == "ftunetformer_swinb":
        try:
            from geoseg.models.FTUNetFormer import ft_unetformer
        except ImportError as e:
            raise ImportError(
                f"Failed to import FTUNetformer from geoseg. "
                f"Ensure tmp/GeoSeg is present and sys.path is correct.\nError: {e}"
            )
        model = ft_unetformer(pretrained=pretrained, num_classes=num_classes, weight_path=None)
        return model

    elif model_name == "segformer_b2":
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError:
            raise ImportError("transformers library required for SegFormer. pip install transformers")
        segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        return _SegFormerWrapper(segformer)

    else:
        raise ValueError(
            f"Unsupported backbone model: {model_name}. "
            f"Supported options: 'unetformer_r18', 'unetformer_r50', 'unetformer_convnexts', "
            f"'ftunetformer_swint', 'ftunetformer_swinb', 'segformer_b2'."
        )


class _SegFormerWrapper(nn.Module):
    """Wrapper making SegFormer compatible with GeoSeg backbone + CAS."""
    def __init__(self, segformer_model):
        super().__init__()
        self.model = segformer_model
        # Replace Linear classifier with Conv2d for CAS _get_W() compatibility
        old_linear = segformer_model.decode_head.classifier
        new_conv = nn.Conv2d(old_linear.in_features, old_linear.out_features, 1)
        new_conv.weight.data = old_linear.weight.data.unsqueeze(-1).unsqueeze(-1)
        new_conv.bias.data = old_linear.bias.data
        self.model.decode_head.classifier = new_conv

    def forward(self, x):
        out = self.model(pixel_values=x)
        logits = out.logits  # (B, K, H/4, W/4) — SegFormer stride-4
        return (logits,)

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)


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
