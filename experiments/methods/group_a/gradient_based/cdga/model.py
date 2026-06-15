"""
Method: CDGA (Class-Directed Gradient Amplification)
Component: Model Wrapper
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
from typing import Optional
from shared.backbones.base_wrapper import BaseModelWrapper
from shared.utils.cdga_utils import CDGAHook
from .loss import CDGALoss

class CDGAModel(BaseModelWrapper):
    """CDGA Model Wrapper."""
    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.loss_fn = CDGALoss(ignore_index=ignore_index)
        
        cdga_cfg = cfg["cdga"]
        self.gamma = cdga_cfg.get("gamma", 10.0)
        self.mode = cdga_cfg.get("mode", "class_directed")
        self._hook = CDGAHook(gamma=self.gamma, mode=self.mode)
        self._hook_handle = None

    def train_mode(self) -> None:
        super().train_mode()

    def eval_mode(self) -> None:
        super().eval_mode()
        self._detach_hook()

    def _detach_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook.detach()
            self._hook_handle = None

    def _get_last_conv_weight(self, module: nn.Module) -> Optional[torch.Tensor]:
        if isinstance(module, nn.Conv2d):
            return module.weight
        for child in reversed(list(module.children())):
            w = self._get_last_conv_weight(child)
            if w is not None:
                return w
        return None

    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert self._is_train_mode, "forward_train() called in eval mode. Call train_mode() first."
        
        self._detach_hook()
        
        from shared.backbones.geoseg_adapter import get_feature_and_logits
        feature_map, logits = get_feature_and_logits(self.backbone, images)
        
        if boundary_mask is not None and self.cfg["cdga"].get("enabled", True):
            feature_map.retain_grad()
            # Extract classification head weights safely from the final layer
            last_layer = self.backbone.decoder.segmentation_head[-1]
            head_weights = self._get_last_conv_weight(last_layer)
            if head_weights is None:
                raise RuntimeError("Could not find nn.Conv2d in the segmentation_head to extract weights.")
            
            self._hook.set_inputs(S_mask=boundary_mask, labels=labels, head_weights=head_weights)
            self._hook_handle = self._hook.attach(feature_map)
            
        loss = self.loss_fn(logits, labels)
        return loss

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        assert not self._is_train_mode, "forward_inference() called in train mode. Call eval_mode() first."
        with torch.no_grad():
            out = self.backbone(images)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            return logits.argmax(dim=1)

    def get_hook_stats(self) -> dict[str, float]:
        return dict(self._hook.grad_stats)

def build_model(cfg: dict) -> BaseModelWrapper:
    """Factory function for CDGAModel."""
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return CDGAModel(backbone=backbone, cfg=cfg)
