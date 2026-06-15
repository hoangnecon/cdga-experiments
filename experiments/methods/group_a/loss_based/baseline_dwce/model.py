"""
Method: Distance-Weighted CE (DW-CE)
Component: Model Wrapper
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
from shared.backbones.base_wrapper import BaseModelWrapper
from .loss import DWCELoss

class DWCETemplateModel(BaseModelWrapper):
    """DW-CE Model Wrapper."""
    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        gamma = cfg["dwce"].get("gamma", 10.0)
        self.loss_fn = DWCELoss(gamma=gamma, ignore_index=ignore_index)

    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        boundary_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self.backbone(images)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        
        if boundary_mask is None:
            boundary_mask = torch.zeros_like(labels).unsqueeze(1).float()
            
        return self.loss_fn(logits, labels, boundary_mask)

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(images)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            return logits.argmax(dim=1)

def build_model(cfg: dict) -> BaseModelWrapper:
    """Factory function for DWCETemplateModel."""
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return DWCETemplateModel(backbone=backbone, cfg=cfg)
