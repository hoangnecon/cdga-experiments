"""
Method: PCGrad Baseline
Component: Model Wrapper
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
from shared.backbones.base_wrapper import BaseModelWrapper
from .loss import BoundaryCELoss, PCGradAutograd

class PCGradTemplateModel(BaseModelWrapper):
    """PCGrad Model Wrapper."""
    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bdy_fn = BoundaryCELoss(ignore_index=ignore_index)

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
            
        return PCGradAutograd.apply(logits, labels, boundary_mask, self.backbone, self.ce_fn, self.bdy_fn)

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(images)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            return logits.argmax(dim=1)

def build_model(cfg: dict) -> BaseModelWrapper:
    """Factory function for PCGradTemplateModel."""
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return PCGradTemplateModel(backbone=backbone, cfg=cfg)
