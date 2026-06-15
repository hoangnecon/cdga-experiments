"""
Method: Boundary F1 Loss (Ma2024)
Component: Model Wrapper
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
from shared.backbones.base_wrapper import BaseModelWrapper
from .loss import BoundaryLoss

class Ma2024Model(BaseModelWrapper):
    """Ma2024 soft BF1 Loss Model Wrapper."""
    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        dilation_width = cfg["ma2024"].get("dilation_width", 3)
        self.bdy_weight = cfg["ma2024"].get("weight", 1.0)
        
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bdy_fn = BoundaryLoss(theta0=dilation_width, theta=5)


    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        out = self.backbone(images)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
            
        ce_loss = self.ce_fn(logits, labels)
        bdy_loss = self.bdy_fn(logits, labels)
        
        return ce_loss + self.bdy_weight * bdy_loss

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(images)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            return logits.argmax(dim=1)

def build_model(cfg: dict) -> BaseModelWrapper:
    """Factory function for Ma2024Model."""
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return Ma2024Model(backbone=backbone, cfg=cfg)
