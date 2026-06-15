"""
Method: Active Boundary Loss (ABL)
Component: Model Wrapper
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
from shared.backbones.base_wrapper import BaseModelWrapper
from .loss import ABLLoss

class ABLModel(BaseModelWrapper):
    """ABL Model Wrapper — faithful to Wang et al., AAAI 2022."""
    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        max_N_ratio = cfg["abl"].get("max_N_ratio", 0.01)
        max_clip_dist = cfg["abl"].get("max_clip_dist", 20.0)
        is_detach = cfg["abl"].get("is_detach", True)
        self.abl_weight = cfg["abl"].get("weight", 1.0)
        
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.abl_fn = ABLLoss(
            max_N_ratio=max_N_ratio,
            ignore_index=ignore_index,
            max_clip_dist=max_clip_dist,
            is_detach=is_detach,
        )
        
        self.step_counter = 0
        self.total_epochs = cfg["train"].get("epochs", 100)
        self.start_epoch = int(0.8 * self.total_epochs)
        
        # Estimate steps per epoch based on dataset
        dataset = cfg["data"]["dataset"].lower()
        batch_size = cfg["train"].get("batch_size", 8)
        if "vaihingen" in dataset:
            num_samples = 1487
        elif "potsdam" in dataset:
            num_samples = 10947
        elif "loveda" in dataset:
            num_samples = 1156
        else:
            num_samples = 1000  # Fallback
            
        self.steps_per_epoch = max(1, num_samples // batch_size)

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
        
        # Calculate current epoch
        self.step_counter += 1
        current_epoch = self.step_counter / self.steps_per_epoch
        
        if current_epoch >= self.start_epoch:
            abl_loss = self.abl_fn(logits, labels)
            loss = ce_loss + self.abl_weight * abl_loss
        else:
            loss = ce_loss
            
        return loss

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(images)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            return logits.argmax(dim=1)

def build_model(cfg: dict) -> BaseModelWrapper:
    """Factory function for ABLModel."""
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return ABLModel(backbone=backbone, cfg=cfg)
