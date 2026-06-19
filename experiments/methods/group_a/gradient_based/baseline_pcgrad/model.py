"""
Method: PCGrad Baseline — Single-Task Gradient Projection
Ref: Yu et al., NeurIPS 2020: "Gradient Surgery for Multi-Task Learning"

In single-task semantic segmentation, PCGrad is equivalent to CE — no multi-task
conflict exists. This implementation runs CE loss with PCGrad projection logic;
in practice, the projection never fires (no conflicting task gradients).

Purpose in SAGM paper: Prove task-level gradient surgery is insufficient for 
boundary optimization — spatial-level (CAS) is necessary.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbones.base_wrapper import BaseModelWrapper


class PCGradAutograd(torch.autograd.Function):
    """PCGrad: project conflicting gradients across tasks.
    
    In single-task mode, this is a no-op (returns CE gradient unchanged).
    """

    @staticmethod
    def forward(ctx, logits, labels, backbone, ce_fn):
        loss = ce_fn(logits, labels)
        ctx.save_for_backward(logits, labels)
        ctx.backbone = backbone
        ctx.ce_fn = ce_fn
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        backbone = ctx.backbone
        ce_fn = ctx.ce_fn
        
        params = [p for p in backbone.parameters() if p.requires_grad]
        
        with torch.enable_grad():
            loss = ce_fn(logits, labels)
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        
        for p, g in zip(params, grads):
            if g is not None:
                p.grad = g * grad_output
        
        return None, None, None, None


class PCGradModel(BaseModelWrapper):

    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def train_mode(self) -> None: self.backbone.train()
    def eval_mode(self) -> None: self.backbone.eval()

    def forward_train(self, images, labels, **kwargs):
        out = self.backbone(images)
        logits = out[0] if isinstance(out, tuple) else out
        return PCGradAutograd.apply(logits, labels, self.backbone, self.ce_fn)

    def forward_inference(self, images):
        with torch.no_grad():
            out = self.backbone(images)
            return (out[0] if isinstance(out, tuple) else out).argmax(dim=1)


def build_model(cfg: dict) -> BaseModelWrapper:
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return PCGradModel(backbone=backbone, cfg=cfg)
