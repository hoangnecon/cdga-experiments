"""SAGS: Spatially-Aware Gradient Surgery. WACV 2027."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbones.base_wrapper import BaseModelWrapper


class SAGSModel(BaseModelWrapper):

    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.gamma = cfg["sags"].get("gamma", 1.0)
        self.cosine_threshold = cfg["sags"].get("cosine_threshold", 0.0)
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def train_mode(self) -> None: self.backbone.train()
    def eval_mode(self) -> None: self.backbone.eval()

    def _get_W(self) -> torch.Tensor:
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1].weight.squeeze(-1).squeeze(-1)  # (K, C)

    def forward_train(self, images, labels, boundary_mask=None, **kwargs):
        out = self.backbone(images)
        logits = out[0] if isinstance(out, tuple) else out
        return self.ce_fn(logits, labels)

    def forward_inference(self, images):
        with torch.no_grad():
            out = self.backbone(images)
            return (out[0] if isinstance(out, tuple) else out).argmax(dim=1)


def build_model(cfg: dict) -> BaseModelWrapper:
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(cfg["model"]["backbone"], cfg["data"]["num_classes"], cfg["model"].get("pretrained", True))
    return SAGSModel(backbone=backbone, cfg=cfg)
