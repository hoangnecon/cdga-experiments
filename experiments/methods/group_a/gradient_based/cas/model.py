"""CAS: Conflict-Aware Scaling. WACV 2027."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbones.base_wrapper import BaseModelWrapper


class CASModel(BaseModelWrapper):

    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.gamma = cfg["cas"].get("gamma", 1.0)
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def train_mode(self) -> None: self.backbone.train()
    def eval_mode(self) -> None: self.backbone.eval()

    def _get_W(self) -> torch.Tensor:
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1].weight.squeeze(-1).squeeze(-1)

    def forward_train(self, images, labels, boundary_mask=None, **kwargs):
        out = self.backbone(images)
        logits = out[0] if isinstance(out, tuple) else out
        if boundary_mask is None:
            boundary_mask = torch.zeros(labels.shape, device=labels.device)

        W = self._get_W()
        B, _, H, W_s = logits.shape
        gamma = self.gamma

        # Resize labels & mask to logit resolution
        k = labels
        if k.shape[-2:] != (H, W_s):
            k = F.interpolate(k.float().unsqueeze(1), size=(H, W_s), mode='nearest').squeeze(1).long()
        S_dist = boundary_mask
        if S_dist.shape[-2:] != (H, W_s):
            S_dist = F.interpolate(S_dist.float(), size=(H, W_s), mode='nearest')
        if S_dist.dim() == 3:
            S_dist = S_dist.unsqueeze(1)

        def cas_hook(g):
            with torch.no_grad():
                g_F = torch.einsum('kc,bkhw->bchw', W, g)

                g_pad = F.pad(g_F, (1, 1, 1, 1), mode='replicate')
                k_pad = F.pad(k.float(), (1, 1, 1, 1), mode='replicate')

                min_cos = torch.ones(B, H, W_s, device=g.device)
                has_neighbor = torch.zeros(B, H, W_s, dtype=torch.bool, device=g.device)

                for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    gn = g_pad[:, :, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                    kn = k_pad[:, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                    diff = (kn != k.float())
                    cos = F.cosine_similarity(g_F, gn, dim=1, eps=1e-8)
                    min_cos = torch.where(diff, torch.minimum(min_cos, cos), min_cos)
                    has_neighbor |= diff

                S_conflict = torch.where(has_neighbor,
                                         (1.0 - min_cos).clamp(0.0, 2.0),
                                         torch.zeros_like(min_cos))
                S_conflict = S_conflict.unsqueeze(1)  # (B, 1, H, W)

                s = 1.0 + gamma * S_dist * S_conflict  # (B, 1, H, W)
                return g * s  # direction preserved

        logits.register_hook(cas_hook)
        return self.ce_fn(logits, labels)

    def forward_inference(self, images):
        with torch.no_grad():
            out = self.backbone(images)
            return (out[0] if isinstance(out, tuple) else out).argmax(dim=1)


def build_model(cfg: dict) -> BaseModelWrapper:
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(cfg["model"]["backbone"], cfg["data"]["num_classes"], cfg["model"].get("pretrained", True))
    return CASModel(backbone=backbone, cfg=cfg)
