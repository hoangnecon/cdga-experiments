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
        return convs[-1].weight.squeeze(-1).squeeze(-1)

    def forward_train(self, images, labels, boundary_mask=None, **kwargs):
        out = self.backbone(images)
        logits = out[0] if isinstance(out, tuple) else out
        if boundary_mask is None:
            boundary_mask = torch.zeros(labels.shape, device=labels.device)

        W = self._get_W()
        K, C = W.shape
        B, _, H, W_s = logits.shape

        k = labels
        if k.shape[-2:] != (H, W_s):
            k = F.interpolate(k.float().unsqueeze(1), size=(H, W_s), mode='nearest').squeeze(1).long()
        m = boundary_mask
        if m.shape[-2:] != (H, W_s):
            m = F.interpolate(m.float(), size=(H, W_s), mode='nearest')
        if m.dim() == 3:
            m = m.unsqueeze(1)

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            kc = k.clamp(0, K - 1)

            G_Z = probs.clone()
            G_Z.scatter_(1, kc.unsqueeze(1), probs.gather(1, kc.unsqueeze(1)) - 1.0)
            G_F = torch.einsum('kc,bkhw->bchw', W, G_Z)

            pm = probs.clone()
            pm.scatter_(1, kc.unsqueeze(1), 0.0)
            j = pm.argmax(dim=1)

            w_j = W[j.reshape(-1)].view(B, H, W_s, C).permute(0, 3, 1, 2)
            n2 = (w_j * w_j).sum(dim=1, keepdim=True).clamp(min=1e-8)
            alpha = ((G_F * w_j).sum(dim=1, keepdim=True) / n2).squeeze(1)

            G_pad = F.pad(G_F, (1, 1, 1, 1), mode='replicate')
            lp = F.pad(kc.float(), (1, 1, 1, 1), mode='replicate')
            Ic = torch.zeros(B, H, W_s, dtype=torch.bool, device=logits.device)
            for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                Gn = G_pad[:, :, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                ln = lp[:, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                Ic |= (ln == j.float()) & (F.cosine_similarity(G_F, Gn, dim=1, eps=1e-8) < self.cosine_threshold)

            correction = self.gamma * m.squeeze(1) * Ic.float() * alpha
            mask_j = F.one_hot(j, num_classes=K).permute(0, 3, 1, 2).float()

        # SAGS via pseudo-loss (AMP-compatible):
        # subtracting correction from logits_j gradient ≡ adding -correction*logits_j to loss
        loss_sags = -(correction.detach().unsqueeze(1) * mask_j.detach() * logits).sum()
        return self.ce_fn(logits, labels) + loss_sags

    def forward_inference(self, images):
        with torch.no_grad():
            out = self.backbone(images)
            return (out[0] if isinstance(out, tuple) else out).argmax(dim=1)


def build_model(cfg: dict) -> BaseModelWrapper:
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(cfg["model"]["backbone"], cfg["data"]["num_classes"], cfg["model"].get("pretrained", True))
    return SAGSModel(backbone=backbone, cfg=cfg)
