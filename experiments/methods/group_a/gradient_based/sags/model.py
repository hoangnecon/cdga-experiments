"""
Method: SAGS (Spatially-Aware Gradient Surgery)
Component: Model Wrapper

Ref: docs/14_sags_detailed_mathematics.md, docs/12_sags_inductive_biases.md
"""
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

    def train_mode(self) -> None:
        self.backbone.train()

    def eval_mode(self) -> None:
        self.backbone.eval()

    def _get_class_weight(self) -> torch.Tensor:
        """Get (K, C) classification weight W."""
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1].weight.squeeze(-1).squeeze(-1)

    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self.backbone(images)
        logits = out[0] if isinstance(out, tuple) else out

        if boundary_mask is None:
            boundary_mask = torch.zeros(labels.shape, device=labels.device)

        W = self._get_class_weight()
        gamma = self.gamma
        threshold = self.cosine_threshold

        # Resize ground truth to logit spatial resolution
        _, _, H, W_s = logits.shape
        if labels.shape[-2:] != (H, W_s):
            k = F.interpolate(labels.float().unsqueeze(1), size=(H, W_s), mode='nearest').squeeze(1).long()
        else:
            k = labels
        if boundary_mask.shape[-2:] != (H, W_s):
            mask = F.interpolate(boundary_mask.float(), size=(H, W_s), mode='nearest')
        else:
            mask = boundary_mask.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Compute SAGS-modified logit gradient analytically
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            kc = k.clamp(0, probs.shape[1] - 1)

            # G_Z = probs - one_hot(k)
            G_Z = probs.clone()
            G_Z.scatter_(1, kc.unsqueeze(1), probs.gather(1, kc.unsqueeze(1)) - 1.0)

            # G_F = W^T · G_Z
            G_F = torch.einsum('kc,bkhw->bchw', W, G_Z)

            # Competing class j
            pm = probs.clone()
            pm.scatter_(1, kc.unsqueeze(1), 0.0)
            j = pm.argmax(dim=1)

            # w_j projection
            B, C = G_F.shape[0], G_F.shape[1]
            w_j = W[j.reshape(-1)].view(B, H, W_s, C).permute(0, 3, 1, 2)
            wj_norm2 = (w_j * w_j).sum(dim=1, keepdim=True).clamp(min=1e-8)
            dot = (G_F * w_j).sum(dim=1, keepdim=True)
            alpha = (dot / wj_norm2).squeeze(1)

            # 3×3 conflict detection
            G_pad = F.pad(G_F, (1, 1, 1, 1), mode='replicate')
            lab_pad = F.pad(kc.float(), (1, 1, 1, 1), mode='replicate')
            I_c = torch.zeros(B, H, W_s, dtype=torch.bool, device=logits.device)
            for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                Gn = G_pad[:, :, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                ln = lab_pad[:, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                cos_val = F.cosine_similarity(G_F, Gn, dim=1, eps=1e-8)
                I_c = I_c | ((ln == j.float()) & (cos_val < threshold))

            # Build modified gradient G_Z_mod
            I = I_c.float()
            correction = gamma * mask.squeeze(1) * I * alpha
            G_Z_mod = G_Z.clone()
            for b in range(B):
                G_Z_mod[b, j[b]] -= correction[b]

        # Use G_Z_mod as the logit gradient by register_hook
        def apply_sags_grad(grad):
            return G_Z_mod * grad  # grad is 1.0 for scalar loss

        logits.register_hook(apply_sags_grad)

        return self.ce_fn(logits, labels)

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
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
    return SAGSModel(backbone=backbone, cfg=cfg)
