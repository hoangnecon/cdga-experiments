"""
Method: SAGS (Spatially-Aware Gradient Surgery)
Component: Model Wrapper — simplest possible implementation

SAGS hooks the logit gradient G_Z and modifies it:
  G_Z_j_mod = G_Z_j - gamma * S * I_conflict * <G_F, w_j>/||w_j||^2

This removes the competing class w_j component from the feature gradient,
since W^T * (delta * e_j) = delta * w_j.

Hooks directly on logits.register_hook() — fires in backward pass.
No autograd.Function, no feature hook, no timing issues.

Ref: docs/14_sags_detailed_mathematics.md, docs/12_sags_inductive_biases.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbones.base_wrapper import BaseModelWrapper


class SAGSModel(BaseModelWrapper):
    """SAGS Model Wrapper — logit-level gradient surgery."""

    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.gamma = cfg["sags"].get("gamma", 1.0)
        self.cosine_threshold = cfg["sags"].get("cosine_threshold", 0.0)
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.grad_stats: dict = {}

    def train_mode(self) -> None:
        self.backbone.train()

    def eval_mode(self) -> None:
        self.backbone.eval()

    def _find_classification_weight(self) -> torch.Tensor:
        """Find W (K x C) — the last Conv2d weight in the backbone."""
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError("No Conv2d found in backbone")
        return convs[-1].weight.squeeze(-1).squeeze(-1)  # (K, C)

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

        # --- SAGS backward hook on logit gradient ---
        gamma = self.gamma
        threshold = self.cosine_threshold
        W = self._find_classification_weight()  # (K, C)
        K, C = W.shape
        probs = F.softmax(logits.detach(), dim=1)
        k_labels = labels.clone()

        # Capture current state for the hook closure
        def sags_hook(G_Z: torch.Tensor) -> torch.Tensor:
            if gamma <= 0:
                return G_Z

            B = G_Z.shape[0]
            _, _, HZ, WZ = G_Z.shape

            with torch.no_grad():
                # Resize labels/mask to logit resolution
                k = k_labels
                if k.shape[-2:] != (HZ, WZ):
                    k = F.interpolate(k.float().unsqueeze(1), size=(HZ, WZ), mode='nearest').squeeze(1).long()
                mask = boundary_mask
                if mask.shape[-2:] != (HZ, WZ):
                    mask = F.interpolate(mask.float(), size=(HZ, WZ), mode='nearest')
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                # G_F = W^T · G_Z
                G_F = torch.einsum('kc,bkhw->bchw', W, G_Z)  # (B, C, HZ, WZ)

                # Step 1: competing class j = argmax_{c≠k} p_c
                pm = probs.clone()
                pm.scatter_(1, k.unsqueeze(1), 0.0)
                j = pm.argmax(dim=1)  # (B, HZ, WZ)

                # Step 2: projection coefficient <G_F, w_j>/||w_j||^2
                w_j = W[j.reshape(-1)].view(B, HZ, WZ, C).permute(0, 3, 1, 2)  # (B, C, HZ, WZ)
                wj_norm2 = (w_j * w_j).sum(dim=1, keepdim=True).clamp(min=1e-8)
                dot = (G_F * w_j).sum(dim=1, keepdim=True)
                alpha = (dot / wj_norm2).squeeze(1)  # (B, HZ, WZ)

                # Step 3: 3×3 conflict detection
                G_pad = F.pad(G_F, (1, 1, 1, 1), mode='replicate')
                lab_pad = F.pad(k.float(), (1, 1, 1, 1), mode='replicate')
                I_c = torch.zeros(B, HZ, WZ, dtype=torch.bool, device=G_Z.device)
                for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    Gn = G_pad[:, :, 1+dx:1+dx+HZ, 1+dy:1+dy+WZ]
                    ln = lab_pad[:, 1+dx:1+dx+HZ, 1+dy:1+dy+WZ]
                    cos_val = F.cosine_similarity(G_F, Gn, dim=1, eps=1e-8)
                    I_c = I_c | ((ln == j.float()) & (cos_val < threshold))

                # Step 4: G_Z_j -= gamma * mask * I_conflict * alpha
                I = I_c.float()
                correction = gamma * mask.squeeze(1) * I * alpha

                G_Z_mod = G_Z.clone()
                for b in range(B):
                    G_Z_mod[b, j[b]] -= correction[b]

                # Diagnostics
                self.grad_stats = {
                    'conflict_ratio': I_c.float().mean().item(),
                    'proj_magnitude': alpha.abs().mean().item(),
                }

                return G_Z_mod

        # Register the hook — fires when gradient reaches logits during backward
        logits.register_hook(sags_hook)

        return self.ce_fn(logits, labels)

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(images)
            logits = out[0] if isinstance(out, tuple) else out
            return logits.argmax(dim=1)


def build_model(cfg: dict) -> BaseModelWrapper:
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return SAGSModel(backbone=backbone, cfg=cfg)
