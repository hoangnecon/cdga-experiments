"""
Method: SAGS (Spatially-Aware Gradient Surgery)
Component: Model Wrapper with autograd.Function

SAGS performs spatial gradient surgery by modifying the logit gradient G_Z:
  G_Z_j_mod = p_j - γ·S·I_conflict·⟨G_F, w_j⟩/||w_j||²

This is equivalent to removing w_j component from feature gradient G_F,
since W^T·(α·e_j) = α·w_j, and only the j-th logit gradient channel changes.

Pattern: torch.autograd.Function (proven PCGrad pattern).

Ref: docs/14_sags_detailed_mathematics.md, docs/12_sags_inductive_biases.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbones.base_wrapper import BaseModelWrapper


class SAGSAutograd(torch.autograd.Function):
    """Autograd function: forward=CE loss, backward=SAGS-modified G_Z."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundary_mask: torch.Tensor,
        backbone: nn.Module,
        ce_fn: nn.Module,
        gamma: float,
        cosine_threshold: float,
    ) -> torch.Tensor:
        loss = ce_fn(logits, labels)
        ctx.save_for_backward(logits.detach(), labels, boundary_mask)
        ctx.backbone = backbone
        ctx.gamma = gamma
        ctx.cosine_threshold = cosine_threshold
        ctx.has_surgery = False  # set in backward
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor) -> tuple:
        logits, labels, mask = ctx.saved_tensors
        gamma = ctx.gamma
        threshold = ctx.cosine_threshold

        if gamma <= 0:
            return grad_loss, None, None, None, None, None, None

        # Find classification weight W
        backbone = ctx.backbone
        convs = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            return grad_loss, None, None, None, None, None, None
        W = convs[-1].weight.squeeze(-1).squeeze(-1)  # (K, C)
        K, C = W.shape
        B, _, H, W_s = logits.shape

        with torch.no_grad():
            # Compute G_Z = probs - one_hot(label)
            probs = F.softmax(logits, dim=1)
            k = labels.clamp(0, K - 1)
            G_Z = probs.clone()
            G_Z.scatter_(1, k.unsqueeze(1), probs.gather(1, k.unsqueeze(1)) - 1.0)

            # Resize labels and mask to logit resolution
            if labels.shape[-2:] != (H, W_s):
                k = F.interpolate(k.float().unsqueeze(1), size=(H, W_s), mode='nearest').squeeze(1).long()
            if mask.shape[-2:] != (H, W_s):
                mask = F.interpolate(mask.float(), size=(H, W_s), mode='nearest')
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

            # Compute G_F = W^T · G_Z
            G_F = torch.einsum('kc,bkhw->bchw', W, G_Z)  # (B, C, H, W)

            # Step 1: competing class j
            pm = probs.clone()
            pm.scatter_(1, k.unsqueeze(1), 0.0)
            j = pm.argmax(dim=1)  # (B, H, W_s)

            # Step 2: w_j and its projection coefficient
            w_j = W[j.view(-1)].view(B, H, W_s, C).permute(0, 3, 1, 2)  # (B, C, H, W_s)
            wj_norm2 = (w_j * w_j).sum(dim=1, keepdim=True).clamp(min=1e-8)
            dot = (G_F * w_j).sum(dim=1, keepdim=True)                     # (B, 1, H, W_s)
            alpha = dot / wj_norm2  # projection coefficient <G_F, w_j>/||w_j||²

            # Step 3: 3×3 conflict detection
            G_pad = F.pad(G_F, (1, 1, 1, 1), mode='replicate')
            lab_pad = F.pad(k.float(), (1, 1, 1, 1), mode='replicate')
            I_c = torch.zeros(B, H, W_s, device=G_F.device)
            for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                Gn = G_pad[:, :, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                ln = lab_pad[:, 1+dx:1+dx+H, 1+dy:1+dy+W_s]
                cos_val = F.cosine_similarity(G_F, Gn, dim=1, eps=1e-8)
                I_c = I_c | ((ln == j.float()) & (cos_val < threshold)).bool()

            # Step 4: modify G_Z — only j-th channel
            I = I_c.float().unsqueeze(1)                       # (B, 1, H, W_s)
            correction = gamma * mask * I * alpha.squeeze(1)   # (B, H, W_s)

            # G_Z_j -= correction (remove w_j component)
            G_Z_mod = G_Z.clone()
            G_Z_mod.scatter_(1, j.unsqueeze(1), G_Z_mod.gather(1, j.unsqueeze(1)) - correction.unsqueeze(1))

            ctx.has_surgery = True

        return grad_loss * G_Z_mod, None, None, None, None, None, None


class SAGSModel(BaseModelWrapper):
    """SAGS Model Wrapper — Spatially-Aware Gradient Surgery."""

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

    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self.backbone(images)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        if boundary_mask is None:
            boundary_mask = torch.zeros(labels.shape, device=labels.device)

        return SAGSAutograd.apply(
            logits, labels, boundary_mask,
            self.backbone, self.ce_fn,
            self.gamma, self.cosine_threshold,
        )

    def forward_inference(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(images)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            return logits.argmax(dim=1)


def build_model(cfg: dict) -> BaseModelWrapper:
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return SAGSModel(backbone=backbone, cfg=cfg)
