"""
Method: SAGS (Spatially-Aware Gradient Surgery)
Component: Model Wrapper with backward hook

SAGS performs spatial gradient surgery via a backward hook:
  1. Identifies competing class j = argmax p_c for c≠k
  2. Checks 3×3 neighborhood for cosine similarity conflict  
  3. Projects out competing w_j component at conflicted boundary pixels

Key insight (from docs/14_sags_detailed_mathematics.md):
  At boundary pixels, G_F has a +p_j·w_j component pushing features away from class j.
  But interior pixels of class j dominate this direction. Removing the w_j component
  at boundaries eliminates destructive interference while preserving discriminative pull.

Ref:
    docs/14_sags_detailed_mathematics.md
    docs/12_sags_inductive_biases.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbones.base_wrapper import BaseModelWrapper


class SAGSModel(BaseModelWrapper):
    """SAGS Model Wrapper — Spatially-Aware Gradient Surgery.
    
    Attaches a backward hook on the feature map before classification head.
    Zero inference overhead — hook only active during training.
    """

    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.gamma = cfg["sags"].get("gamma", 1.0)
        self.cosine_threshold = cfg["sags"].get("cosine_threshold", 0.0)
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Hook state — set each training step
        self._feature_hook_handle = None
        self._features_for_hook: torch.Tensor | None = None
        self._weight_for_hook: torch.Tensor | None = None
        self._labels_for_hook: torch.Tensor | None = None
        self._probs_for_hook: torch.Tensor | None = None
        self._mask_for_hook: torch.Tensor | None = None
        self.grad_stats: dict = {}

    def _capture_features(self, module, input, output):
        """Forward hook: capture features right before classification head."""
        # Detach and clone to avoid interfering with forward graph
        self._features_for_hook = output

    def _sags_backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward hook: modify gradient on feature map (G_F)."""
        gamma = self.gamma
        if gamma <= 0 or self._weight_for_hook is None:
            return grad

        G_F = grad  # (B, C, H, W)
        B, C, H, W = G_F.shape
        K = self._weight_for_hook.shape[0]
        threshold = self.cosine_threshold

        # --- Identify competing class j = argmax p_c for c≠k ---
        with torch.no_grad():
            k = self._labels_for_hook  # (B, H, W)
            probs = self._probs_for_hook
            # Zero out correct class probability to find argmax over wrong classes
            probs_masked = probs.clone()
            k_clamped = k.clamp(0, K - 1)
            probs_masked.scatter_(1, k_clamped.unsqueeze(1), 0.0)
            j = probs_masked.argmax(dim=1)  # (B, H, W)

            # --- Compute w_j projection ---
            weight = self._weight_for_hook  # (K, C)
            # Gather w_j for each pixel: (B, H, W) → indices → (B, H, W, C)
            w_j = weight[j.view(-1)].view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

            # proj_{w_j}(G_F) = <G_F, w_j>/||w_j||² * w_j
            w_j_norm_sq = (w_j * w_j).sum(dim=1, keepdim=True).clamp(min=1e-8)
            dot = (G_F * w_j).sum(dim=1, keepdim=True)
            proj_wj = (dot / w_j_norm_sq) * w_j  # (B, C, H, W)

            # --- 3×3 neighborhood cosine similarity conflict detection ---
            G_F_pad = F.pad(G_F, (1, 1, 1, 1), mode='replicate')
            labels_pad = F.pad(k_clamped.float(), (1, 1, 1, 1), mode='replicate')

            I_conflict = torch.zeros(B, H, W, device=G_F.device)
            offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dx, dy in offsets:
                G_F_n = G_F_pad[:, :, 1+dx:1+dx+H, 1+dy:1+dy+W]
                label_n = labels_pad[:, 1+dx:1+dx+H, 1+dy:1+dy+W]
                cos = F.cosine_similarity(G_F, G_F_n, dim=1, eps=1e-8)
                # Conflict: neighbor's GT label IS the competing class j AND cosine < threshold
                conflict = (label_n == j.float()) & (cos < threshold)
                I_conflict = I_conflict | conflict

            # --- Apply surgical projection ---
            S = self._mask_for_hook.float()  # (B, 1, H, W) or (B, H, W)
            if S.dim() == 3:
                S = S.unsqueeze(1)
            I = I_conflict.float().unsqueeze(1)  # (B, 1, H, W)

            # Diagnostic stats
            conflict_ratio = I_conflict.float().mean().item()
            proj_mag = proj_wj.norm(dim=1).mean().item()
            orig_mag = G_F.norm(dim=1).mean().item()
            self.grad_stats = {
                'conflict_ratio': conflict_ratio,
                'proj_magnitude': proj_mag,
                'orig_magnitude': orig_mag,
                'snr': orig_mag / max(proj_mag, 1e-8),
            }

            # G_F_modified = G_F - gamma * S * I_conflict * proj_wj
            correction = gamma * S * I * proj_wj
            return G_F - correction

    def train_mode(self) -> None:
        """Register hooks — only during training."""
        self.backbone.train()
        # Hook the feature extraction point before classification head
        if hasattr(self.backbone, 'backbone'):
            target = self.backbone.backbone
        else:
            target = self.backbone
        # Hook last feature-producing layer
        self._feature_hook_handle = target.register_forward_hook(self._capture_features)

    def eval_mode(self) -> None:
        """Remove hooks — zero inference overhead."""
        self.backbone.eval()
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()
            self._feature_hook_handle = None

    def _find_seg_head_conv(self) -> nn.Conv2d:
        """Find the final 1x1 conv layer of segmentation head."""
        seg_head = None
        for name in ['head', 'decode_head', 'seg_head']:
            if hasattr(self.backbone, name):
                seg_head = getattr(self.backbone, name)
                break
        if seg_head is None:
            raise AttributeError("Cannot find segmentation head")
        convs = [m for m in seg_head.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise AttributeError("No Conv2d in seg head")
        return convs[-1]

    def forward_train(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Get features + logits from backbone
        out = self.backbone(images)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        if boundary_mask is None:
            boundary_mask = torch.zeros(
                labels.shape[0], 1, labels.shape[1], labels.shape[2],
                device=labels.device
            )

        # Prepare context for backward hook
        seg_conv = self._find_seg_head_conv()
        self._weight_for_hook = seg_conv.weight.squeeze(-1).squeeze(-1)  # (K, C)
        self._labels_for_hook = labels
        self._probs_for_hook = F.softmax(logits.detach(), dim=1)  # detach: don't affect grad
        self._mask_for_hook = boundary_mask

        # Register backward hook on captured features
        if self._features_for_hook is not None and self._features_for_hook.requires_grad:
            self._features_for_hook.register_hook(self._sags_backward_hook)

        loss = self.ce_fn(logits, labels)

        # Cleanup
        self._features_for_hook = None
        self._labels_for_hook = None
        self._probs_for_hook = None
        self._mask_for_hook = None

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
    """Factory function for SAGSModel."""
    from shared.backbones.geoseg_adapter import load_geoseg_backbone
    backbone = load_geoseg_backbone(
        model_name=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    )
    return SAGSModel(backbone=backbone, cfg=cfg)
