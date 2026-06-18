"""
Method: SAGS (Spatially-Aware Gradient Surgery)
Component: Model Wrapper with backward hook

SAGS performs spatial gradient surgery via a backward hook:
  1. Identifies competing class j = argmax p_c for c≠k
  2. Checks 3×3 neighborhood for cosine similarity conflict  
  3. Projects out competing w_j component at conflicted boundary pixels

Ref: docs/14_sags_detailed_mathematics.md, docs/12_sags_inductive_biases.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbones.base_wrapper import BaseModelWrapper


class SAGSModel(BaseModelWrapper):
    """SAGS Model Wrapper — Spatially-Aware Gradient Surgery."""

    def __init__(self, backbone: nn.Module, cfg: dict) -> None:
        super().__init__(backbone, cfg)
        ignore_index = cfg["data"].get("ignore_index", 255)
        self.gamma = cfg["sags"].get("gamma", 1.0)
        self.cosine_threshold = cfg["sags"].get("cosine_threshold", 0.0)
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._forward_hook_handle = None
        self.grad_stats: dict = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------
    def train_mode(self) -> None:
        self.backbone.train()
        seg_head_0 = self._get_seg_head_first_conv()
        self._forward_hook_handle = seg_head_0.register_forward_pre_hook(self._on_pre_forward)

    def eval_mode(self) -> None:
        self.backbone.eval()
        if self._forward_hook_handle is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None

    def _get_seg_head(self):
        if hasattr(self.backbone, 'decoder') and hasattr(self.backbone.decoder, 'segmentation_head'):
            return self.backbone.decoder.segmentation_head
        raise AttributeError(f"{type(self.backbone).__name__}: no decoder.segmentation_head")

    def _get_seg_head_first_conv(self) -> nn.Conv2d:
        seg_head = self._get_seg_head()
        for m in seg_head.children():
            if isinstance(m, nn.Conv2d):
                return m  # first Conv2d
        raise AttributeError("No Conv2d in seg_head")

    def _get_seg_head_last_conv(self) -> nn.Conv2d:
        seg_head = self._get_seg_head()
        last = None
        for m in seg_head.children():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise AttributeError("No Conv2d in seg_head")
        return last  # last Conv2d = classification layer

    # ------------------------------------------------------------------
    # Forward hook — capture feature map F before classification head
    # ------------------------------------------------------------------
    def _on_pre_forward(self, module, args):
        """Pre-forward hook: captures input to first Conv2d of seg_head = feature map F."""
        self._fwd_features = args[0]  # (B, C, H, W) — raw decoder features
        self._fwd_labels = None
        self._fwd_probs = None
        self._fwd_mask = None
        self._fwd_weight = None

    def _set_hook_context(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor,
        boundary_mask: torch.Tensor,
        seg_weight: torch.Tensor,
    ):
        """Set context that the backward hook will read."""
        self._fwd_labels = labels
        self._fwd_probs = F.softmax(logits.detach(), dim=1)
        self._fwd_mask = boundary_mask
        self._fwd_weight = seg_weight

    # ------------------------------------------------------------------
    # Backward hook — gradient surgery via register_hook on features
    # ------------------------------------------------------------------
    def _make_backward_hook(self):
        """Return a closure that captures current context for gradient modification."""
        gamma = self.gamma
        threshold = self.cosine_threshold
        labels = self._fwd_labels
        probs = self._fwd_probs
        mask = self._fwd_mask
        weight = self._fwd_weight  # (K, C)

        def sags_grad_hook(grad: torch.Tensor) -> torch.Tensor:
            if gamma <= 0 or weight is None or labels is None:
                return grad

            G_F = grad
            B, C, H, W = G_F.shape
            K = weight.shape[0]

            with torch.no_grad():
                # Step 1: competing class j = argmax_{c≠k} p_c
                k = labels.clamp(0, K - 1)
                pm = probs.clone()
                pm.scatter_(1, k.unsqueeze(1), 0.0)
                j = pm.argmax(dim=1)  # (B, H, W)

                # Step 2: w_j prototype for each pixel
                w_j = weight[j.view(-1)].view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
                wj_norm2 = (w_j * w_j).sum(dim=1, keepdim=True).clamp(min=1e-8)
                dot = (G_F * w_j).sum(dim=1, keepdim=True)
                proj_wj = (dot / wj_norm2) * w_j  # (B, C, H, W)

                # Step 3: 3×3 conflict detection
                G_pad = F.pad(G_F, (1, 1, 1, 1), mode='replicate')
                lab_pad = F.pad(k.float(), (1, 1, 1, 1), mode='replicate')
                I_c = torch.zeros(B, H, W, device=G_F.device)
                for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    Gn = G_pad[:, :, 1+dx:1+dx+H, 1+dy:1+dy+W]
                    ln = lab_pad[:, 1+dx:1+dx+H, 1+dy:1+dy+W]
                    cos = F.cosine_similarity(G_F, Gn, dim=1, eps=1e-8)
                    I_c = I_c | ((ln == j.float()) & (cos < threshold))

                # Step 4: apply surgery
                S = mask.float()
                if S.dim() == 3:
                    S = S.unsqueeze(1)
                I = I_c.float().unsqueeze(1)

                # Diagnostics
                cr = I_c.float().mean().item()
                pm_val = proj_wj.norm(dim=1).mean().item()
                self.grad_stats = {
                    'conflict_ratio': cr,
                    'proj_magnitude': pm_val,
                    'orig_magnitude': G_F.norm(dim=1).mean().item(),
                }
                if pm_val > 1e-8:
                    self.grad_stats['snr'] = self.grad_stats['orig_magnitude'] / pm_val

                return G_F - gamma * S * I * proj_wj

        return sags_grad_hook

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
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
            boundary_mask = torch.zeros(
                labels.shape[0], 1, labels.shape[1], labels.shape[2],
                device=labels.device
            )

        # Register backward hook on captured feature tensor
        features = self._fwd_features
        if features is not None and features.requires_grad:
            seg_conv = self._get_seg_head_last_conv()
            W = seg_conv.weight.squeeze(-1).squeeze(-1)  # (K, C)
            self._set_hook_context(labels, logits, boundary_mask, W)
            features.register_hook(self._make_backward_hook())

        return self.ce_fn(logits, labels)

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
