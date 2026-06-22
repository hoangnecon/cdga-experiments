"""
Method: Active Boundary Loss (ABL)
Component: Loss Definition
Ref: 
    - Wang et al., AAAI 2022: "Active Boundary Loss for Semantic Segmentation"
    - Official code: tmp/active-boundary-loss/abl.py
    - Lovász-Softmax: Berman et al., CVPR 2018
    - rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute Lovász gradient (Eq. 6 in Berman et al. 2018)."""
    gts = gt_sorted.sum(dim=1, keepdim=True)      # (B, 1)
    intersection = gts - gt_sorted.float().cumsum(dim=1)  # (B, N)
    union = gts + (1 - gt_sorted).float().cumsum(dim=1)   # (B, N)
    jaccard = 1.0 - intersection / union.clamp(min=1e-7)
    if jaccard.shape[1] > 1:
        jaccard[:, 1:] = jaccard[:, 1:] - jaccard[:, :-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    """Multi-class Lovász-Softmax IoU loss — Berman et al., CVPR 2018.
    
    Used in the ABL paper (Wang et al., AAAI 2022) as the IoU component
    of the CE + IoU + ABL training objective.
    """
    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(logits, dim=1)
        C = probas.shape[1]
        B, H, W = labels.shape
        N = H * W  # pixels per image
        labels = labels.clamp(0, C - 1)
        
        loss = torch.tensor(0.0, device=logits.device)
        n_class = 0
        for c in range(C):
            if c == self.ignore_index:
                continue
            fg = (labels == c).float().view(B, -1)                   # (B, N)
            if fg.sum() == 0:
                continue
            errors = (fg - probas[:, c].view(B, -1)).abs()            # (B, N)
            errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
            fg_sorted = fg.gather(1, perm)
            grad = _lovasz_grad(fg_sorted)
            # Normalize by total pixels (B*N) for proper loss scale
            loss += torch.dot(F.relu(errors_sorted.reshape(-1)), grad.reshape(-1)) / (B * N)
            n_class += 1
        
        return loss / max(1, n_class)


class ABLLoss(nn.Module):
    """Active Boundary Loss — faithful to Wang et al., AAAI 2022.
    
    Args:
        max_N_ratio: Maximum fraction of pixels considered as PDB (default 1/100).
        ignore_index: Label value to ignore.
        max_clip_dist: Maximum distance for clamping weight.
        is_detach: Whether to detach neighbor logits in KL computation.
        label_smoothing: Label smoothing for direction CE (paper uses 0.2).
    """
    def __init__(
        self,
        max_N_ratio: float = 1.0 / 100.0,
        ignore_index: int = 255,
        max_clip_dist: float = 20.0,
        is_detach: bool = True,
        label_smoothing: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_N_ratio = max_N_ratio
        self.ignore_index = ignore_index
        self.max_clip_dist = max_clip_dist
        self.is_detach = is_detach
        self.label_smoothing = label_smoothing

    @staticmethod
    def kl_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """KL(b || a) = softmax(b) * (log_softmax(b) - log_softmax(a)).
        
        Matches the official kl_div in tmp/active-boundary-loss/abl.py line 14-15:
          def kl_div(a,b): # q,p
              return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))
        
        Args:
            a: (B, C, H, W) conditioning distribution.
            b: (B, C, H, W) reference distribution.
        Returns:
            (B, C, H, W) per-pixel per-class KL divergence.
        """
        return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))

    # ------------------------------------------------------------------
    #  PDB Detection — Adaptive Epsilon Threshold (matches official lines 65-87)
    # ------------------------------------------------------------------
    def logits2boundary(self, logits: torch.Tensor) -> torch.Tensor:
        """Detect potential boundary pixels via KL divergence + adaptive threshold.
        
        Official code lines 65-87:
          - KL between vertically adjacent pixels: KL(upper || lower)
          - KL between horizontally adjacent pixels: KL(left || right)
          - Sum to get combined KL map
          - Adaptive epsilon: increase eps *= 1.2 until boundary pixels <= max_N
          - Dilate with 3x3 conv
        
        Args:
            logits: (B, C, H, W) model output before softmax.
        Returns:
            (B, H, W) boolean mask where True = potential boundary pixel.
        """
        B, C, H, W = logits.shape
        max_N = int(H * W * self.max_N_ratio)

        # KL in 2 principal directions (official lines 69-74)
        # kl_ud: a=bottom row, b=top row -> KL(top || bottom)
        kl_ud = self.kl_div(
            logits[:, :, 1:, :],    # a: bottom (lines 2:H)
            logits[:, :, :-1, :]     # b: top    (lines 0:H-1)
        ).sum(dim=1, keepdim=True)   # sum over classes

        # kl_lr: a=right col, b=left col -> KL(left || right)
        kl_lr = self.kl_div(
            logits[:, :, :, 1:],     # a: right  (cols 1:W)
            logits[:, :, :, :-1]      # b: left   (cols 0:W-1)
        ).sum(dim=1, keepdim=True)

        # Pad to original spatial size (official lines 71-74)
        kl_ud = F.pad(kl_ud, [0, 0, 0, 1], mode='constant', value=0)
        kl_lr = F.pad(kl_lr, [0, 1, 0, 0], mode='constant', value=0)

        kl_combine = kl_ud + kl_lr  # (B, 1, H, W)

        # Adaptive epsilon threshold (official lines 76-82)
        eps = 1e-5
        while True:
            kl_combine_bin = (kl_combine > eps).float()
            if kl_combine_bin.sum() > max_N:
                eps *= 1.2
            else:
                break

        # Dilate with 3x3 convolution (official lines 83-87)
        dilate_weight = torch.ones(1, 1, 3, 3, device=logits.device)
        edge2 = F.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # (B, H, W)
        kl_combine_bin = edge2 > 0

        return kl_combine_bin

    # ------------------------------------------------------------------
    #  Distance Map — GPU-based (equivalent to scipy distance_transform_edt)
    # ------------------------------------------------------------------
    def compute_distance_map_gpu(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute distance transform from GT boundary via iterative GPU dilation.
        
        Equivalent behavior to scipy.ndimage.distance_transform_edt
        used in the official code (lines 17-24, 42-49, 165-171).
        
        Produces: 0 at boundary pixels, positive distance (up to 20) inside regions.
        
        Args:
            labels: (B, H, W) integer label map.
        Returns:
            (B, H, W) float distance map.
        """
        B, H, W = labels.shape
        device = labels.device

        # Detect boundary via 4-connectivity diff (equivalent to official gt2boundary lines 89-101)
        padded = F.pad(labels, (1, 1, 1, 1), mode='replicate')
        up = padded[:, 0:H, 1:W+1]
        down = padded[:, 2:H+2, 1:W+1]
        left = padded[:, 1:H+1, 0:W]
        right = padded[:, 1:H+1, 2:W+2]

        boundary = (labels != up) | (labels != down) | (labels != left) | (labels != right)
        valid_mask = labels != self.ignore_index
        boundary = boundary & valid_mask

        # Iterative dilation — GPU equivalent of scipy distance_transform_edt
        boundary_float = boundary.unsqueeze(1).float()
        distance_map = torch.full_like(boundary_float, self.max_clip_dist)
        distance_map[boundary_float == 1.0] = 0.0

        for d in range(1, int(self.max_clip_dist) + 1):
            dilated = F.max_pool2d(boundary_float, kernel_size=2*d+1, stride=1, padding=d)
            mask = (dilated == 1.0) & (distance_map == self.max_clip_dist)
            distance_map[mask] = float(d)

        return distance_map.squeeze(1)

    # ------------------------------------------------------------------
    #  Direction GT & Prediction — 9-position with center exclusion
    # ------------------------------------------------------------------
    def get_direction_gt_predkl(
        self,
        pred_dist_map: torch.Tensor,
        pred_bound: torch.Tensor,
        logits: torch.Tensor,
    ):
        """Compute direction ground truth and KL-based direction prediction.
        
        Official code lines 103-163:
          - 9 positions: 8 neighbors + center (index 8)
          - direction_gt = argmin(dist_map) over 9 positions
          - Filter: direction_gt != 8 (exclude pixels where center is argmin)
          - direction_pred = KL(neighbor || center) for 8 non-center directions
        
        Args:
            pred_dist_map: (B, H, W) distance map from compute_distance_map_gpu.
            pred_bound: (B, H, W) boolean PDB mask from logits2boundary.
            logits: (B, C, H, W) model output before softmax.
        Returns:
            direction_gt: (K',) filtered direction labels [0..7].
            direction_pred: (K', 8) KL-based prediction logits.
            weight_ce: (K',) distance-based loss weights.
            OR (None, None, None) if no valid PDB pixels.
        """
        B, C, H, W = logits.shape
        max_dis = 1e5

        # Get PDB pixel indices (official lines 107-108)
        bound_indices = torch.nonzero(pred_bound, as_tuple=False)  # (K, 3): [b, h, w]

        if bound_indices.numel() == 0:
            return None, None, None

        n, x, y = bound_indices[:, 0], bound_indices[:, 1], bound_indices[:, 2]

        # Pad distance map (official line 113)
        dist_map_padded = F.pad(
            pred_dist_map.unsqueeze(1), (1, 1, 1, 1),
            mode='constant', value=max_dis
        ).squeeze(1)  # (B, H+2, W+2)

        # Permute logits to NHWC for easier indexing (official line 111)
        logits_nhwc = logits.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        # Pad logits and copy edge values (official lines 115-119)
        logits_pad = F.pad(logits_nhwc, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
        logits_pad[:, 0, :, :] = logits_pad[:, 1, :, :]
        logits_pad[:, -1, :, :] = logits_pad[:, -2, :, :]
        logits_pad[:, :, 0, :] = logits_pad[:, :, 1, :]
        logits_pad[:, :, -1, :] = logits_pad[:, :, -2, :]

        # 9-position offsets (official lines 121-127)
        # Index:  0=right, 1=left, 2=up, 3=down, 4=up-right, 5=down-right, 6=up-left, 7=down-left, 8=center
        x_range = [1, -1,  0,  0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1,  1,  1,  1, -1, -1, 0]

        K = n.numel()

        # Center logits (official line 131)
        kl_center = logits_nhwc[n, x, y]  # (K, C)

        dist_maps_list = []
        kl_maps_list = []

        for dx, dy in zip(x_range, y_range):
            # Distance value at this neighbor position (official line 134)
            # Note: x = height idx, y = width idx, dx = height offset, dy = width offset
            dist_now = dist_map_padded[n, x + dx + 1, y + dy + 1]  # (K,)
            dist_maps_list.append(dist_now.unsqueeze(0))

            if dx != 0 or dy != 0:
                # Neighbor logits (official line 138)
                logits_now = logits_pad[n, x + dx + 1, y + dy + 1, :]  # (K, C)

                if self.is_detach:
                    logits_now = logits_now.detach()

                # KL(neighbor || center) (official lines 140-145)
                # kl_div(a=center, b=neighbor) = KL(neighbor || center)
                kl_map_now = self.kl_div(kl_center, logits_now)  # (K, C)
                kl_map_now = kl_map_now.sum(dim=1)  # sum over classes (K,)
                kl_maps_list.append(kl_map_now.unsqueeze(0))  # (1, K)

        dist_maps = torch.cat(dist_maps_list, dim=0)   # (9, K)
        kl_maps = torch.cat(kl_maps_list, dim=0)        # (8, K) — 8 non-center directions

        # Direction GT: which neighbor has minimum distance (official line 149)
        direction_gt = torch.argmin(dist_maps, dim=0)  # (K,) values in [0, 8]

        # Weight: distance at the center pixel (official line 151)
        weight_ce = pred_dist_map[n, x, y]  # (K,)

        # Filter: exclude pixels where center is argmin (official lines 155-161)
        valid_direction = direction_gt != 8
        direction_gt = direction_gt[valid_direction]           # (K',)
        kl_maps = kl_maps[:, valid_direction]                  # (8, K')
        weight_ce = weight_ce[valid_direction]                 # (K',)

        if direction_gt.numel() == 0:
            return None, None, None

        # Transpose to (K', 8) for CE loss (official line 159)
        direction_pred = kl_maps.transpose(0, 1)  # (K', 8)

        return direction_gt, direction_pred, weight_ce

    # ------------------------------------------------------------------
    #  Forward — Main loss computation
    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Active Boundary Loss.
        
        Args:
            logits: (B, C, H, W) model output before softmax.
            targets: (B, H, W) ground truth labels.
        Returns:
            Scalar loss value.
        """
        B, C, H, W = logits.shape

        # 1. Detect PDB pixels via adaptive KL threshold + dilation
        pdb_mask = self.logits2boundary(logits)  # (B, H, W) bool

        if pdb_mask.sum() < 1:
            return logits.sum() * 0.0

        # 2. Compute GT distance map M (0 at boundary, positive inside)
        dist_map = self.compute_distance_map_gpu(targets)  # (B, H, W)

        # 3. Get direction GT (with center-index 8 exclusion) and KL-based direction pred
        # get_direction_gt_predkl faithfully matches official abl.py lines 103-163:
        #   - 9-position scan (8 neighbors + center idx=8)
        #   - direction_gt = argmin over 9 → filter != 8 → values [0..7]
        #   - direction_pred = KL-based logits for 8 directions (K', 8)
        #   - weight_ce = distance at each PDB pixel (K',)
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(
            dist_map, pdb_mask, logits
        )

        if direction_gt is None or direction_gt.numel() == 0:
            return logits.sum() * 0.0

        # 4. Cross-entropy loss on direction prediction (official line 193)
        loss = F.cross_entropy(direction_pred, direction_gt, reduction='none',
                               label_smoothing=self.label_smoothing)  # (K',)

        # 5. Weight by distance: clamp(M, max=20) / 20 (official lines 195-196)
        weight_ce = torch.clamp(weight_ce, max=self.max_clip_dist) / self.max_clip_dist
        loss = (loss * weight_ce).mean()

        return loss