"""
Method: PCGrad Baseline
Component: Loss Definition & Autograd Function
Ref: rules/CONVENTIONS.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryCELoss(nn.Module):
    """Cross Entropy computed only on boundary pixels."""
    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        loss_pixel = self.ce_fn(logits, targets)  # (B, H, W)
        weighted_loss = loss_pixel * boundary_mask.squeeze(1)
        
        valid_mask = (targets != self.ignore_index).float()
        total_valid = (valid_mask * boundary_mask.squeeze(1)).sum()
        
        if total_valid == 0:
            return weighted_loss.sum() * 0.0
            
        return weighted_loss.sum() / total_valid

class PCGradAutograd(torch.autograd.Function):
    """Custom PyTorch autograd function to implement PCGrad projection."""
    @staticmethod
    def forward(ctx, logits, labels, boundary_mask, model, ce_fn, bdy_fn):
        ctx.save_for_backward(logits, labels, boundary_mask)
        ctx.model = model
        ctx.ce_fn = ce_fn
        ctx.bdy_fn = bdy_fn
        
        # Return combined loss value
        with torch.no_grad():
            loss1 = ce_fn(logits, labels)
            loss2 = bdy_fn(logits, labels, boundary_mask)
            total_loss = loss1 + loss2
        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, boundary_mask = ctx.saved_tensors
        model = ctx.model
        ce_fn = ctx.ce_fn
        bdy_fn = ctx.bdy_fn
        
        # Filter parameters that require grad to avoid PyTorch errors in autograd.grad
        params = [p for p in model.parameters() if p.requires_grad]
        
        with torch.enable_grad():
            # 1. Gradients for CE loss
            loss1 = ce_fn(logits, labels)
            grads1 = torch.autograd.grad(loss1, params, retain_graph=True, allow_unused=True)
            
            # 2. Gradients for boundary loss
            loss2 = bdy_fn(logits, labels, boundary_mask)
            grads2 = torch.autograd.grad(loss2, params, allow_unused=True)
            
        # 3. Perform PCGrad projection and assign to parameters
        for p, g1, g2 in zip(params, grads1, grads2):
            if g1 is not None and g2 is not None:
                dot = torch.sum(g1 * g2)
                if dot < 0:
                    g1_proj = g1 - (dot / (torch.sum(g2 * g2) + 1e-8)) * g2
                    g2_proj = g2 - (dot / (torch.sum(g1 * g1) + 1e-8)) * g1
                    p.grad = (g1_proj + g2_proj) * grad_output
                else:
                    p.grad = (g1 + g2) * grad_output
            elif g1 is not None:
                p.grad = g1 * grad_output
            elif g2 is not None:
                p.grad = g2 * grad_output
                
        return None, None, None, None, None, None
