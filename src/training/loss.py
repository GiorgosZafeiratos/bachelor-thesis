import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss with optional class weighting.
    """
    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: (batch, num_classes)
        targets: (batch,)
        """
        return self.ce(logits, targets)

class AttentionRegularizedLoss(nn.Module):
    """
    Combines cross-entropy with L1 regularization on attention weights.
    """
    def __init__(self, alpha: float = 1e-4, weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, attn_weights: torch.Tensor = None):
        loss = self.ce(logits, targets)
        if attn_weights is not None:
            attn_reg = attn_weights.abs().mean()
            loss = loss + self.alpha * attn_reg
        return loss
