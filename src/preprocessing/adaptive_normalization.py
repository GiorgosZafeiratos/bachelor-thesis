import torch
import torch.nn as nn

class AdaptiveNormalization(nn.Module):
    """
    Combines learnable affine transformation with statistical normalization.
    Can be applied to each channel separately.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        x: (batch, channels, seq_len)
        """
        if training:
            mean = x.mean(dim=[0, 2])
            var = x.var(dim=[0, 2], unbiased=False)
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        out = self.gamma[None, :, None] * x_norm + self.beta[None, :, None]
        return out

    @staticmethod
    def statistical_normalization(x: torch.Tensor) -> torch.Tensor:
        """
        Standard z-score normalization per channel.
        """
        mean = x.mean(dim=[0, 2], keepdim=True)
        std = x.std(dim=[0, 2], unbiased=False, keepdim=True)
        return (x - mean) / (std + 1e-5)
