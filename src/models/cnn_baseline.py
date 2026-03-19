import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    """
    Simple CNN baseline for time-series classification.
    Input: (batch, channels, seq_len)
    Output: logits
    """
    def __init__(self, input_channels: int = 1, num_classes: int = 5,
                 conv_filters: list = [64, 128, 256], kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for out_ch in conv_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            in_ch = out_ch
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_filters[-1], num_classes)

    def forward(self, x):
        # x: (batch, C, T)
        for layer in self.conv_layers:
            x = layer(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, features)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
