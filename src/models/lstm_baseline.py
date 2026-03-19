import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    """
    Simple LSTM baseline for sequence classification.
    Input: (batch, channels, seq_len)
    Output: logits
    """
    def __init__(self, input_channels: int = 1, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 5, dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # x: (batch, C, T) -> transpose for LSTM
        x = x.transpose(1, 2)  # (batch, T, C)
        out, _ = self.lstm(x)  # (batch, T, hidden)
        out = out[:, -1, :]    # last timestep
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
