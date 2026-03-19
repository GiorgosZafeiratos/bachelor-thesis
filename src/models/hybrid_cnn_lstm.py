import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, V)
        return context, attn_weights

class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM with attention for multimodal biomedical signals.
    Input: (batch, C, T)
    Output: logits + attention weights
    """
    def __init__(self, input_channels: int = 1, cnn_filters: list = [64, 128, 256],
                 kernel_size: int = 3, lstm_hidden: int = 128, lstm_layers: int = 2,
                 num_classes: int = 5, dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for out_ch in cnn_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            in_ch = out_ch

        self.lstm = nn.LSTM(input_size=cnn_filters[-1], hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.attention = SelfAttention(lstm_hidden * (2 if bidirectional else 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # x: (batch, C, T)
        for layer in self.conv_layers:
            x = layer(x)  # (batch, features, T')
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        attn_out, attn_weights = self.attention(lstm_out)
        # Pool over sequence
        pooled = attn_out.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits, attn_weights
