import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeCNN(nn.Module):
    """
    Lightweight CNN for edge devices. 
    Reduced parameters for real-time inference.
    """
    def __init__(self, input_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def export_torchscript(self, example_input):
        """
        Export model to TorchScript for edge deployment.
        """
        self.eval()
        return torch.jit.trace(self, example_input)
