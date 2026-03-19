import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Simple fusion module for multiple modality embeddings.
    """
    def __init__(self, input_dims: list, hidden_dim: int = 256, num_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.total_dim = sum(input_dims)
        self.fc1 = nn.Linear(self.total_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: list):
        """
        features: list of tensors (batch, feature_dim)
        """
        x = torch.cat(features, dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
