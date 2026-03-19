import torch
from pathlib import Path
from deployment.edge_model import EdgeCNN

class InferenceEngine:
    """
    Batch and real-time inference engine for EdgeCNN.
    """
    def __init__(self, checkpoint_path: str, device: torch.device = torch.device("cpu")):
        self.device = device
        self.model = EdgeCNN()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, x: torch.Tensor):
        """
        Predict a batch of signals.
        x: (batch, C, T)
        """
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu()

    def predict_stream(self, stream_gen):
        """
        Real-time streaming prediction.
        stream_gen: generator yielding (C, T) tensors
        """
        for x in stream_gen:
            x = x.unsqueeze(0)  # add batch dim
            yield self.predict_batch(x)
