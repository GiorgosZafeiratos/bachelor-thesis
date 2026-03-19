import torch
from torch.utils.data import DataLoader
from pathlib import Path
import importlib

class Benchmark:
    """
    Utilities to benchmark multiple models on the same dataset.
    """
    def __init__(self, dataset_loader: DataLoader, device: torch.device):
        self.dataset_loader = dataset_loader
        self.device = device

    def load_checkpoint(self, model_class: str, checkpoint_path: str, model_kwargs: dict = None):
        """
        Dynamically load a model class and weights.
        """
        module_name, class_name = model_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)
        model = ModelClass(**(model_kwargs or {}))
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def evaluate_models(self, model_paths: dict):
        """
        model_paths: dict {model_name: (model_class_path, checkpoint_path, model_kwargs)}
        Returns: dict of metrics
        """
        from .metrics import Metrics
        results = {}
        for name, (cls_path, ckpt, kwargs) in model_paths.items():
            model = self.load_checkpoint(cls_path, ckpt, kwargs)
            metrics = Metrics.batch_metrics(model, self.dataset_loader, self.device)
            results[name] = metrics
        return results
