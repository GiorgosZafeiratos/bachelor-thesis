import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Metrics:
    """
    Compute standard classification metrics for evaluation.
    """
    @staticmethod
    def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, average: str = "macro"):
        """
        logits: (num_samples, num_classes)
        targets: (num_samples,)
        Returns dict of metrics
        """
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "f1": f1_score(targets, preds, average=average),
            "precision": precision_score(targets, preds, average=average, zero_division=0),
            "recall": recall_score(targets, preds, average=average, zero_division=0)
        }
        return metrics

    @staticmethod
    def batch_metrics(model, data_loader, device=torch.device("cpu"), average="macro"):
        """
        Evaluate model over a dataset
        """
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                all_logits.append(logits)
                all_labels.append(y)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return Metrics.compute_metrics(all_logits, all_labels, average=average)
