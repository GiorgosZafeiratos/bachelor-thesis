import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ConfusionMatrix:
    """
    Compute and plot confusion matrices.
    """
    @staticmethod
    def compute(model, data_loader, device=torch.device("cpu")):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        cm = confusion_matrix(all_labels, all_preds)
        return cm

    @staticmethod
    def plot(cm: np.ndarray, class_names: list = None, figsize=(8, 6), cmap="Blues"):
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.show()
