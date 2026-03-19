import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import time
import numpy as np

class Trainer:
    """
    General-purpose trainer for classification models with attention support.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                 scheduler=None, num_epochs: int = 50, log_interval: int = 10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.log_interval = log_interval

    def train_epoch(self):
        self.model.train()
        losses = []
        all_preds, all_labels = [], []
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            # Handle models with attention output
            out = self.model(x)
            if isinstance(out, tuple):
                logits, attn_weights = out
                loss = self.criterion(logits, y, attn_weights)
            else:
                logits = out
                loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.detach().cpu().numpy())
            if batch_idx % self.log_interval == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return np.mean(losses), acc, f1

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        losses = []
        all_preds, all_labels = [], []
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model(x)
            if isinstance(out, tuple):
                logits, attn_weights = out
                loss = self.criterion(logits, y, attn_weights)
            else:
                logits = out
                loss = self.criterion(logits, y)
            losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.detach().cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return np.mean(losses), acc, f1

    def fit(self):
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate_epoch()
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
                  f"Time: {elapsed:.1f}s")
