"""

This module provides:
  1. BioDataset       — PyTorch Dataset wrapper around HDF5 pipeline output
  2. CNN1DBaseline    — Convolutional baseline (spatial / morphological)
  3. LSTMBaseline     — Recurrent baseline (temporal / sequential)
  4. BaselineTrainer  — Training loop with early stopping, LR scheduling,
                        gradient clipping and metric logging
  5. evaluate()       — Full evaluation: accuracy, F1, confusion matrix,
                        per-class report, inference latency
  6. run_baselines()  — End-to-end entry point for all modalities

Inputs  : HDF5 files produced by preprocessing_pipeline.py
Outputs : Saved model checkpoints (.pt), metrics CSVs, confusion-matrix PNGs

Dependencies:
  pip install torch scikit-learn matplotlib seaborn h5py tqdm

"""

# Standard library
import os
import json
import time
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# 1. CONFIGURATION

@dataclass
class TrainingConfig:
    """All hyperparameters for baseline training."""

    # Paths
    processed_dir: str = "data/processed"
    output_dir:    str = "outputs/baselines"

    # Data
    modalities: List[str] = field(
        default_factory=lambda: ["ecg", "eeg", "ppg"]
    )
    batch_size:  int = 64
    num_workers: int = 4

    # Training
    epochs:        int   = 100
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    grad_clip:     float = 1.0  # max gradient norm
    patience:      int   = 15   # early stopping patience
    lr_patience:   int   = 7    # ReduceLROnPlateau patience
    lr_factor:     float = 0.5  # LR reduction factor
    min_lr:        float = 1e-6

    # CNN architecture
    cnn_filters:     List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int       = 7
    cnn_pool_size:   int       = 2
    cnn_dropout:     float     = 0.3

    # LSTM architecture
    lstm_hidden:  int   = 128
    lstm_layers:  int   = 2
    lstm_dropout: float = 0.3
    lstm_bidir:   bool  = True

    # Shared head
    fc_hidden:  int   = 128
    fc_dropout: float = 0.4
    seed:       int   = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = TrainingConfig()


# Minimal config accepted by make_loaders — any object with these four
# attributes works (TrainingConfig, HybridConfig, or LoaderConfig).
@dataclass
class LoaderConfig:
    """Lightweight config for callers that only need DataLoaders."""
    batch_size:  int = 64
    num_workers: int = 4
    device:      str = "cuda" if torch.cuda.is_available() else "cpu"
    drop_last:   bool = True


# 2. REPRODUCIBILITY

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# 3. DATASET WRAPPER

class BioDataset(Dataset):
    """
    PyTorch Dataset that reads directly from the HDF5 files produced by
    preprocessing_pipeline.py.

    Parameters
    ----------
    h5_path   : path to <modality>.h5
    modality  : "ecg" | "eeg" | "ppg"
    split     : "train" | "val" | "test"
    augment   : lightweight train-time augmentation flag

    Returns (x, y) where:
      x : float32 tensor (C, T)
      y : long   tensor scalar
    """

    def __init__(
        self,
        h5_path:  str,
        modality: str,
        split:    str,
        augment:  bool = False,
    ):
        super().__init__()
        self.augment = augment

        with h5py.File(h5_path, "r") as f:
            grp        = f[modality][split]
            self.X     = grp["X"][:].astype(np.float32)   # (N, C, T)
            self.y     = grp["y"][:].astype(np.int64)
            self.label_map = json.loads(
                f[modality].attrs.get("label_map", "{}")
            )

        self.n_channels = self.X.shape[1]
        self.seq_len    = self.X.shape[2]
        self.n_classes  = int(self.y.max()) + 1

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()
        y = self.y[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Signal-space augmentations that preserve clinical morphology.
          - Gaussian noise injection  (σ ~ U[0, 0.02])
          - Random amplitude scaling  (α ~ U[0.9, 1.1])
          - Random time shift         (± 5 % of window)
        """
        if np.random.rand() < 0.5:
            x += np.random.normal(0, np.random.uniform(0, 0.02),
                                  x.shape).astype(np.float32)
        if np.random.rand() < 0.5:
            x *= np.random.uniform(0.9, 1.1)
        if np.random.rand() < 0.5:
            shift = np.random.randint(-int(0.05 * x.shape[-1]),
                                       int(0.05 * x.shape[-1]))
            x = np.roll(x, shift, axis=-1)
        return x


def make_loaders(
    h5_path:  str,
    modality: str,
    cfg       = None,       # accepts TrainingConfig, HybridConfig, LoaderConfig, or None
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders for one modality.

    cfg can be any object that exposes batch_size, num_workers, and device
    attributes (duck typing).  Passing None uses LoaderConfig defaults.
    """
    if cfg is None:
        cfg = LoaderConfig()
    # Safely read attributes with defaults so any partial config works
    batch_size  = getattr(cfg, "batch_size",  64)
    num_workers = getattr(cfg, "num_workers", 4)
    device      = getattr(cfg, "device",      "cpu")

    loaders = {}
    for split in ("train", "val", "test"):
        ds = BioDataset(h5_path, modality, split, augment=(split == "train"))
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = (device == "cuda"),
            drop_last   = (split == "train"),
        )
        log.info(
            "%s [%s]: %d samples | %d classes | C=%d T=%d",
            modality.upper(), split, len(ds), ds.n_classes,
            ds.n_channels, ds.seq_len,
        )
    return loaders


# 4. SHARED BUILDING BLOCKS

class ConvBlock(nn.Module):
    """
    Conv1d → BatchNorm → ReLU → MaxPool → Dropout

    Design notes:
    - BatchNorm before activation reduces internal covariate shift.
    - same-padding (padding = kernel // 2) preserves temporal resolution
      before pooling, so all pool layers compress at the same rate.
    - Depthwise-separable convolutions are reserved for the lightweight
      edge-deployment model; the baseline intentionally uses standard
      convolutions for a clean comparison.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        pool_size:    int,
        dropout:      float,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FCHead(nn.Module):
    """Shared fully-connected classification head used by both baselines."""

    def __init__(
        self,
        in_features: int,
        hidden:      int,
        n_classes:   int,
        dropout:     float,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# 5. CNN BASELINE

class CNN1DBaseline(nn.Module):
    """
    1-D Convolutional Neural Network for biomedical signal classification.

    Architecture
    ------------
    Input  (B, C, T)
      │
      ├─ ConvBlock(32,  k=7, pool=2)   low-level temporal edges
      ├─ ConvBlock(64,  k=7, pool=2)   mid-level waveform shapes
      └─ ConvBlock(128, k=7, pool=2)   high-level class-discriminative patterns
      │
      ├─ Global Average Pooling        (B, 128)
      │    replaces Flatten to reduce params and improve generalisation
      │
      └─ FC Head (128 → n_classes)

    Academic motivation
    -------------------
    CNNs excel at detecting local morphological patterns in short signal
    windows: P-wave / QRS complex in ECG, alpha-burst bursts in EEG,
    systolic peak in PPG. The three-block hierarchy mirrors classical
    feature hierarchies used in biomedical time-series literature [2, 3].

    Limitation (discuss in thesis §5.1)
    -----------------------------------
    After three pooling layers (factor 8× compression) long-range temporal
    context is lost. A window of 1000 samples is reduced to ~125 timesteps
    before GAP collapses all remaining context. This motivates the LSTM
    baseline and, ultimately, the hybrid architecture.
    """

    model_type = "CNN"

    def __init__(
        self,
        in_channels: int,
        n_classes:   int,
        cfg:         TrainingConfig = CFG,
    ):
        super().__init__()
        layers = []
        ch_in  = in_channels
        for ch_out in cfg.cnn_filters:
            layers.append(ConvBlock(
                ch_in, ch_out,
                cfg.cnn_kernel_size,
                cfg.cnn_pool_size,
                cfg.cnn_dropout,
            ))
            ch_in = ch_out
        self.conv_blocks     = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_head = FCHead(
            cfg.cnn_filters[-1], cfg.fc_hidden, n_classes, cfg.fc_dropout
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)     # (B, 128, T')
        x = self.global_avg_pool(x) # (B, 128, 1)
        x = x.squeeze(-1)           # (B, 128)
        return self.fc_head(x)      # (B, n_classes)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# 6. LSTM BASELINE

class LSTMBaseline(nn.Module):
    """
    Bidirectional LSTM for biomedical signal classification.

    Architecture
    ------------
    Input  (B, C, T)
      │
      ├─ Transpose → (B, T, C)           raw channels become input features
      │
      ├─ BiLSTM (hidden=128, layers=2)   temporal dependency modeling
      │    forward  pass captures causal dynamics
      │    backward pass captures anti-causal context (valid for offline windows)
      │
      ├─ Temporal Attention Pooling      scalar importance weight per timestep
      │    context = Σ_t  softmax(W·h_t) · h_t     (B, hidden*2)
      │
      └─ FC Head (256 → n_classes)

    Academic motivation
    -------------------
    LSTMs capture sequential dynamics that CNNs miss after pooling:
    RR-interval variability in ECG, event-related (de)synchronisation
    in EEG, and BVP waveform trends across PPG activity transitions.
    Bidirectionality is appropriate because classification is performed
    on complete offline windows rather than in real time.

    Temporal Attention
    ------------------
    A single-layer additive attention head assigns scalar importance
    weights to each timestep. This provides a lightweight interpretability
    signal (complement to LRP in the XAI module) and outperforms naive
    last-state or mean pooling in preliminary experiments.

    Limitation (discuss in thesis §5.2)
    -----------------------------------
    Raw channel values are fed as input features, so subtle spectral
    patterns (e.g., ECG QRS frequency content) must be discovered
    implicitly by the LSTM. CNNs handle this more efficiently. This
    motivates the CNN-LSTM hybrid which combines both strengths.
    """

    model_type = "LSTM"

    def __init__(
        self,
        in_channels: int,
        n_classes:   int,
        seq_len:     int,
        cfg:         TrainingConfig = CFG,
    ):
        super().__init__()
        self.bidir       = cfg.lstm_bidir
        hidden_dir       = cfg.lstm_hidden * (2 if cfg.lstm_bidir else 1)

        self.lstm = nn.LSTM(
            input_size    = in_channels,
            hidden_size   = cfg.lstm_hidden,
            num_layers    = cfg.lstm_layers,
            batch_first   = True,
            dropout       = cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional = cfg.lstm_bidir,
        )
        self.attn_fc = nn.Linear(hidden_dir, 1, bias=False)
        self.fc_head = FCHead(
            hidden_dir, cfg.fc_hidden, n_classes, cfg.fc_dropout
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.permute(0, 2, 1)                         # (B, T, C)
        out, _ = self.lstm(x)                            # (B, T, H)
        attn_w = torch.softmax(self.attn_fc(out), dim=1) # (B, T, 1)
        ctx    = (out * attn_w).sum(dim=1)               # (B, H)
        return self.fc_head(ctx)

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, attention_weights) for interpretability analysis."""
        x      = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn_fc(out), dim=1).squeeze(-1)  # (B, T)
        ctx    = (out * attn_w.unsqueeze(-1)).sum(dim=1)
        return self.fc_head(ctx), attn_w

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)   # forget-gate bias = 1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# 7. MODEL FACTORY

def build_model(
    model_type:  str,
    in_channels: int,
    n_classes:   int,
    seq_len:     int,
    cfg:         TrainingConfig = CFG,
) -> nn.Module:
    if model_type.upper() == "CNN":
        model = CNN1DBaseline(in_channels, n_classes, cfg)
    elif model_type.upper() == "LSTM":
        model = LSTMBaseline(in_channels, n_classes, seq_len, cfg)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    model = model.to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s — Trainable parameters: %s", model_type, f"{n_params:,}")
    return model


# 8. CLASS-WEIGHTED LOSS

def compute_class_weights(
    y_train: np.ndarray,
    device:  str,
) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss.

    Biomedical datasets are almost always imbalanced (MIT-BIH has ~90%
    Normal beats vs ~0.6% Fusion beats). Weighting prevents the model
    from collapsing to the majority class prediction.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# 9. TRAINER

@dataclass
class EpochMetrics:
    loss: float
    acc:  float
    f1:   float


class BaselineTrainer:
    """
    Training loop with:
      - Weighted cross-entropy loss        (handles class imbalance)
      - Adam optimiser with weight decay
      - ReduceLROnPlateau on val macro-F1  (monitored metric)
      - Gradient clipping at cfg.grad_clip (critical for LSTM stability)
      - Early stopping on val macro-F1
      - Best-model checkpointing
      - Training history saved as CSV + PNG
    """

    def __init__(
        self,
        model:      nn.Module,
        loaders:    Dict[str, DataLoader],
        n_classes:  int,
        output_dir: str,
        cfg:        TrainingConfig = CFG,
    ):
        self.model      = model
        self.loaders    = loaders
        self.cfg        = cfg
        self.output_dir = output_dir
        self.device     = cfg.device
        self.n_classes  = n_classes

        y_train = loaders["train"].dataset.y
        weights = compute_class_weights(y_train, self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=cfg.lr_factor,
            patience=cfg.lr_patience, min_lr=cfg.min_lr,
        )

        self.history:     List[Dict] = []
        self.best_val_f1: float      = -1.0
        self.best_state:  dict       = {}
        self.no_improve:  int        = 0

    # Single epoch

    def _run_epoch(self, split: str) -> EpochMetrics:
        training = (split == "train")
        self.model.train(training)
        loader   = self.loaders[split]

        total_loss  = 0.0
        all_preds:  List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss   = self.criterion(logits, y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.optimizer.step()

                total_loss += loss.item() * len(y)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        avg    = "macro" if self.n_classes > 2 else "binary"

        return EpochMetrics(
            loss = total_loss / len(y_true),
            acc  = accuracy_score(y_true, y_pred),
            f1   = f1_score(y_true, y_pred, average=avg, zero_division=0),
        )

    # Full training loop

    def fit(self) -> pd.DataFrame:
        log.info(
            "Training %s | %d epochs | device=%s",
            self.model.model_type, self.cfg.epochs, self.device,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            t0      = time.time()
            train_m = self._run_epoch("train")
            val_m   = self._run_epoch("val")
            self.scheduler.step(val_m.f1)
            elapsed = time.time() - t0

            row = {
                "epoch":      epoch,
                "train_loss": train_m.loss, "train_acc": train_m.acc,
                "train_f1":   train_m.f1,
                "val_loss":   val_m.loss,   "val_acc":   val_m.acc,
                "val_f1":     val_m.f1,
                "lr":         self.optimizer.param_groups[0]["lr"],
                "elapsed_s":  round(elapsed, 2),
            }
            self.history.append(row)

            log.info(
                "Ep %3d/%d | train loss=%.4f acc=%.3f f1=%.3f | "
                "val loss=%.4f acc=%.3f f1=%.3f | lr=%.2e | %.1fs",
                epoch, self.cfg.epochs,
                train_m.loss, train_m.acc, train_m.f1,
                val_m.loss,   val_m.acc,   val_m.f1,
                self.optimizer.param_groups[0]["lr"], elapsed,
            )

            if val_m.f1 > self.best_val_f1:
                self.best_val_f1 = val_m.f1
                self.best_state  = deepcopy(self.model.state_dict())
                self.no_improve  = 0
                self._save_checkpoint(epoch, val_m.f1)
            else:
                self.no_improve += 1
                if self.no_improve >= self.cfg.patience:
                    log.info(
                        "Early stopping at epoch %d (%d epochs without "
                        "val F1 improvement)", epoch, self.cfg.patience,
                    )
                    break

        self.model.load_state_dict(self.best_state)
        df = pd.DataFrame(self.history)
        df.to_csv(
            os.path.join(self.output_dir, "training_history.csv"), index=False
        )
        self._plot_history(df)
        return df

    def _save_checkpoint(self, epoch: int, val_f1: float) -> None:
        path = os.path.join(
            self.output_dir,
            f"best_{self.model.model_type.lower()}.pt",
        )
        torch.save({
            "epoch":       epoch,
            "val_f1":      val_f1,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, path)

    def _plot_history(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, metric in zip(axes, ["loss", "acc", "f1"]):
            ax.plot(df["epoch"], df[f"train_{metric}"], label="train")
            ax.plot(df["epoch"], df[f"val_{metric}"],   label="val")
            ax.set_xlabel("Epoch")
            ax.set_title(metric.capitalize())
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"{self.model.model_type} — Training History", fontsize=13
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "training_history.png"), dpi=150
        )
        plt.close()


# 10. EVALUATION

def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    label_map:  Dict[str, str],
    output_dir: str,
    cfg:        TrainingConfig = CFG,
) -> Dict:
    """
    Full evaluation on the test set.

    Computes and saves:
      - Accuracy, macro-F1, weighted-F1
      - Per-class precision / recall / F1
      - Confusion matrix (raw counts + normalised heatmap)
      - Inference latency (ms / sample)

    Returns a metrics summary dict.
    """
    model.eval()
    all_preds:    List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []
    total_time    = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(cfg.device)
            t0 = time.perf_counter()
            logits = model(X)
            total_time    += time.perf_counter() - t0
            total_samples += len(y)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    n_classes   = int(y_true.max()) + 1
    class_names = [label_map.get(str(i), str(i)) for i in range(n_classes)]

    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    lat_ms = (total_time / total_samples) * 1000

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    log.info("\n%s", report)
    log.info(
        "Accuracy=%.4f | Macro-F1=%.4f | Weighted-F1=%.4f | "
        "Latency=%.3f ms/sample",
        acc, f1_mac, f1_wt, lat_ms,
    )

    os.makedirs(output_dir, exist_ok=True)
    _plot_confusion_matrix(y_true, y_pred, class_names, output_dir)

    metrics = {
        "accuracy":              round(acc,    4),
        "macro_f1":              round(f1_mac, 4),
        "weighted_f1":           round(f1_wt,  4),
        "latency_ms_per_sample": round(lat_ms, 4),
    }
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as fh:
        fh.write(report)

    return metrics


def _plot_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: List[str],
    output_dir:  str,
) -> None:
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


# 11. COMPARISON TABLE

def build_comparison_table(
    results:    Dict[str, Dict[str, Dict]],
    output_dir: str,
) -> pd.DataFrame:
    """
    Aggregate CNN vs LSTM results across modalities into one CSV.

    results format:
      results[modality][model_type] = metrics_dict
    """
    rows = []
    for modality, models in results.items():
        for model_type, metrics in models.items():
            rows.append({
                "Modality":     modality.upper(),
                "Model":        model_type,
                "Accuracy":     metrics["accuracy"],
                "Macro F1":     metrics["macro_f1"],
                "Weighted F1":  metrics["weighted_f1"],
                "Latency (ms)": metrics["latency_ms_per_sample"],
            })
    df = pd.DataFrame(rows).sort_values(["Modality", "Model"])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison_table.csv")
    df.to_csv(path, index=False)
    log.info("Comparison table → %s\n%s", path, df.to_string(index=False))
    return df


# 12. MAIN ENTRY POINT

def run_baselines(cfg: TrainingConfig = CFG) -> None:
    """
    Train and evaluate CNN and LSTM baselines on all three modalities.

    Per modality:
      1. Build DataLoaders from HDF5
      2. Train CNN  → checkpoint + history CSV/PNG
      3. Train LSTM → checkpoint + history CSV/PNG
      4. Evaluate both on test set → metrics JSON + confusion matrix PNG
    Finally: cross-modality comparison table CSV.
    """
    set_seed(cfg.seed)
    log.info("=" * 60)
    log.info("Baseline Model Training — CNN & LSTM")
    log.info("=" * 60)
    log.info("Device: %s", cfg.device)

    all_results: Dict[str, Dict[str, Dict]] = {}

    for modality in cfg.modalities:
        h5_path = os.path.join(cfg.processed_dir, f"{modality}.h5")
        if not os.path.exists(h5_path):
            log.warning(
                "HDF5 not found for %s at %s — skipping.", modality, h5_path
            )
            continue

        log.info("\n%s\n── Modality: %s\n%s",
                 "=" * 60, modality.upper(), "=" * 60)
        loaders   = make_loaders(h5_path, modality, cfg)
        train_ds  = loaders["train"].dataset
        n_ch      = train_ds.n_channels
        seq_len   = train_ds.seq_len
        n_classes = train_ds.n_classes
        label_map = train_ds.label_map

        all_results[modality] = {}

        for model_type in ("CNN", "LSTM"):
            log.info("\n── Model: %s ──", model_type)
            out_dir = os.path.join(
                cfg.output_dir, modality, model_type.lower()
            )
            model   = build_model(
                model_type, n_ch, n_classes, seq_len, cfg
            )
            trainer = BaselineTrainer(
                model=model, loaders=loaders, n_classes=n_classes,
                output_dir=out_dir, cfg=cfg,
            )
            trainer.fit()
            metrics = evaluate(
                model, loaders["test"], label_map, out_dir, cfg
            )
            all_results[modality][model_type] = metrics

    build_comparison_table(all_results, cfg.output_dir)
    log.info("\nAll baselines complete. Outputs → %s", cfg.output_dir)



# USAGE
# 
#
# Run with defaults:
#   python baseline_models.py
#
# Custom config:
#   from baseline_models import run_baselines, TrainingConfig
#   cfg = TrainingConfig(
#       processed_dir = "/path/to/processed",
#       modalities    = ["ecg"],
#       epochs        = 50,
#       batch_size    = 128,
#   )
#   run_baselines(cfg)
#
# Load a checkpoint:
#   ckpt = torch.load("outputs/baselines/ecg/cnn/best_cnn.pt")
#   model.load_state_dict(ckpt["model_state"])

if __name__ == "__main__":
    run_baselines(CFG)
