"""
This module provides:
  1. NormalizationConfig        — controls preprocessing vs model-level norm
  2. AblationConfig             — feature flags for every architectural component
  3. AblationHybrid             — HybridCNNLSTM variant driven by AblationConfig
  4. AblationRunner             — runs all ablation experiments, logs results
  5. run_ablations()            — entry point, produces ablation_results.csv

Ablation matrix (16 experiments × 3 modalities = 48 runs):
────────────────────────────────────────────────────────────
Row  Preprocessing  Model Norm    LSTM   MS-CNN   SE    Gated-Fusion
A0   z-score        none          ✓      ✓        ✓     ✓       (preprocessing norm only)
A1   none           learnable     ✓      ✓        ✓     ✓       (proposed, no preprocessing)
A2   none           layer         ✓      ✓        ✓     ✓       (LayerNorm ablation)
A3   none           instance      ✓      ✓        ✓     ✓       (InstanceNorm ablation)
A4   z-score        none          ✗      ✓        ✓     ✓       (CNN only, no LSTM)
A5   z-score        none          ✓      ✗        ✓     ✓       (LSTM only, no MS-CNN)
A6   z-score        none          ✓      ✓        ✗     ✓       (no SE attention)
A7   z-score        none          ✓      ✓        ✓     ✗       (no gated fusion, concat)
A8   z-score        none          ✓      single   ✓     ✓       (single-scale CNN k=7)
A9   none           learnable     ✓      ✓        ✓     ✓       (full proposed system)

A0 vs A1 → proves LearnableNorm contributes independently of preprocessing.
A1 vs A2/A3 → chooses best norm variant.
A4 → CNN-only baseline (same as baseline_models.py CNN for cross-check).
A5 → LSTM-only baseline (same as baseline_models.py LSTM for cross-check).
A6 → isolates SE channel attention contribution.
A7 → isolates gated fusion vs simple concatenation.
A8 → proves multi-scale CNN outperforms single-scale.
A9 → full system (best expected result).
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# 1. NORMALIZATION CONFIGURATION

@dataclass
class NormalizationConfig:
    """
    Separates preprocessing-level and model-level normalization.

    preprocessing_norm : "zscore" | "minmax" | "robust" | "none"
        Applied offline in preprocessing_pipeline.py before training.
        "none" passes raw (filtered) signals to the model.

    model_norm : "none" | "learnable" | "layer" | "instance"
        Applied inside the model as the first operation.
        "none" means the model trusts the preprocessing normalization.

    Recommended experimental groups:
        Control:  preprocessing="zscore",  model="none"
        Proposed: preprocessing="none",    model="learnable"
        Ablation: preprocessing="none",    model="layer"
        Ablation: preprocessing="none",    model="instance"
    """
    preprocessing_norm: str = "zscore"   # applied in preprocessing_pipeline.py
    model_norm:         str = "none"     # applied inside AblationHybrid

    def __post_init__(self):
        valid_pre   = {"zscore", "minmax", "robust", "none"}
        valid_model = {"none", "learnable", "layer", "instance"}
        if self.preprocessing_norm not in valid_pre:
            raise ValueError(f"preprocessing_norm must be one of {valid_pre}")
        if self.model_norm not in valid_model:
            raise ValueError(f"model_norm must be one of {valid_model}")

    def validate_no_double_norm(self) -> None:
        """Warn if both preprocessing and model normalization are active."""
        if self.preprocessing_norm != "none" and self.model_norm != "none":
            log.warning(
                "DOUBLE NORMALIZATION DETECTED: preprocessing='%s' and "
                "model='%s' are both active. This makes it impossible to "
                "attribute performance gains to the model's norm layer. "
                "Consider setting one to 'none' for ablation experiments.",
                self.preprocessing_norm, self.model_norm,
            )


# 2. ABLATION CONFIG

@dataclass
class AblationConfig:
    """
    Feature flags for every architectural component in HybridCNNLSTM.

    Each flag maps to one ablation row in the matrix above.
    The full proposed system has all flags True / set to "learnable".
    """
    name: str = "full_hybrid"

    # Normalization (Gap #3)
    norm: NormalizationConfig = field(
        default_factory=lambda: NormalizationConfig("none", "learnable")
    )

    # Architectural components
    use_lstm:         bool = True    # False → CNN-only path
    use_multiscale:   bool = True    # False → single kernel k=7
    use_se:           bool = True    # False → remove SE blocks
    use_gated_fusion: bool = True    # False → simple concat + Linear

    # CNN settings
    cnn_kernels:      List[int] = field(default_factory=lambda: [3, 7, 15, 31])
    cnn_base_filters: int       = 32
    cnn_depth:        int       = 3

    # LSTM settings
    lstm_hidden: int  = 192
    lstm_layers: int  = 2
    lstm_bidir:  bool = True
    attn_heads:  int  = 4

    # Head
    fusion_hidden: int   = 256
    fc_hidden:     int   = 128
    fc_dropout:    float = 0.4

    # Training
    epochs:        int   = 80
    learning_rate: float = 5e-4
    weight_decay:  float = 1e-4
    batch_size:    int   = 64
    patience:      int   = 15
    seed:          int   = 42
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"


def build_ablation_matrix() -> List[AblationConfig]:
    """
    Returns the 10 ablation configurations described in the module docstring.
    Each row is a different experimental condition.
    """
    configs = []

    # A0: preprocessing norm only (control baseline — no model norm)
    configs.append(AblationConfig(
        name="A0_preprocess_norm_only",
        norm=NormalizationConfig("zscore", "none"),
    ))

    # A1: proposed — LearnableNorm, no preprocessing norm
    configs.append(AblationConfig(
        name="A1_learnable_norm",
        norm=NormalizationConfig("none", "learnable"),
    ))

    # A2: LayerNorm ablation
    configs.append(AblationConfig(
        name="A2_layer_norm",
        norm=NormalizationConfig("none", "layer"),
    ))

    # A3: InstanceNorm ablation
    configs.append(AblationConfig(
        name="A3_instance_norm",
        norm=NormalizationConfig("none", "instance"),
    ))

    # A4: CNN-only (no LSTM)
    configs.append(AblationConfig(
        name="A4_cnn_only",
        norm=NormalizationConfig("zscore", "none"),
        use_lstm=False,
    ))

    # A5: LSTM-only (no multi-scale CNN — single projection layer instead)
    configs.append(AblationConfig(
        name="A5_lstm_only",
        norm=NormalizationConfig("zscore", "none"),
        use_multiscale=False,
        cnn_kernels=[7],
        cnn_depth=0,
    ))

    # A6: no SE channel attention
    configs.append(AblationConfig(
        name="A6_no_se",
        norm=NormalizationConfig("zscore", "none"),
        use_se=False,
    ))

    # A7: no gated fusion (simple concatenation)
    configs.append(AblationConfig(
        name="A7_no_gated_fusion",
        norm=NormalizationConfig("zscore", "none"),
        use_gated_fusion=False,
    ))

    # A8: single-scale CNN (k=7 only)
    configs.append(AblationConfig(
        name="A8_single_scale_cnn",
        norm=NormalizationConfig("zscore", "none"),
        use_multiscale=False,
        cnn_kernels=[7],
    ))

    # A9: full proposed system (should be best)
    configs.append(AblationConfig(
        name="A9_full_proposed",
        norm=NormalizationConfig("none", "learnable"),
        use_lstm=True,
        use_multiscale=True,
        use_se=True,
        use_gated_fusion=True,
    ))

    return configs


# 3. ABLATION-DRIVEN MODEL

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dropout, use_se, se_ratio=4):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        self.se = None
        if use_se:
            reduced = max(out_ch // se_ratio, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(out_ch, reduced, bias=False), nn.ReLU(inplace=True),
                nn.Linear(reduced, out_ch, bias=False), nn.Sigmoid(),
            )

    def forward(self, x):
        res = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            scale = self.se(out).unsqueeze(2)
            out   = out * scale
        return F.relu(out + res, inplace=True)


def _build_model_norm(method: str, n_ch: int) -> Optional[nn.Module]:
    if method == "none":
        return None
    elif method == "learnable":
        from hybrid_model import LearnableNorm
        return LearnableNorm(n_ch)
    elif method == "layer":
        from hybrid_model import AdaptiveLayerNorm
        return AdaptiveLayerNorm(n_ch)
    elif method == "instance":
        return nn.InstanceNorm1d(n_ch, affine=True)
    raise ValueError(f"Unknown model_norm: {method}")


class AblationHybrid(nn.Module):
    """
    HybridCNNLSTM controlled entirely by AblationConfig feature flags.

    Enables any combination of:
        - Model-level adaptive norm vs none
        - Multi-scale CNN vs single kernel
        - SE channel attention vs none
        - Bidirectional LSTM vs CNN-only path
        - Gated fusion vs simple concatenation
    """

    def __init__(self, in_channels: int, n_classes: int, cfg: AblationConfig):
        super().__init__()
        self.cfg       = cfg
        self.use_lstm  = cfg.use_lstm
        self.use_gated = cfg.use_gated_fusion
        T_OUT = 64

        kernels    = cfg.cnn_kernels if cfg.use_multiscale else [7]
        n_branches = len(kernels)
        F_base     = cfg.cnn_base_filters
        cnn_out_ch = F_base * n_branches

        # Input normalization
        cfg.norm.validate_no_double_norm()
        self.input_norm = _build_model_norm(cfg.norm.model_norm, in_channels)

        # CNN branches
        self.branches = nn.ModuleList()
        for k in kernels:
            layers = [nn.Sequential(
                nn.Conv1d(in_channels, F_base, k, padding=k//2, bias=False),
                nn.BatchNorm1d(F_base), nn.ReLU(inplace=True),
            )]
            for _ in range(cfg.cnn_depth):
                layers.append(_ConvBlock(F_base, F_base, k, 0.2, cfg.use_se))
            self.branches.append(nn.Sequential(*layers))

        self.cnn_pool     = nn.AdaptiveAvgPool1d(T_OUT)
        self.cnn_gap      = nn.AdaptiveAvgPool1d(1)

        # LSTM (optional)
        self.lstm      = None
        self.attn_fc   = None
        lstm_out_dim   = 0

        if cfg.use_lstm:
            hidden_dir = cfg.lstm_hidden * (2 if cfg.lstm_bidir else 1)
            self.lstm  = nn.LSTM(
                input_size=cnn_out_ch, hidden_size=cfg.lstm_hidden,
                num_layers=cfg.lstm_layers, batch_first=True,
                dropout=0.3 if cfg.lstm_layers > 1 else 0.0,
                bidirectional=cfg.lstm_bidir,
            )
            self.lstm_norm = nn.LayerNorm(hidden_dir)
            self.attn_fc   = nn.Linear(hidden_dir, 1, bias=False)
            lstm_out_dim   = hidden_dir

        # Fusion
        if cfg.use_lstm and cfg.use_gated_fusion:
            from hybrid_model import GatedFusion
            self.fusion = GatedFusion(
                d_cnn=cnn_out_ch, d_lstm=lstm_out_dim,
                fusion_hidden=cfg.fusion_hidden, dropout=0.4,
            )
            head_in = cfg.fusion_hidden

        elif cfg.use_lstm:
            # Simple concatenation (no gate)
            self.fusion   = None
            self.concat_proj = nn.Sequential(
                nn.Linear(cnn_out_ch + lstm_out_dim, cfg.fusion_hidden),
                nn.ReLU(inplace=True),
            )
            head_in = cfg.fusion_hidden

        else:
            # CNN-only: no fusion needed
            self.fusion      = None
            self.concat_proj = None
            head_in          = cnn_out_ch

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.fc_hidden),
            nn.BatchNorm1d(cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.fc_dropout),
            nn.Linear(cfg.fc_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input norm (optional)
        if self.input_norm is not None:
            x = self.input_norm(x)

        # Multi-scale CNN
        branch_outs = [b(x) for b in self.branches]
        cnn_seq     = torch.cat(branch_outs, dim=1)     # (B, C_cnn, T)
        cnn_seq     = self.cnn_pool(cnn_seq)            # (B, C_cnn, T_out)
        f_cnn       = self.cnn_gap(cnn_seq).squeeze(-1) # (B, C_cnn)

        if not self.use_lstm:
            return self.head(f_cnn)

        # LSTM
        lstm_in     = cnn_seq.permute(0, 2, 1)                     # (B, T_out, C_cnn)
        lstm_out, _ = self.lstm(lstm_in)                           # (B, T_out, H)
        lstm_out    = self.lstm_norm(lstm_out)
        attn_w      = torch.softmax(self.attn_fc(lstm_out), dim=1) # (B, T_out, 1)
        f_lstm      = (lstm_out * attn_w).sum(dim=1)               # (B, H)

        # Fusion
        if self.fusion is not None:
            fused = self.fusion(f_cnn, f_lstm)
        else:
            fused = self.concat_proj(torch.cat([f_cnn, f_lstm], dim=1))

        return self.head(fused)


# 4. ABLATION RUNNER

class AblationRunner:
    """
    Runs all ablation experiments for one modality and records results.

    For each AblationConfig:
      1. Instantiates AblationHybrid with feature flags from config
      2. Trains with early stopping
      3. Evaluates on test set
      4. Records mean ± std across n_runs random seeds

    Parameters
    ----------
    loaders   : train/val/test DataLoaders (from make_loaders)
    n_classes : int
    n_runs    : number of independent training runs per config (for stats)
    output_dir: where to save results
    """

    def __init__(
        self,
        loaders:    Dict[str, DataLoader],
        n_classes:  int,
        in_channels: int,
        output_dir: str,
        n_runs:     int = 3,
    ):
        self.loaders     = loaders
        self.n_classes   = n_classes
        self.in_channels = in_channels
        self.output_dir  = output_dir
        self.n_runs      = n_runs

    def _train_and_eval(
        self,
        model:  AblationHybrid,
        cfg:    AblationConfig,
        seed:   int,
    ) -> Dict:
        """Single training run; returns test metrics dict."""
        import torch.optim as optim
        from sklearn.metrics import accuracy_score, f1_score

        torch.manual_seed(seed)
        np.random.seed(seed)

        device    = cfg.device
        model     = model.to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )

        y_train  = self.loaders["train"].dataset.y
        classes, counts = np.unique(y_train, return_counts=True)
        weights  = torch.tensor(
            1.0 / counts.astype(np.float32), dtype=torch.float32
        ).to(device)
        weights /= weights.sum() / len(classes)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

        best_val_f1  = -1.0
        best_state   = {}
        no_improve   = 0

        for epoch in range(1, cfg.epochs + 1):
            # Train
            model.train()
            for X, y in self.loaders["train"]:
                X, y = X.to(device), y.to(device)
                loss = criterion(model(X), y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Val
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for X, y in self.loaders["val"]:
                    preds.append(model(X.to(device)).argmax(1).cpu().numpy())
                    labels.append(y.numpy())
            val_f1 = f1_score(
                np.concatenate(labels), np.concatenate(preds),
                average="macro", zero_division=0,
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state  = deepcopy(model.state_dict())
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        # Test
        t_start = time.perf_counter()
        preds, labels = [], []
        n_samples = 0
        with torch.no_grad():
            for X, y in self.loaders["test"]:
                preds.append(model(X.to(device)).argmax(1).cpu().numpy())
                labels.append(y.numpy())
                n_samples += len(y)
        lat_ms = (time.perf_counter() - t_start) / n_samples * 1000

        y_true = np.concatenate(labels)
        y_pred = np.concatenate(preds)
        return {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "macro_f1":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "latency_ms":  float(lat_ms),
            "n_params":    sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    def run(self, ablation_configs: List[AblationConfig]) -> pd.DataFrame:
        """
        Run all ablation configs for this modality.
        Each config is run n_runs times with different seeds.
        Returns DataFrame with mean ± std for each metric.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        rows = []

        for cfg in ablation_configs:
            log.info("\n── Ablation: %s ──", cfg.name)
            cfg.norm.validate_no_double_norm()

            run_metrics = []
            for run_idx in range(self.n_runs):
                seed  = cfg.seed + run_idx * 100
                model = AblationHybrid(self.in_channels, self.n_classes, cfg)
                m     = self._train_and_eval(model, cfg, seed)
                run_metrics.append(m)
                log.info(
                    "  Run %d/%d | acc=%.4f | macro_f1=%.4f",
                    run_idx + 1, self.n_runs, m["accuracy"], m["macro_f1"],
                )

            # Aggregate
            agg = {}
            for k in run_metrics[0]:
                vals = [rm[k] for rm in run_metrics]
                agg[f"{k}_mean"] = np.mean(vals)
                agg[f"{k}_std"]  = np.std(vals)

            row = {
                "ablation":          cfg.name,
                "preprocess_norm":   cfg.norm.preprocessing_norm,
                "model_norm":        cfg.norm.model_norm,
                "use_lstm":          cfg.use_lstm,
                "use_multiscale":    cfg.use_multiscale,
                "use_se":            cfg.use_se,
                "use_gated_fusion":  cfg.use_gated_fusion,
                **agg,
            }
            rows.append(row)
            log.info(
                "  → acc=%.4f±%.4f | macro_f1=%.4f±%.4f",
                agg["accuracy_mean"], agg["accuracy_std"],
                agg["macro_f1_mean"], agg["macro_f1_std"],
            )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, "ablation_results.csv"), index=False)
        self._plot_ablation(df)
        log.info("Ablation results → %s", self.output_dir)
        return df

    def _plot_ablation(self, df: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        metrics  = ["accuracy_mean", "macro_f1_mean", "weighted_f1_mean"]
        stds     = ["accuracy_std",  "macro_f1_std",  "weighted_f1_std"]
        labels_m = ["Accuracy", "Macro F1", "Weighted F1"]
        names    = df["ablation"].tolist()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

        for ax, metric, std_col, label in zip(axes, metrics, stds, labels_m):
            vals    = df[metric].values
            err     = df[std_col].values
            bars    = ax.barh(names, vals, xerr=err, color=colors,
                              ecolor="black", capsize=3, height=0.6)
            ax.set_xlabel(label)
            ax.set_title(label)
            ax.set_xlim(max(0, vals.min() - 0.1), min(1.0, vals.max() + 0.1))
            # Highlight best
            best_idx = int(np.argmax(vals))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.5)
            ax.grid(axis="x", alpha=0.3)

        fig.suptitle("Ablation Study Results (mean ± std)", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "ablation_results.png"), dpi=150
        )
        plt.close()


# 5. ENTRY POINT

def run_ablations(
    modality:    str,
    h5_path:     str,
    output_dir:  str,
    n_runs:      int = 3,
    device:      str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    """
    Run full ablation study for one modality.

    Usage:
        df = run_ablations("ecg", "data/processed/ecg.h5",
                           "outputs/ablations/ecg", n_runs=3)
    """
    from baseline_models import make_loaders, LoaderConfig

    loader_cfg = LoaderConfig(batch_size=64, num_workers=4, device=device)
    loaders    = make_loaders(h5_path, modality, loader_cfg)
    train_ds   = loaders["train"].dataset
    n_ch       = train_ds.n_channels
    n_classes  = train_ds.n_classes

    configs = build_ablation_matrix()
    for c in configs:
        c.device = device

    runner = AblationRunner(
        loaders=loaders, n_classes=n_classes, in_channels=n_ch,
        output_dir=output_dir, n_runs=n_runs,
    )
    return runner.run(configs)


# USAGE

# from ablation_framework import run_ablations
# df = run_ablations("ecg", "data/processed/ecg.h5", "outputs/ablations/ecg")
