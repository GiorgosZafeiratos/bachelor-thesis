"""

This module provides:
  1. Adaptive Normalization layers
       - LearnableNorm     : input-conditioned γ/β via a small MLP
       - AdaptiveLayerNorm : drop-in LayerNorm with learnable affine params
  2. Multi-Scale CNN Encoder
       - Parallel branches at kernel sizes [3, 7, 15, 31]
       - Residual blocks with squeeze-and-excitation channel attention
  3. Temporal LSTM Encoder
       - Stacked bidirectional LSTM
       - Multi-head self-attention over time (replaces single-head from baseline)
  4. HybridCNNLSTM
       - Combines all of the above with a gated feature fusion layer
  5. HybridTrainer
       - Cosine annealing with warm restarts
       - Mixup augmentation
       - Label smoothing
       - Everything from BaselineTrainer (early stopping, grad clipping …)
  6. run_hybrid()
       - End-to-end training + evaluation + comparison against baselines

Inputs  : HDF5 files from preprocessing_pipeline.py
          (Optional) baseline metrics JSONs from baseline_models.py
Outputs : Checkpoints, history CSV/PNG, confusion matrix, final report

Dependencies:
  pip install torch scikit-learn matplotlib seaborn h5py tqdm

"""

# Standard library
import os
import json
import math
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
)

# Re-use dataset / evaluation helpers from baseline module
# (place baseline_models.py in the same directory)
from baseline_models import (
    BioDataset, make_loaders,
    compute_class_weights,
    _plot_confusion_matrix,
    set_seed,
    TrainingConfig as BaseConfig,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# 1. CONFIGURATION

@dataclass
class HybridConfig:
    """Hyperparameters for the hybrid model and its training."""

    # Paths
    processed_dir:  str = "data/processed"
    baseline_dir:   str = "outputs/baselines"
    output_dir:     str = "outputs/hybrid"

    modalities: List[str] = field(
        default_factory=lambda: ["ecg", "eeg", "ppg"]
    )
    batch_size:  int = 64
    num_workers: int = 4

    # Training
    epochs:        int   = 120
    learning_rate: float = 5e-4
    weight_decay:  float = 1e-4
    grad_clip:     float = 1.0
    patience:      int   = 20
    warmup_epochs: int   = 5       # linear LR warm-up
    seed:          int   = 42

    # Mixup
    mixup_alpha:   float = 0.2     # Beta distribution α; 0 disables mixup
    label_smoothing: float = 0.1   # CrossEntropyLoss label smoothing

    # Multi-scale CNN encoder
    cnn_kernels:    List[int] = field(
        default_factory=lambda: [3, 7, 15, 31]
    )                                # parallel branches
    cnn_base_filters: int     = 32   # filters per branch (× n_branches = total)
    cnn_depth:        int     = 3    # residual blocks per branch
    cnn_dropout:      float   = 0.2
    se_ratio:         int     = 4    # squeeze-and-excitation reduction ratio

    # Adaptive normalization
    # "learnable" → LearnableNorm (input-conditioned affine params via MLP)
    # "layer"     → AdaptiveLayerNorm (standard learnable LayerNorm)
    # "instance"  → InstanceNorm1d with affine=True
    adaptive_norm: str = "learnable"

    # LSTM encoder
    lstm_hidden:  int   = 192
    lstm_layers:  int   = 2
    lstm_dropout: float = 0.3
    lstm_bidir:   bool  = True

    # Multi-head attention
    attn_heads:   int   = 4
    attn_dropout: float = 0.1

    # Gated fusion
    fusion_hidden: int   = 256
    fusion_dropout: float = 0.4

    # Classifier head
    fc_hidden:  int   = 128
    fc_dropout: float = 0.4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = HybridConfig()


# 2. ADAPTIVE NORMALIZATION LAYERS

class LearnableNorm(nn.Module):
    """
    Input-conditioned Adaptive Normalization.

    Motivation
    ----------
    Standard BatchNorm computes statistics over the whole batch, which
    blurs inter-subject variability. InstanceNorm computes per-sample
    statistics but ignores cross-channel context. LayerNorm is powerful
    but uses fixed affine parameters shared across all inputs.

    LearnableNorm combines the benefits:
      1. Compute per-sample, per-channel statistics (like InstanceNorm).
      2. Predict affine parameters γ and β from a summary of the input
         signal itself via a small two-layer MLP ("hyper-network").
         This makes the normalization *context-aware*: two windows from
         the same class but different patients will receive different
         γ/β, adapting to signal amplitude, baseline wander, etc.

    This is the primary novel contribution of the normalization module
    described in thesis objective 3.2.3.

    Parameters
    ----------
    num_features : number of channels C
    hidden_dim   : width of the hyper-network MLP
    eps          : numerical stability term

    Input  : (B, C, T)
    Output : (B, C, T) — normalized and re-scaled
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim:   int = 32,
        eps:          float = 1e-5,
    ):
        super().__init__()
        self.eps = eps

        # Hyper-network: global signal summary → γ, β
        # Input: (B, C*2) — per-channel mean and std concatenated
        self.hyper = nn.Sequential(
            nn.Linear(num_features * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features * 2), # predicts γ and β
        )
        # Initialise hyper-net output to identity (γ=1, β=0)
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mu  = x.mean(dim=2)                         # (B, C)
        std = x.std(dim=2).clamp(min=self.eps)      # (B, C)

        # Normalise
        x_norm = (x - mu.unsqueeze(2)) / std.unsqueeze(2)   # (B, C, T)

        # Predict affine parameters from signal statistics
        summary = torch.cat([mu, std], dim=1)        # (B, C*2)
        affine  = self.hyper(summary)                # (B, C*2)
        gamma   = 1.0 + affine[:, :x.shape[1]]       # (B, C) — centred at 1
        beta    = affine[:, x.shape[1]:]             # (B, C) — centred at 0

        return x_norm * gamma.unsqueeze(2) + beta.unsqueeze(2)


class AdaptiveLayerNorm(nn.Module):
    """
    LayerNorm applied over the time axis with learnable affine parameters.

    Simpler than LearnableNorm but still per-sample (not per-batch).
    Used as an ablation baseline when adaptive_norm="layer".

    Input  : (B, C, T)
    Output : (B, C, T)
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm expects (..., C); x is (B, C, T)
        x = x.permute(0, 2, 1)     # (B, T, C)
        x = self.norm(x)
        return x.permute(0, 2, 1)  # (B, C, T)


def build_adaptive_norm(method: str, num_features: int) -> nn.Module:
    """Factory for adaptive normalization layers."""
    if method == "learnable":
        return LearnableNorm(num_features)
    elif method == "layer":
        return AdaptiveLayerNorm(num_features)
    elif method == "instance":
        return nn.InstanceNorm1d(num_features, affine=True)
    else:
        raise ValueError(f"Unknown adaptive_norm method: '{method}'")


# 3. SQUEEZE-AND-EXCITATION CHANNEL ATTENTION

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.

    Recalibrates channel-wise feature responses by modelling channel
    interdependencies. Particularly effective for biomedical signals
    where different channels (e.g., ECG leads) carry complementary
    information with varying relevance per class.

    Input  : (B, C, T)
    Output : (B, C, T) — channel-recalibrated
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                        # (B, C, 1)
            nn.Flatten(),                                   # (B, C)
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(2)    # (B, C, 1)
        return x * scale


# 4. RESIDUAL CNN BLOCK

class ResidualConvBlock(nn.Module):
    """
    Pre-activation residual block for 1-D signals.

      BN → ReLU → Conv → BN → ReLU → Conv → SE → + (skip)

    Pre-activation (He et al., 2016) places BatchNorm before the
    convolution, which improves gradient flow and is standard in
    deep biomedical signal networks. The skip connection is a 1×1
    conv projection when in/out channels differ.

    Input  : (B, C_in,  T)
    Output : (B, C_out, T)  — same temporal length (stride=1, same-padding)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dropout:      float,
        se_ratio:     int,
        norm_layer:   Optional[nn.Module] = None,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.pre_norm1 = norm_layer if norm_layer else nn.BatchNorm1d(in_channels)
        self.pre_norm2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=pad, bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=pad, bias=False,
        )
        self.se       = SEBlock(out_channels, se_ratio)
        self.dropout  = nn.Dropout(dropout)

        # Skip projection if channel dimensions differ
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = F.relu(self.pre_norm1(x), inplace=True)
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.pre_norm2(out), inplace=True)
        out = self.conv2(out)
        out = self.se(out)

        return out + residual


# 5. MULTI-SCALE CNN ENCODER

class MultiScaleCNNEncoder(nn.Module):
    """
    Parallel multi-scale convolutional feature extractor.

    Architecture
    ------------
    For each kernel size k in [3, 7, 15, 31]:
      Input (B, C, T)
        → AdaptiveNorm                         signal conditioning
        → Conv1d(C, F, k) + BN + ReLU          feature projection
        → ResidualConvBlock × depth            deep feature extraction
        → AdaptiveAvgPool1d(T_out)             fixed-length output

    All branches are concatenated along the channel axis:
      output: (B, F × n_branches, T_out)

    Motivation
    ----------
    Different physiological patterns appear at different temporal scales:
      k=3  → high-frequency noise / spike detection (QRS peaks, EEG spikes)
      k=7  → waveform morphology (P-wave, T-wave in ECG)
      k=15 → mid-range patterns (alpha/beta bursts in EEG, BVP cycles)
      k=31 → slow trends (baseline wander, respiration modulation in PPG)
    Concatenating all branches gives the LSTM a rich multi-resolution
    feature sequence that a single-kernel CNN cannot provide.

    T_out: temporal resolution passed to the LSTM.
           Fixed at T // (pool_factor) regardless of input length.
    """

    def __init__(
        self,
        in_channels:   int,
        base_filters:  int,
        kernels:       List[int],
        depth:         int,
        dropout:       float,
        se_ratio:      int,
        adaptive_norm: str,
        t_out:         int = 64, # fixed output length for LSTM
    ):
        super().__init__()
        self.kernels    = kernels
        self.t_out      = t_out
        n_branches      = len(kernels)
        self.out_channels = base_filters * n_branches

        self.branches = nn.ModuleList()
        for k in kernels:
            # Input adaptive norm (per-branch, input-conditioned)
            norm = build_adaptive_norm(adaptive_norm, in_channels)

            # Initial projection
            proj = nn.Sequential(
                nn.Conv1d(in_channels, base_filters,
                          kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(inplace=True),
            )

            # Residual blocks
            blocks = nn.Sequential(*[
                ResidualConvBlock(
                    base_filters, base_filters, k,
                    dropout=dropout, se_ratio=se_ratio,
                )
                for _ in range(depth)
            ])

            self.branches.append(nn.ModuleDict({
                "norm":   norm,
                "proj":   proj,
                "blocks": blocks,
            }))

        # Fixed-length pooling so all inputs produce the same T_out
        self.pool = nn.AdaptiveAvgPool1d(t_out)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns
        -------
        fused    : (B, F×n_branches, T_out)  — concatenated branch outputs
        branches : list of per-branch tensors (for ablation / XAI)
        """
        branch_outs = []
        for branch in self.branches:
            out = branch["norm"](x)
            out = branch["proj"](out)
            out = branch["blocks"](out)
            out = self.pool(out)              # (B, F, T_out)
            branch_outs.append(out)

        fused = torch.cat(branch_outs, dim=1) # (B, F×n_branches, T_out)
        return fused, branch_outs


# 6. MULTI-HEAD TEMPORAL ATTENTION

class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head self-attention pooling over the time axis.

    Unlike the single-head additive attention in LSTMBaseline, this
    module allows different heads to specialise in different temporal
    patterns simultaneously (e.g., one head for QRS complexes, another
    for P-waves in ECG).

    The output is a weighted sum over time steps for each head,
    then concatenated and projected back to d_model.

    Input  : (B, T, d_model)
    Output : (B, d_model)    — time-pooled representation
             (B, n_heads, T) — attention weights per head (for XAI)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

        # Learnable global query vector (one per head) — replaces CLS token
        # This lets each head learn what "summary feature" to attend to.
        self.global_q = nn.Parameter(
            torch.randn(1, n_heads, 1, self.d_head) * 0.02
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        # Project and reshape to (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)

        # Global query: (1, H, 1, Dh) → (B, H, 1, Dh)
        q = self.global_q.expand(B, -1, -1, -1)

        # Scaled dot-product attention: (B, H, 1, T)
        scores  = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        weights = torch.softmax(scores, dim=-1) # (B, H, 1, T)
        weights = self.drop(weights)

        # Context vector: (B, H, 1, Dh) → (B, D)
        ctx = (weights @ v)                     # (B, H, 1, Dh)
        ctx = ctx.squeeze(2).reshape(B, D)      # (B, D)
        ctx = self.out(ctx)

        # Squeeze attention weights for interpretability: (B, H, T)
        attn_map = weights.squeeze(2)

        return ctx, attn_map


# 7. GATED FEATURE FUSION

class GatedFusion(nn.Module):
    """
    Learnable gated fusion of CNN and LSTM feature vectors.

    Simple concatenation gives both streams equal weight regardless
    of which is more informative for a given sample. A gating mechanism
    lets the model learn to trust the CNN more for morphology-heavy
    classes (e.g., arrhythmia types) and the LSTM more for temporal
    pattern classes (e.g., motor imagery in EEG).

    Gate: g = σ(W_g [f_cnn; f_lstm] + b)
    Out : f_fused = g ⊙ f_cnn_proj + (1-g) ⊙ f_lstm_proj

    Both streams are projected to fusion_hidden before gating,
    so no single stream dominates by virtue of dimensionality.

    Input  : f_cnn  (B, d_cnn),  f_lstm (B, d_lstm)
    Output : (B, fusion_hidden)
    """

    def __init__(
        self,
        d_cnn:         int,
        d_lstm:        int,
        fusion_hidden: int,
        dropout:       float,
    ):
        super().__init__()
        self.proj_cnn  = nn.Linear(d_cnn,  fusion_hidden, bias=False)
        self.proj_lstm = nn.Linear(d_lstm, fusion_hidden, bias=False)
        self.gate      = nn.Linear(d_cnn + d_lstm, fusion_hidden)
        self.norm      = nn.LayerNorm(fusion_hidden)
        self.drop      = nn.Dropout(dropout)

    def forward(
        self,
        f_cnn:  torch.Tensor,   # (B, d_cnn)
        f_lstm: torch.Tensor,   # (B, d_lstm)
    ) -> torch.Tensor:
        pc = self.proj_cnn(f_cnn)                       # (B, F)
        pl = self.proj_lstm(f_lstm)                     # (B, F)
        g  = torch.sigmoid(
            self.gate(torch.cat([f_cnn, f_lstm], dim=1))
        )                                               # (B, F)
        fused = g * pc + (1 - g) * pl                   # (B, F)
        return self.drop(self.norm(fused))


# 8. HYBRID CNN–LSTM MODEL

class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN–LSTM model for multimodal biomedical signal classification.

    Full Architecture
    -----------------

    Input (B, C, T)
      │
      ├── Multi-Scale CNN Encoder
      │     ├── Branch k=3  [AdaptiveNorm → Conv → ResBlocks × depth → AvgPool]
      │     ├── Branch k=7  [AdaptiveNorm → Conv → ResBlocks × depth → AvgPool]
      │     ├── Branch k=15 [AdaptiveNorm → Conv → ResBlocks × depth → AvgPool]
      │     └── Branch k=31 [AdaptiveNorm → Conv → ResBlocks × depth → AvgPool]
      │           ↓ concat
      │     CNN features: (B, F×4, T_out)
      │           │
      │           ├──→ CNN global pool → f_cnn (B, F×4)    ← for gated fusion
      │           │
      │           └──→ permute (B, T_out, F×4)
      │                       ↓
      ├── Bidirectional LSTM Encoder
      │     ├── BiLSTM stack (hidden=192, layers=2)
      │     └── Multi-head temporal attention (heads=4)
      │           ↓
      │     LSTM features: f_lstm (B, hidden*2)
      │
      ├── Gated Feature Fusion
      │     f_cnn ──┐
      │             ├──→ Gate → f_fused (B, fusion_hidden)
      │     f_lstm ─┘
      │
      └── Classification Head
            Linear(fusion_hidden, fc_hidden) → BN → ReLU → Dropout
            Linear(fc_hidden, n_classes)

    Design Rationale
    ----------------
    The model is designed so each component addresses a specific
    limitation of the baselines:

    CNN Baseline limitation → no temporal context after pooling
    Solution → the CNN output is NOT immediately pooled; instead it is
    passed as a sequence of T_out feature vectors to the LSTM, which
    can then model temporal relationships BETWEEN spatial features.

    LSTM Baseline limitation → raw channels fed as features
    Solution → the CNN pre-processes signals into rich multi-scale
    feature maps before the LSTM ever sees them. The LSTM receives
    "pre-digested" morphological features, not raw samples.

    Both → no mechanism to weight which stream is more relevant
    Solution → GatedFusion dynamically balances CNN vs LSTM contributions
    per sample and per class.

    Adaptive Normalization → per-branch, input-conditioned γ/β
    This ensures that the model does not rely on the preprocessing
    pipeline's static normalization; it can adapt to distribution
    shifts at inference time (e.g., different subjects, sensor drift).
    """

    model_type = "HybridCNNLSTM"

    def __init__(
        self,
        in_channels: int,
        n_classes:   int,
        cfg:         HybridConfig = CFG,
    ):
        super().__init__()
        n_branches  = len(cfg.cnn_kernels)
        cnn_out_ch  = cfg.cnn_base_filters * n_branches # e.g. 32×4=128
        lstm_in     = cnn_out_ch
        lstm_hidden = cfg.lstm_hidden * (2 if cfg.lstm_bidir else 1) # 384
        t_out       = 64 # fixed temporal resolution from CNN

        # Encoders
        self.cnn_encoder = MultiScaleCNNEncoder(
            in_channels   = in_channels,
            base_filters  = cfg.cnn_base_filters,
            kernels       = cfg.cnn_kernels,
            depth         = cfg.cnn_depth,
            dropout       = cfg.cnn_dropout,
            se_ratio      = cfg.se_ratio,
            adaptive_norm = cfg.adaptive_norm,
            t_out         = t_out,
        )

        self.lstm = nn.LSTM(
            input_size    = lstm_in,
            hidden_size   = cfg.lstm_hidden,
            num_layers    = cfg.lstm_layers,
            batch_first   = True,
            dropout       = cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional = cfg.lstm_bidir,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)

        self.mh_attn = MultiHeadTemporalAttention(
            d_model  = lstm_hidden,
            n_heads  = cfg.attn_heads,
            dropout  = cfg.attn_dropout,
        )

        # CNN global pooling path (for gated fusion)
        self.cnn_global_pool = nn.AdaptiveAvgPool1d(1)

        # Gated fusion
        self.fusion = GatedFusion(
            d_cnn         = cnn_out_ch,
            d_lstm        = lstm_hidden,
            fusion_hidden = cfg.fusion_hidden,
            dropout       = cfg.fusion_dropout,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(cfg.fusion_hidden, cfg.fc_hidden),
            nn.BatchNorm1d(cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.fc_dropout),
            nn.Linear(cfg.fc_hidden, n_classes),
        )

        self._init_weights()

    # Forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_full(x)[0]

    def _forward_full(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logits, attn_weights, gate_values) — used for XAI.

        attn_weights : (B, n_heads, T_out)  multi-head temporal attention
        gate_values  : (B, fusion_hidden)   gating weights ≈ CNN vs LSTM trust
        """
        # CNN encoder
        cnn_seq, _ = self.cnn_encoder(x) # (B, C_cnn, T_out)

        # CNN global representation (for fusion)
        f_cnn = self.cnn_global_pool(cnn_seq).squeeze(-1) # (B, C_cnn)

        # LSTM encoder
        lstm_in = cnn_seq.permute(0, 2, 1) # (B, T_out, C_cnn)
        lstm_out, _ = self.lstm(lstm_in)   # (B, T_out, H_bidir)
        lstm_out    = self.lstm_norm(lstm_out)

        f_lstm, attn_weights = self.mh_attn(lstm_out) # (B, H_bidir), (B, heads, T)

        # Gated fusion
        f_fused = self.fusion(f_cnn, f_lstm) # (B, fusion_hidden)

        # Store gate signal for XAI module
        with torch.no_grad():
            gate_vals = torch.sigmoid(
                self.fusion.gate(torch.cat([f_cnn, f_lstm], dim=1))
            ).detach()

        logits = self.head(f_fused)
        return logits, attn_weights, gate_vals

    # Interpretability accessors

    def get_attention_maps(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, attention_maps) for XAI analysis."""
        logits, attn, _ = self._forward_full(x)
        return logits, attn

    def get_cnn_branch_activations(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """Return per-scale CNN feature maps — input for gradient-based XAI."""
        _, branches = self.cnn_encoder(x)
        return branches

    # Weight initialization

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0) # forget-gate = 1
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)


# 9. MIXUP AUGMENTATION

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    n_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup data augmentation (Zhang et al., 2018).

    Interpolates pairs of training examples:
      x_mix  = λ · x_i  + (1-λ) · x_j
      y_mix  = λ · y_i  + (1-λ) · y_j (soft labels)

    λ ~ Beta(α, α). When α → 0, mixup degenerates to no augmentation.

    Particularly useful for biomedical signals because:
      - Interpolated ECG/EEG segments are physiologically plausible
      - Smoothes the decision boundary between adjacent classes
      - Acts as additional regularisation against overfitting on small datasets

    Returns (x_mixed, y_soft_onehot) where y_soft is float (B, n_classes).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B   = x.size(0)
    idx = torch.randperm(B, device=x.device)

    x_mix = lam * x + (1 - lam) * x[idx]

    # Convert to one-hot soft labels
    y_onehot = F.one_hot(y, n_classes).float()
    y_mix    = lam * y_onehot + (1 - lam) * y_onehot[idx]

    return x_mix, y_mix


# 10. LEARNING RATE SCHEDULE

class WarmupCosineScheduler:
    """
    Linear warm-up followed by cosine annealing with warm restarts.

    Warm-up prevents large gradient updates from the randomly initialised
    LearnableNorm hyper-network in the first few epochs. Cosine annealing
    then explores the loss landscape efficiently.

    Uses PyTorch's CosineAnnealingWarmRestarts internally.
    """

    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        base_lr:       float,
        min_lr:        float = 1e-6,
        T_0:           int   = 30, # cosine restart period
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.epoch         = 0

        self.cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=min_lr,
        )

    def step(self) -> float:
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.base_lr * self.epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.cosine.step(self.epoch - self.warmup_epochs)
            lr = self.optimizer.param_groups[0]["lr"]
        return lr


# 11. HYBRID TRAINER

class HybridTrainer:
    """
    Training loop for HybridCNNLSTM.

    Additions over BaselineTrainer:
      - Mixup augmentation with soft-label cross-entropy
      - Label smoothing in the loss function
      - Warm-up + cosine-annealing LR schedule
      - Gradient norm logging (useful for diagnosing LSTM instability)
    """

    def __init__(
        self,
        model:      HybridCNNLSTM,
        loaders:    Dict[str, DataLoader],
        n_classes:  int,
        output_dir: str,
        cfg:        HybridConfig = CFG,
    ):
        self.model      = model
        self.loaders    = loaders
        self.cfg        = cfg
        self.output_dir = output_dir
        self.device     = cfg.device
        self.n_classes  = n_classes

        y_train = loaders["train"].dataset.y
        weights = compute_class_weights(y_train, self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight         = weights,
            label_smoothing= cfg.label_smoothing,
            reduction      = "mean",
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg.learning_rate,
            weight_decay = cfg.weight_decay,
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs = cfg.warmup_epochs,
            total_epochs  = cfg.epochs,
            base_lr       = cfg.learning_rate,
        )

        self.history:     List[Dict] = []
        self.best_val_f1: float      = -1.0
        self.best_state:  dict       = {}
        self.no_improve:  int        = 0

    # Single epoch

    def _run_epoch(self, split: str) -> Dict:
        training = (split == "train")
        self.model.train(training)
        loader   = self.loaders[split]

        total_loss      = 0.0
        total_grad_norm = 0.0
        n_batches       = 0
        all_preds:      List[np.ndarray] = []
        all_labels:     List[np.ndarray] = []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)

                if training and self.cfg.mixup_alpha > 0:
                    X, y_soft = mixup_batch(
                        X, y, self.cfg.mixup_alpha, self.n_classes
                    )
                    logits     = self.model(X)
                    # Soft-label loss: manual cross-entropy
                    log_probs = F.log_softmax(logits, dim=1)
                    loss      = -(y_soft * log_probs).sum(dim=1).mean()
                    y_hard    = y # for metric computation
                else:
                    logits = self.model(X)
                    loss   = self.criterion(logits, y)
                    y_hard = y

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    total_grad_norm += grad_norm.item()
                    self.optimizer.step()

                total_loss += loss.item() * len(y_hard)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_labels.append(y_hard.cpu().numpy())
                n_batches += 1

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        avg    = "macro" if self.n_classes > 2 else "binary"

        return {
            "loss":      total_loss / len(y_true),
            "acc":       accuracy_score(y_true, y_pred),
            "f1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
            "grad_norm": total_grad_norm / max(n_batches, 1),
        }

    # Full training loop

    def fit(self) -> pd.DataFrame:
        log.info(
            "Training HybridCNNLSTM | %d epochs | device=%s | "
            "mixup_α=%.2f | label_smooth=%.2f",
            self.cfg.epochs, self.device,
            self.cfg.mixup_alpha, self.cfg.label_smoothing,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            lr = self.scheduler.step()

            train_m = self._run_epoch("train")
            val_m   = self._run_epoch("val")
            elapsed = time.time() - t0

            row = {
                "epoch":           epoch,
                "lr":              lr,
                "train_loss":      train_m["loss"],
                "train_acc":       train_m["acc"],
                "train_f1":        train_m["f1"],
                "train_grad_norm": train_m["grad_norm"],
                "val_loss":        val_m["loss"],
                "val_acc":         val_m["acc"],
                "val_f1":          val_m["f1"],
                "elapsed_s":       round(elapsed, 2),
            }
            self.history.append(row)

            log.info(
                "Ep %3d/%d | lr=%.2e | "
                "train loss=%.4f acc=%.3f f1=%.3f gn=%.2f | "
                "val loss=%.4f acc=%.3f f1=%.3f | %.1fs",
                epoch, self.cfg.epochs, lr,
                train_m["loss"], train_m["acc"],
                train_m["f1"],   train_m["grad_norm"],
                val_m["loss"],   val_m["acc"],  val_m["f1"], elapsed,
            )

            if val_m["f1"] > self.best_val_f1:
                self.best_val_f1 = val_m["f1"]
                self.best_state  = deepcopy(self.model.state_dict())
                self.no_improve  = 0
                self._save_checkpoint(epoch, val_m["f1"])
            else:
                self.no_improve += 1
                if self.no_improve >= self.cfg.patience:
                    log.info(
                        "Early stopping at epoch %d", epoch
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
        torch.save({
            "epoch":       epoch,
            "val_f1":      val_f1,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "cfg":         self.cfg.__dict__,
        }, os.path.join(self.output_dir, "best_hybrid.pt"))

    def _plot_history(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        pairs = [("loss", axes[0, 0]), ("acc", axes[0, 1]),
                 ("f1",   axes[1, 0]), ("train_grad_norm", axes[1, 1])]
        for metric, ax in pairs:
            if metric == "train_grad_norm":
                ax.plot(df["epoch"], df["train_grad_norm"], color="purple")
                ax.set_title("Gradient Norm (train)")
            else:
                ax.plot(df["epoch"], df[f"train_{metric}"], label="train")
                ax.plot(df["epoch"], df[f"val_{metric}"],   label="val")
                ax.set_title(metric.capitalize())
                ax.legend()
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        fig.suptitle("HybridCNNLSTM — Training History", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "training_history.png"), dpi=150
        )
        plt.close()


# 12. EVALUATION

def evaluate_hybrid(
    model:      HybridCNNLSTM,
    loader:     DataLoader,
    label_map:  Dict[str, str],
    output_dir: str,
    cfg:        HybridConfig = CFG,
) -> Dict:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []
    total_time    = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(cfg.device)
            t0 = time.perf_counter()
            logits, attn, _ = model._forward_full(X)
            total_time    += time.perf_counter() - t0
            total_samples += len(y)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
            all_attn.append(attn.mean(dim=1).cpu().numpy()) # mean over heads

    y_true   = np.concatenate(all_labels)
    y_pred   = np.concatenate(all_preds)
    attn_all = np.concatenate(all_attn, axis=0) # (N, T_out)

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
    _plot_attention_heatmap(attn_all, y_true, class_names, output_dir)

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


def _plot_attention_heatmap(
    attn:        np.ndarray, # (N, T_out)
    y_true:      np.ndarray,
    class_names: List[str],
    output_dir:  str,
) -> None:
    """
    Plot mean attention weight profile per class.

    Shows WHICH temporal positions the model consistently attends to
    for each class — a lightweight XAI visualisation. More rigorous
    attribution (LRP, GradCAM) is handled in the explainability module.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        1, n_classes, figsize=(4 * n_classes, 3), sharey=True
    )
    if n_classes == 1:
        axes = [axes]

    for cls_idx, (ax, name) in enumerate(zip(axes, class_names)):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            continue
        mean_attn = attn[mask].mean(axis=0) # (T_out,)
        ax.plot(mean_attn, linewidth=1.5)
        ax.fill_between(range(len(mean_attn)), mean_attn, alpha=0.3)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Timestep (T_out)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Mean attention weight")
    fig.suptitle("Temporal Attention Profile per Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "attention_profiles.png"), dpi=150
    )
    plt.close()


# 13. FINAL COMPARISON

def build_final_report(
    hybrid_results:   Dict[str, Dict], # {modality: metrics}
    baseline_dir:     str,
    output_dir:       str,
) -> pd.DataFrame:
    """
    Load baseline metrics and build a three-way comparison table:
    CNN vs LSTM vs Hybrid across all modalities and metrics.

    Saves: final_comparison.csv and final_comparison.png
    """
    rows = []
    for modality, h_metrics in hybrid_results.items():
        for model_type in ("cnn", "lstm"):
            path = os.path.join(
                baseline_dir, modality, model_type, "test_metrics.json"
            )
            if not os.path.exists(path):
                log.warning("Baseline metrics not found: %s", path)
                continue
            with open(path) as f:
                b_metrics = json.load(f)
            rows.append({
                "Modality":     modality.upper(),
                "Model":        model_type.upper(),
                "Accuracy":     b_metrics["accuracy"],
                "Macro F1":     b_metrics["macro_f1"],
                "Weighted F1":  b_metrics["weighted_f1"],
                "Latency (ms)": b_metrics["latency_ms_per_sample"],
            })
        rows.append({
            "Modality":     modality.upper(),
            "Model":        "Hybrid",
            "Accuracy":     h_metrics["accuracy"],
            "Macro F1":     h_metrics["macro_f1"],
            "Weighted F1":  h_metrics["weighted_f1"],
            "Latency (ms)": h_metrics["latency_ms_per_sample"],
        })

    df = pd.DataFrame(rows).sort_values(["Modality", "Model"])
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "final_comparison.csv")
    df.to_csv(csv_path, index=False)
    log.info("Final comparison →\n%s", df.to_string(index=False))

    _plot_final_comparison(df, output_dir)
    return df


def _plot_final_comparison(df: pd.DataFrame, output_dir: str) -> None:
    modalities = df["Modality"].unique()
    metrics    = ["Accuracy", "Macro F1", "Weighted F1"]
    palette    = {"CNN": "#4C72B0", "LSTM": "#DD8452", "Hybrid": "#55A868"}

    fig, axes = plt.subplots(
        len(modalities), len(metrics),
        figsize=(5 * len(metrics), 4 * len(modalities)),
        squeeze=False,
    )
    for r, mod in enumerate(modalities):
        sub = df[df["Modality"] == mod]
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            colors = [palette.get(m, "gray") for m in sub["Model"]]
            bars   = ax.bar(sub["Model"], sub[metric], color=colors,
                            edgecolor="white", linewidth=0.8)
            ax.set_ylim(
                max(0, sub[metric].min() - 0.05),
                min(1.0, sub[metric].max() + 0.07),
            )
            for bar, val in zip(bars, sub[metric]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )
            ax.set_title(f"{mod} — {metric}", fontsize=10)
            ax.set_ylabel(metric if c == 0 else "")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("CNN vs LSTM vs Hybrid — Final Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "final_comparison.png"), dpi=150
    )
    plt.close()


# 14. ABLATION STUDY HELPERS

def ablation_no_adaptive_norm(
    in_channels: int,
    n_classes:   int,
    cfg:         HybridConfig = CFG,
) -> HybridCNNLSTM:
    """Hybrid model with InstanceNorm instead of LearnableNorm."""
    ablation_cfg = deepcopy(cfg)
    ablation_cfg.adaptive_norm = "instance"
    return HybridCNNLSTM(in_channels, n_classes, ablation_cfg)


def ablation_single_scale_cnn(
    in_channels: int,
    n_classes:   int,
    cfg:         HybridConfig = CFG,
) -> HybridCNNLSTM:
    """Hybrid model with single CNN branch (k=7) instead of multi-scale."""
    ablation_cfg = deepcopy(cfg)
    ablation_cfg.cnn_kernels = [7]
    return HybridCNNLSTM(in_channels, n_classes, ablation_cfg)


def ablation_no_se(
    in_channels: int,
    n_classes:   int,
    cfg:         HybridConfig = CFG,
) -> HybridCNNLSTM:
    """Hybrid model with SE ratio = 0 (no channel attention)."""
    ablation_cfg = deepcopy(cfg)
    ablation_cfg.se_ratio = 0
    return HybridCNNLSTM(in_channels, n_classes, ablation_cfg)


def ablation_no_gated_fusion(
    in_channels: int,
    n_classes:   int,
    cfg:         HybridConfig = CFG,
) -> HybridCNNLSTM:
    """
    Replace GatedFusion with simple concatenation.
    Requires subclassing — provided as a reference configuration.
    Set fusion_hidden to cnn_out_ch + lstm_hidden and swap GatedFusion
    for a single Linear projection in post-processing analysis.
    """
    log.info(
        "No-gate ablation: use simple concat + Linear(%d, %d) instead "
        "of GatedFusion.",
        len(cfg.cnn_kernels) * cfg.cnn_base_filters
        + cfg.lstm_hidden * (2 if cfg.lstm_bidir else 1),
        cfg.fusion_hidden,
    )
    return HybridCNNLSTM(in_channels, n_classes, cfg)


# 15. MAIN ENTRY POINT

def run_hybrid(cfg: HybridConfig = CFG) -> None:
    """
    Train and evaluate the HybridCNNLSTM on all three modalities,
    then produce a three-way comparison against the saved baselines.
    """
    set_seed(cfg.seed)
    log.info("=" * 60)
    log.info("Hybrid CNN–LSTM Training")
    log.info("=" * 60)
    log.info(
        "Device: %s | norm: %s | kernels: %s | lstm_hidden: %d",
        cfg.device, cfg.adaptive_norm,
        cfg.cnn_kernels, cfg.lstm_hidden,
    )

    hybrid_results: Dict[str, Dict] = {}

    for modality in cfg.modalities:
        h5_path = os.path.join(cfg.processed_dir, f"{modality}.h5")
        if not os.path.exists(h5_path):
            log.warning("HDF5 not found for %s — skipping.", modality)
            continue

        log.info("\n%s\n── Modality: %s\n%s",
                 "=" * 60, modality.upper(), "=" * 60)

        loaders   = make_loaders(h5_path, modality, cfg)
        train_ds  = loaders["train"].dataset
        n_ch      = train_ds.n_channels
        seq_len   = train_ds.seq_len
        n_classes = train_ds.n_classes
        label_map = train_ds.label_map

        out_dir = os.path.join(cfg.output_dir, modality)
        os.makedirs(out_dir, exist_ok=True)

        model = HybridCNNLSTM(n_ch, n_classes, cfg).to(cfg.device)
        n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        log.info("HybridCNNLSTM — Trainable parameters: %s", f"{n_params:,}")

        trainer = HybridTrainer(
            model=model, loaders=loaders, n_classes=n_classes,
            output_dir=out_dir, cfg=cfg,
        )
        trainer.fit()

        metrics = evaluate_hybrid(
            model, loaders["test"], label_map, out_dir, cfg
        )
        hybrid_results[modality] = metrics

    build_final_report(hybrid_results, cfg.baseline_dir, cfg.output_dir)
    log.info("\nHybrid training complete. Outputs → %s", cfg.output_dir)

# USAGE
# 
#
# Run with defaults (requires baseline_models.py in same directory):
#   python hybrid_model.py
#
# Custom config:
#   from hybrid_model import run_hybrid, HybridConfig
#   cfg = HybridConfig(
#       modalities    = ["ecg"],
#       adaptive_norm = "learnable",   # or "layer" / "instance"
#       cnn_kernels   = [3, 7, 15, 31],
#       epochs        = 80,
#   )
#   run_hybrid(cfg)
#
# Run ablation study:
#   from hybrid_model import ablation_no_adaptive_norm, HybridTrainer
#   model = ablation_no_adaptive_norm(in_channels=2, n_classes=5)
#
# Load checkpoint:
#   ckpt = torch.load("outputs/hybrid/ecg/best_hybrid.pt")
#   model.load_state_dict(ckpt["model_state"])

if __name__ == "__main__":
    run_hybrid(CFG)
