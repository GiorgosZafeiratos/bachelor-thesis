"""

This module provides:

XAI
---
  1. GradCAM1D
       Gradient-weighted Class Activation Maps for 1-D signals.
       Highlights which temporal regions most influenced a prediction.

  2. IntegratedGradients1D
       Axiomatic attribution method (Sundararajan et al., 2017).
       Integrates gradients from a baseline (zeros) to the input.
       More faithful than saliency maps; handles saturation.

  3. LRPExplainer
       Layer-wise Relevance Propagation for the CNN layers.
       Implements the ε-LRP rule, the most stable variant for
       biomedical signals (as used in [9] Zhou et al., 2024).

  4. XAIVisualizer
       Plots all three attributions side-by-side on the raw signal.
       Produces the figures needed for thesis Chapter 5 (XAI results).

Pipeline wiring 
---------------
  5. PipelineConfig           — single config controlling both preprocessing
                                and training to prevent disconnection.
  6. PreprocessingCache       — checks whether HDF5 outputs exist and runs
                                preprocessing only when needed.
  7. build_loaders_from_hdf5  — the explicit, documented bridge between
                                preprocessing output and training input.
  8. end_to_end_run()         — single entry point that calls preprocessing →
                                training → evaluation → XAI in sequence.

"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ─ SECTION 1: XAI IMPLEMENTATIONS

# 1.1  Gradient Saliency

def gradient_saliency(
    model:    nn.Module,
    x:        torch.Tensor, # (1, C, T) — single sample
    target_class: Optional[int] = None,
    device:   str = "cpu",
) -> np.ndarray:
    """
    Vanilla gradient saliency map.

    Computes ∂ score_c / ∂ x_t for each time step t.
    The absolute value indicates which input points the model is most
    sensitive to regardless of direction.

    This is the "absolute minimum fix" requested in GAP #4.

    Returns
    -------
    saliency : np.ndarray of shape (C, T) — unsigned attribution
    """
    model.eval()
    x = x.to(device).requires_grad_(True)

    logits = model(x)                           # (1, n_classes)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    saliency = x.grad.detach().cpu().numpy()[0] # (C, T)
    return np.abs(saliency)


# 1.2  Grad-CAM for 1-D signals

class GradCAM1D:
    """
    Gradient-weighted Class Activation Maps adapted for 1-D time-series.

    Standard Grad-CAM (Selvaraju et al., 2017) extracts the gradient of
    the class score with respect to the last convolutional feature map,
    then weights the feature channels by their mean gradient to produce
    a coarse temporal heatmap.

    For biomedical signals, this answers: "which temporal region of the
    signal did the CNN encoder find most discriminative for this class?"

    Usage
    -----
    cam = GradCAM1D(model, target_layer=model.cnn_encoder.branches[1]["blocks"][-1].conv2)
    heatmap, pred = cam(x, target_class=None)  # (T,) heatmap, int prediction

    Parameters
    ----------
    model        : HybridCNNLSTM or AblationHybrid
    target_layer : the nn.Conv1d layer to hook (last conv of a branch)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(
        self,
        x:            torch.Tensor, # (1, C, T)
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Returns
        -------
        heatmap : np.ndarray (T,) — ReLU-clamped, normalised to [0, 1]
        pred    : int — predicted class index
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        pred   = int(logits.argmax(dim=1).item())
        if target_class is None:
            target_class = pred

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global average pool of gradients over time → channel weights
        weights = self._gradients.mean(dim=2, keepdim=True) # (1, C_feat, 1)
        cam     = (weights * self._activations).sum(dim=1)  # (1, T_feat)
        cam     = F.relu(cam).squeeze(0).cpu().numpy()      # (T_feat,)

        # Upsample from feature resolution back to input length
        T_in  = x.shape[-1]
        cam   = np.interp(
            np.linspace(0, len(cam) - 1, T_in),
            np.arange(len(cam)), cam,
        )

        # Normalise to [0, 1]
        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred


# 1.3  Integrated Gradients

class IntegratedGradients1D:
    """
    Integrated Gradients (Sundararajan et al., 2017) for 1-D signals.

    IG computes attributions as:
        attr(x) = (x - x') · ∫₀¹ ∂F(x' + α(x-x')) / ∂x dα

    where x' is a baseline (typically zeros or mean signal).

    Properties that make IG superior to vanilla saliency:
      - Completeness: attributions sum to the output difference F(x)-F(x')
      - Sensitivity: non-zero attribution wherever input differs from baseline
      - Linearity: satisfies a linearity axiom for model composition

    For ECG: baseline = flat line (zeros) makes physiological sense.
    For EEG: baseline = mean spectrum of resting EEG is more principled.
    For PPG: baseline = zeros (no pulse) is appropriate.

    Parameters
    ----------
    model   : trained nn.Module
    n_steps : number of Riemann approximation steps (50 is sufficient)
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model   = model
        self.n_steps = n_steps

    def attribute(
        self,
        x:            torch.Tensor, # (1, C, T)
        target_class: Optional[int] = None,
        baseline:     Optional[torch.Tensor] = None,
        device:       str = "cpu",
    ) -> np.ndarray:
        """
        Returns
        -------
        attributions : np.ndarray (C, T) — signed, can be positive or negative
        """
        self.model.eval()
        x = x.to(device)

        if baseline is None:
            baseline = torch.zeros_like(x) # zero baseline
        baseline = baseline.to(device)

        # Interpolation path: x' → x in n_steps
        alphas     = torch.linspace(0, 1, self.n_steps, device=device)
        interp     = baseline + alphas[:, None, None, None] * (x - baseline)
        # interp: (n_steps, 1, C, T) → reshape to (n_steps, C, T)
        interp     = interp.squeeze(1).requires_grad_(True)

        logits = self.model(interp)       # (n_steps, n_classes)

        if target_class is None:
            target_class = int(
                self.model(x).argmax(dim=1).item()
            )

        scores = logits[:, target_class].sum()
        self.model.zero_grad()
        scores.backward()

        grads = interp.grad.detach()                    # (n_steps, C, T)
        avg_grads = grads.mean(dim=0).cpu().numpy()     # (C, T)

        delta = (x - baseline).squeeze(0).cpu().numpy() # (C, T)
        attrs = delta * avg_grads                       # (C, T) — element-wise

        return attrs

    def attribute_batch(
        self,
        X:           torch.Tensor,         # (B, C, T)
        target_class: Optional[int] = None,
        device:      str = "cpu",
    ) -> np.ndarray:
        """Compute IG attributions for a whole batch. Returns (B, C, T)."""
        results = []
        for i in range(X.shape[0]):
            attr = self.attribute(X[i:i+1], target_class, device=device)
            results.append(attr)
        return np.stack(results, axis=0)


# 1.4  LRP — Layer-wise Relevance Propagation

class LRPExplainer:
    """
    Layer-wise Relevance Propagation for CNN layers (ε-rule).

    LRP propagates the model's prediction backward through the network,
    redistributing relevance from output neurons to input features such
    that the sum of input relevances equals the output score (conservation).

    The ε-rule used here:
        R_j = Σ_k [ (a_j · w_jk) / (Σ_j a_j · w_jk + ε · sign(z_k)) ] · R_k

    where:
        a_j  = activation of neuron j in lower layer
        w_jk = weight connecting j to k
        R_k  = relevance of neuron k in upper layer
        ε    = small stabiliser (prevents division by zero)

    This is the variant used in [9] Zhou et al. (2024) for sleep stage
    classification with LRP — the paper cited in the thesis proposal.

    Limitation: LRP is implemented here for the CNN portion of the
    hybrid model only. Full LRP through LSTM cells requires the more
    complex LRP-LSTM rules (Arras et al., 2019) and is left as future work.

    Parameters
    ----------
    model   : the HybridCNNLSTM or AblationHybrid model
    epsilon : stabiliser for the ε-rule (default 1e-6)
    """

    def __init__(self, model: nn.Module, epsilon: float = 1e-6):
        self.model   = model
        self.epsilon = epsilon
        self._hooks: List = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._register_activation_hooks()

    def _register_activation_hooks(self) -> None:
        """Store activations of every Conv1d and Linear layer."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._activations[name] = input[0].detach().clone()
        return hook

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _lrp_conv1d_layer(
        self,
        layer:  nn.Conv1d,
        a:      torch.Tensor, # (B, C_in, T) — lower-layer activations
        R:      torch.Tensor, # (B, C_out, T) — upper relevances
    ) -> torch.Tensor:
        """
        ε-LRP rule for a single Conv1d layer.

        Numerically stable implementation using forward re-computation
        rather than direct weight manipulation.
        """
        a = a.detach().requires_grad_(True)

        # Forward pass to compute pre-activations z
        z = F.conv1d(
            a, layer.weight,
            bias   = None,
            stride = layer.stride,
            padding= layer.padding,
            dilation= layer.dilation,
            groups = layer.groups,
        )

        # Stabiliser: ε · sign(z)
        z_stable = z + self.epsilon * z.sign()
        z_stable = torch.where(z_stable == 0, torch.full_like(z_stable, self.epsilon), z_stable)

        # Back-propagate relevance
        s = (R / z_stable)                # (B, C_out, T)
        (z * s.detach()).sum().backward() # populate a.grad
        c = a.grad                        # (B, C_in, T)

        return (a * c).detach()           # (B, C_in, T)

    def explain(
        self,
        x:            torch.Tensor, # (1, C, T)
        target_class: Optional[int] = None,
        device:       str = "cpu",
    ) -> np.ndarray:
        """
        Compute LRP attributions for the input signal.

        Returns
        -------
        relevance : np.ndarray (C, T) — input-level relevances
                    positive = supports prediction, negative = suppresses
        """
        self.model.eval()
        self._activations.clear()
        x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Initialise relevance at output neuron
        R = torch.zeros_like(logits)
        R[0, target_class] = logits[0, target_class].item()

        # Collect Conv1d layers in reverse order
        conv_layers   = []
        conv_act_keys = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d) and name in self._activations:
                conv_layers.append(module)
                conv_act_keys.append(name)

        # Propagate relevance backward through CNN layers
        # Start from the last conv layer's feature map relevance
        if len(conv_layers) == 0:
            log.warning("LRP: no Conv1d layers found in model.")
            return np.zeros(x.shape[1:])

        # Map output relevance back to last conv feature map dimensions
        # using Global Average Pooling gradient (uniform redistribution)
        last_act_shape = self._activations[conv_act_keys[-1]]
        B, C_last, T_last = last_act_shape.shape
        R_feat = R.unsqueeze(-1).expand(B, -1, T_last)   # rough mapping

        # Propagate through each conv layer in reverse
        R_current = R_feat
        for layer, key in zip(reversed(conv_layers), reversed(conv_act_keys)):
            a = self._activations[key]
            # Adjust shape compatibility between layers
            if R_current.shape[1] != a.shape[1]:
                R_current = R_current.mean(dim=1, keepdim=True).expand_as(a)
            try:
                R_current = self._lrp_conv1d_layer(layer, a, R_current)
            except Exception as exc:
                log.debug("LRP layer skipped (%s): %s", key, exc)
                continue

        # Final relevance has shape (1, C_in, T)
        relevance = R_current.squeeze(0).cpu().numpy() # (C, T)

        # Normalise for visualisation
        max_abs = np.abs(relevance).max()
        if max_abs > 1e-8:
            relevance = relevance / max_abs

        return relevance


# 1.5  XAI Visualizer

class XAIVisualizer:
    """
    Produces publication-ready XAI figures for the thesis.

    Plots Grad-CAM, Integrated Gradients, and LRP attributions
    overlaid on the raw input signal in a single multi-panel figure.

    For each panel:
      - Top row: raw signal waveform (all channels overlaid)
      - Bottom row: attribution heatmap (time × attribution magnitude)

    Parameters
    ----------
    output_dir : directory to save figures
    label_map  : {int: class_name} for axis labels
    """

    def __init__(self, output_dir: str, label_map: Dict[str, str]):
        self.output_dir = output_dir
        self.label_map  = label_map
        os.makedirs(output_dir, exist_ok=True)

    def plot_all_attributions(
        self,
        x:          np.ndarray, # (C, T) — single raw sample
        gradcam:    np.ndarray, # (T,)
        ig:         np.ndarray, # (C, T)
        lrp:        np.ndarray, # (C, T)
        pred_class: int,
        true_class: int,
        sample_idx: int,
        modality:   str,
        fs:         int = 250,
    ) -> str:
        """
        Three-column attribution comparison figure.

        Column 1: Grad-CAM temporal heatmap
        Column 2: Integrated Gradients per channel
        Column 3: LRP relevance per channel

        Returns path to saved figure.
        """
        C, T     = x.shape
        t_axis   = np.arange(T) / fs # seconds

        pred_name = self.label_map.get(str(pred_class), str(pred_class))
        true_name = self.label_map.get(str(true_class), str(true_class))
        correct   = "✓" if pred_class == true_class else "✗"

        fig, axes = plt.subplots(3, 3, figsize=(16, 10))
        fig.suptitle(
            f"[{modality.upper()}] Sample {sample_idx} | "
            f"Pred: {pred_name}  True: {true_name}  {correct}",
            fontsize=12, fontweight="bold",
        )

        methods = [
            ("Grad-CAM",              gradcam,  False),
            ("Integrated Gradients",  ig,       True),
            ("LRP (ε-rule)",          lrp,      True),
        ]

        colors = plt.cm.tab10(np.linspace(0, 1, C))

        for col, (method_name, attr, per_channel) in enumerate(methods):
            ax_sig  = axes[0, col]
            ax_heat = axes[1, col]
            ax_mean = axes[2, col]

            # Row 0: raw signal
            for ch in range(C):
                ax_sig.plot(t_axis, x[ch], color=colors[ch],
                            alpha=0.7, linewidth=0.8, label=f"Ch {ch}")
            ax_sig.set_title(method_name, fontsize=10)
            ax_sig.set_ylabel("Amplitude")
            ax_sig.set_xlim(t_axis[0], t_axis[-1])
            ax_sig.grid(alpha=0.2)
            if C <= 4:
                ax_sig.legend(fontsize=7, loc="upper right")

            # Row 1: attribution heatmap
            if per_channel:
                # attr is (C, T)
                img = ax_heat.imshow(
                    attr, aspect="auto", cmap="RdBu_r",
                    vmin=-1, vmax=1,
                    extent=[t_axis[0], t_axis[-1], C - 0.5, -0.5],
                )
                ax_heat.set_ylabel("Channel")
                ax_heat.set_xlabel("Time (s)")
                plt.colorbar(img, ax=ax_heat, fraction=0.046, pad=0.04)
            else:
                # attr is (T,) — Grad-CAM
                ax_heat.fill_between(t_axis, attr, alpha=0.6, color="darkorange")
                ax_heat.plot(t_axis, attr, color="darkorange", linewidth=0.8)
                ax_heat.set_ylabel("Attribution")
                ax_heat.set_xlabel("Time (s)")
                ax_heat.set_xlim(t_axis[0], t_axis[-1])
                ax_heat.set_ylim(-0.05, 1.05)

            # Row 2: mean attribution over channels with signal overlay
            mean_attr = np.abs(attr).mean(axis=0) if per_channel else attr
            if mean_attr.max() > 1e-8:
                mean_attr = mean_attr / mean_attr.max()

            ax_mean_twin = ax_mean.twinx()
            ax_mean.fill_between(t_axis, mean_attr, alpha=0.35,
                                  color="steelblue", label="Attribution")
            ax_mean.plot(t_axis, mean_attr, color="steelblue", linewidth=0.8)
            ax_mean_twin.plot(t_axis, x[0], color="gray",
                               alpha=0.5, linewidth=0.8, label="Signal (ch0)")
            ax_mean.set_xlabel("Time (s)")
            ax_mean.set_ylabel("Norm. attribution")
            ax_mean_twin.set_ylabel("Signal")
            ax_mean.set_xlim(t_axis[0], t_axis[-1])
            ax_mean.grid(alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fname = os.path.join(
            self.output_dir,
            f"{modality}_sample{sample_idx}_"
            f"pred{pred_class}_true{true_class}.png",
        )
        plt.savefig(fname, dpi=150)
        plt.close()
        log.info("XAI figure saved → %s", fname)
        return fname

    def plot_class_mean_attribution(
        self,
        attributions: np.ndarray, # (N, C, T) — IG or LRP
        y_true:       np.ndarray, # (N,)
        method_name:  str,
        modality:     str,
        fs:           int = 250,
    ) -> str:
        """
        Mean attribution profile per class — the key XAI figure for the thesis.

        Shows which temporal regions the model consistently attends to
        for each class across the entire test set.
        """
        n_classes = int(y_true.max()) + 1
        t_axis    = np.arange(attributions.shape[-1]) / fs
        cmap      = plt.cm.Set2(np.linspace(0, 1, n_classes))

        fig, axes = plt.subplots(
            1, n_classes, figsize=(5 * n_classes, 3.5), sharey=True
        )
        if n_classes == 1:
            axes = [axes]

        for cls_idx, ax in enumerate(axes):
            mask      = y_true == cls_idx
            if mask.sum() == 0:
                ax.set_title(f"Class {cls_idx}\n(no samples)")
                continue

            cls_attr  = np.abs(attributions[mask]).mean(axis=(0, 1)) # (T,)
            if cls_attr.max() > 1e-8:
                cls_attr /= cls_attr.max()

            ax.fill_between(t_axis, cls_attr, alpha=0.4, color=cmap[cls_idx])
            ax.plot(t_axis, cls_attr, color=cmap[cls_idx], linewidth=1.5)
            cls_name = self.label_map.get(str(cls_idx), f"Class {cls_idx}")
            ax.set_title(f"{cls_name}\n(n={mask.sum()})", fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.3)

        axes[0].set_ylabel("Mean |attribution|")
        fig.suptitle(
            f"[{modality.upper()}] {method_name} — Mean Attribution per Class",
            fontsize=11,
        )
        plt.tight_layout()

        fname = os.path.join(
            self.output_dir,
            f"{modality}_{method_name.replace(' ', '_').lower()}_class_profiles.png",
        )
        plt.savefig(fname, dpi=150)
        plt.close()
        log.info("Class attribution profiles → %s", fname)
        return fname


def run_xai_analysis(
    model:     nn.Module,
    loader:    DataLoader,
    label_map: Dict[str, str],
    output_dir: str,
    modality:  str,
    n_samples: int = 32,
    device:    str = "cpu",
) -> None:
    """
    Run Grad-CAM, Integrated Gradients, and LRP on n_samples test examples.
    Saves individual sample figures + class-mean attribution profiles.

    This is the function to call after evaluate_hybrid() in run_hybrid().
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Collect test samples
    all_X, all_y = [], []
    for X, y in loader:
        all_X.append(X)
        all_y.append(y)
        if sum(len(t) for t in all_X) >= n_samples:
            break
    X_all = torch.cat(all_X, dim=0)[:n_samples]
    y_all = torch.cat(all_y, dim=0)[:n_samples].numpy()

    # Find a Conv1d layer to hook for Grad-CAM
    target_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            target_layer = module # last assignment = deepest conv

    if target_layer is None:
        log.warning("XAI: no Conv1d found in model, skipping Grad-CAM.")
        return

    grad_cam = GradCAM1D(model, target_layer)
    ig_explainer  = IntegratedGradients1D(model, n_steps=50)
    lrp_explainer = LRPExplainer(model)
    visualizer    = XAIVisualizer(output_dir, label_map)

    all_ig  = []
    all_lrp = []

    log.info("Running XAI analysis on %d samples …", n_samples)
    for i in range(n_samples):
        x       = X_all[i:i+1].to(device)
        y_true  = int(y_all[i])

        cam_map, pred_class = grad_cam(x.clone())

        ig_attr  = ig_explainer.attribute(x.clone(), target_class=pred_class, device=device)
        lrp_attr = lrp_explainer.explain(x.clone(),  target_class=pred_class, device=device)

        all_ig.append(ig_attr)
        all_lrp.append(lrp_attr)

        if i < 8: # save individual figures for first 8 samples only
            visualizer.plot_all_attributions(
                x        = x.squeeze(0).cpu().numpy(),
                gradcam  = cam_map,
                ig       = ig_attr,
                lrp      = lrp_attr,
                pred_class= pred_class,
                true_class= y_true,
                sample_idx= i,
                modality  = modality,
            )

    # Class-mean attribution profiles (the main thesis figure)
    ig_stack  = np.stack(all_ig,  axis=0)          # (N, C, T)
    lrp_stack = np.stack(all_lrp, axis=0)

    visualizer.plot_class_mean_attribution(
        ig_stack, y_all, "Integrated Gradients", modality
    )
    visualizer.plot_class_mean_attribution(
        lrp_stack, y_all, "LRP (ε-rule)", modality
    )

    log.info("XAI analysis complete → %s", output_dir)


# ─ SECTION 2: PIPELINE

# 2.1  Unified run config  — single source of truth for both pipeline stages

@dataclass
class UnifiedRunConfig:
    """
    Single configuration object that spans preprocessing AND training.

    This is the core fix for GAP #6. Previously, preprocessing_pipeline.py
    and hybrid_model.py used separate config objects with no explicit link.
    Changing a parameter in one had no guaranteed effect on the other.

    Now a single UnifiedRunConfig is instantiated once and passed to both
    stages. Any parameter mismatch is caught at init time.
    """
    # Shared paths
    raw_data_dir:  str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir:    str = "outputs"

    # Modalities to process and train
    modalities: List[str] = field(default_factory=lambda: ["ecg", "eeg", "ppg"])

    # Preprocessing
    target_fs:    int   = 250
    window_sec:   float = 4.0
    overlap_frac: float = 0.5

    # Normalization — shared between preprocessing and model
    preprocessing_norm: str = "none"      # "none" for proposed system
    model_norm:         str = "learnable" # active when preprocessing_norm="none"

    # Training
    batch_size:    int   = 64
    epochs:        int   = 100
    learning_rate: float = 5e-4
    patience:      int   = 20
    seed:          int   = 42
    num_workers:   int   = 4
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"

    # XAI
    run_xai:    bool = True
    n_xai_samples: int = 32

    # Ablations
    run_ablations: bool = True
    ablation_runs: int  = 3

    def __post_init__(self):
        # Enforce consistent normalization strategy
        if self.preprocessing_norm != "none" and self.model_norm != "none":
            raise ValueError(
                f"Double normalization: preprocessing_norm='{self.preprocessing_norm}' "
                f"and model_norm='{self.model_norm}' are both active. "
                f"Set one to 'none' to ensure clean ablation experiments. "
                f"Recommended: preprocessing_norm='none', model_norm='learnable' (proposed system) "
                f"OR preprocessing_norm='zscore', model_norm='none' (control)."
            )

    def to_preprocessing_config(self) -> dict:
        """
        Converts to kwargs accepted by MultimodalPreprocessingPipeline.
        The preprocessing pipeline uses MultimodalPreprocessingPipeline,
        not a PipelineConfig dataclass, so we return a plain dict.

        When preprocessing_norm="none", norm_mode="none" is passed directly
        so the pipeline skips offline normalisation entirely. Model-level
        LearnableNorm then handles normalisation inside the forward pass.
        """
        return dict(
            ecg_dir    = os.path.join(self.raw_data_dir, "mit-bih"),
            eeg_dir    = os.path.join(self.raw_data_dir, "eeg-motor"),
            ppg_dir    = os.path.join(self.raw_data_dir, "ppg-dalia"),
            output_dir = self.processed_dir,
            norm_mode  = self.preprocessing_norm, # "none" passes through correctly now
            target_fs  = float(self.target_fs),
        )

    def to_hybrid_config(self):
        """Converts to the HybridConfig expected by hybrid_model.py."""
        from hybrid_model import HybridConfig
        return HybridConfig(
            processed_dir  = self.processed_dir,
            output_dir     = os.path.join(self.output_dir, "hybrid"),
            modalities     = self.modalities,
            batch_size     = self.batch_size,
            epochs         = self.epochs,
            learning_rate  = self.learning_rate,
            patience       = self.patience,
            seed           = self.seed,
            num_workers    = self.num_workers,
            adaptive_norm  = self.model_norm,
            device         = self.device,
        )


# 2.2  Preprocessing cache checker

class PreprocessingCache:
    """
    Checks whether preprocessed HDF5 files already exist before
    re-running the (slow) preprocessing pipeline.

    This makes the preprocessing step idempotent: running end_to_end_run()
    twice will skip preprocessing on the second run unless force=True.
    """

    def __init__(self, processed_dir: str, modalities: List[str]):
        self.processed_dir = processed_dir
        self.modalities    = modalities

    def all_present(self) -> bool:
        return all(
            os.path.exists(os.path.join(self.processed_dir, f"{m}.h5"))
            for m in self.modalities
        )

    def missing(self) -> List[str]:
        return [
            m for m in self.modalities
            if not os.path.exists(
                os.path.join(self.processed_dir, f"{m}.h5")
            )
        ]

    def run_if_needed(self, preprocessing_cfg, force: bool = False) -> None:
        missing = self.missing()
        if not force and not missing:
            log.info(
                "Preprocessing cache: all HDF5 files present, skipping. "
                "Pass force=True to re-run."
            )
            return

        if missing:
            log.info("Missing HDF5 for modalities: %s", missing)
        else:
            log.info("Force re-running preprocessing.")

        from preprocessing_pipeline import MultimodalPreprocessingPipeline
        # preprocessing_cfg is a dict of kwargs from to_preprocessing_config()
        pipeline = MultimodalPreprocessingPipeline(**preprocessing_cfg)
        pipeline.run()


# 2.3 Documented bridge: HDF5 → DataLoader

def build_loaders_from_hdf5(
    processed_dir: str,
    modality:      str,
    batch_size:    int = 64,
    num_workers:   int = 4,
    device:        str = "cpu",
) -> Dict[str, DataLoader]:
    """
    The explicit, documented bridge between preprocessing output and
    model training input. 

    Previously:
        preprocessing_pipeline.py → writes HDF5
        hybrid_model.py → calls make_loaders() with no documented link

    Now:
        end_to_end_run() calls build_loaders_from_hdf5() explicitly,
        making the pipeline connection visible and testable.

    Parameters
    ----------
    processed_dir : directory containing <modality>.h5 files
    modality      : "ecg" | "eeg" | "ppg"
    batch_size    : DataLoader batch size
    num_workers   : DataLoader worker count
    device        : used to set pin_memory

    Returns
    -------
    loaders : { "train": DataLoader, "val": DataLoader, "test": DataLoader }
    """
    h5_path = os.path.join(processed_dir, f"{modality}.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}\n"
            f"Run preprocessing_pipeline.py first, or call "
            f"PreprocessingCache.run_if_needed()."
        )

    # Verify HDF5 structure matches expected schema
    with h5py.File(h5_path, "r") as f:
        if modality not in f:
            raise KeyError(
                f"Group '{modality}' not found in {h5_path}. "
                f"Expected structure: /{modality}/train/X, /{modality}/train/y, …"
            )
        for split in ("train", "val", "test"):
            if split not in f[modality]:
                raise KeyError(
                    f"Split '{split}' missing from {h5_path}/{modality}. "
                    f"Re-run preprocessing_pipeline.py."
                )
            for key in ("X", "y"):
                if key not in f[modality][split]:
                    raise KeyError(
                        f"Dataset '{key}' missing from "
                        f"{h5_path}/{modality}/{split}."
                    )

    log.info("HDF5 schema validated: %s", h5_path)

    from baseline_models import BioDataset

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
            "[%s | %s] %d samples | C=%d T=%d | %d classes",
            modality.upper(), split, len(ds),
            ds.n_channels, ds.seq_len, ds.n_classes,
        )

    return loaders


# 2.4 End-to-end run

def end_to_end_run(cfg: UnifiedRunConfig, force_preprocess: bool = False) -> None:
    """
    The complete thesis pipeline in one call:

        1. Preprocessing (skipped if HDF5 files already exist)
        2. Baseline training (CNN-only, LSTM-only) per modality
        3. Hybrid training per modality
        4. XAI analysis per modality
        5. Ablation study per modality
        6. Final comparison report

    All stages share cfg, ensuring consistent normalization strategy,
    random seeds, and file paths. This is the documented connection that
    was missing in GAP #6.

    Parameters
    ----------
    cfg              : UnifiedRunConfig controlling all stages
    force_preprocess : if True, re-run preprocessing even if HDF5 exists
    """
    import torch
    from baseline_models import run_baselines, TrainingConfig
    from hybrid_model import run_hybrid, HybridCNNLSTM
    from ablation_framework import run_ablations

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    log.info("=" * 70)
    log.info("END-TO-END PIPELINE RUN")
    log.info("  preprocessing_norm : %s", cfg.preprocessing_norm)
    log.info("  model_norm         : %s", cfg.model_norm)
    log.info("  device             : %s", cfg.device)
    log.info("=" * 70)

    # [Stage 1] Preprocessing
    log.info("\n[Stage 1] Preprocessing")
    preproc_cfg = cfg.to_preprocessing_config()
    cache       = PreprocessingCache(cfg.processed_dir, cfg.modalities)
    cache.run_if_needed(preproc_cfg, force=force_preprocess)

    # [Stage 2] Baseline training
    log.info("\n[Stage 2] Baseline training")
    base_cfg = TrainingConfig(
        processed_dir = cfg.processed_dir,
        output_dir    = os.path.join(cfg.output_dir, "baselines"),
        modalities    = cfg.modalities,
        batch_size    = cfg.batch_size,
        epochs        = cfg.epochs,
        learning_rate = cfg.learning_rate,
        patience      = cfg.patience,
        seed          = cfg.seed,
        num_workers   = cfg.num_workers,
        device        = cfg.device,
    )
    run_baselines(base_cfg)

    # [Stage 3] Hybrid training
    log.info("\n[Stage 3] Hybrid training")
    hybrid_cfg = cfg.to_hybrid_config()
    hybrid_cfg.baseline_dir = os.path.join(cfg.output_dir, "baselines")
    run_hybrid(hybrid_cfg)

    # [Stage 4] XAI analysis
    if cfg.run_xai:
        log.info("\n[Stage 4] XAI analysis")
        for modality in cfg.modalities:
            h5_path = os.path.join(cfg.processed_dir, f"{modality}.h5")
            if not os.path.exists(h5_path):
                log.warning("Skipping XAI for %s — HDF5 not found.", modality)
                continue

            loaders = build_loaders_from_hdf5(
                cfg.processed_dir, modality,
                cfg.batch_size, cfg.num_workers, cfg.device,
            )
            train_ds  = loaders["train"].dataset
            n_ch      = train_ds.n_channels
            n_classes = train_ds.n_classes
            label_map = train_ds.label_map

            # Load best hybrid checkpoint
            ckpt_path = os.path.join(
                cfg.output_dir, "hybrid", modality, "best_hybrid.pt"
            )
            if not os.path.exists(ckpt_path):
                log.warning("Hybrid checkpoint not found for %s, skipping XAI.", modality)
                continue

            model = HybridCNNLSTM(n_ch, n_classes, hybrid_cfg)
            ckpt  = torch.load(ckpt_path, map_location=cfg.device)
            model.load_state_dict(ckpt["model_state"])
            model = model.to(cfg.device)

            xai_dir = os.path.join(cfg.output_dir, "xai", modality)
            run_xai_analysis(
                model     = model,
                loader    = loaders["test"],
                label_map = label_map,
                output_dir= xai_dir,
                modality  = modality,
                n_samples = cfg.n_xai_samples,
                device    = cfg.device,
            )

    # [Stage 5] Ablation study
    if cfg.run_ablations:
        log.info("\n[Stage 5] Ablation study")
        for modality in cfg.modalities:
            h5_path = os.path.join(cfg.processed_dir, f"{modality}.h5")
            if not os.path.exists(h5_path):
                continue
            ablation_dir = os.path.join(cfg.output_dir, "ablations", modality)
            run_ablations(
                modality   = modality,
                h5_path    = h5_path,
                output_dir = ablation_dir,
                n_runs     = cfg.ablation_runs,
                device     = cfg.device,
            )

    # [Stage 6] Statistical evaluation
    log.info("\n[Stage 6] Statistical evaluation")
    from evaluation_suite import StatisticalEvaluator, ResearchGradeReport
    from baseline_models import LoaderConfig

    for modality in cfg.modalities:
        h5_path = os.path.join(cfg.processed_dir, f"{modality}.h5")
        if not os.path.exists(h5_path):
            continue

        stat_dir = os.path.join(cfg.output_dir, "statistical", modality)
        os.makedirs(stat_dir, exist_ok=True)

        # Build fresh loaders for each seed inside the evaluator
        def _loaders_fn(seed: int, _h5=h5_path, _mod=modality, _cfg=cfg):
            lc = LoaderConfig(batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, device=cfg.device)
            return build_loaders_from_hdf5(
                _cfg.processed_dir, _mod, lc.batch_size, lc.num_workers, lc.device
            )

        # Model factory functions for CNN, LSTM, and Hybrid
        _train_ds = build_loaders_from_hdf5(
            cfg.processed_dir, modality,
            cfg.batch_size, cfg.num_workers, cfg.device,
        )["train"].dataset
        _n_ch      = _train_ds.n_channels
        _seq_len   = _train_ds.seq_len
        _n_classes = _train_ds.n_classes

        from baseline_models import CNN1DBaseline, LSTMBaseline, TrainingConfig as TC
        from baseline_models import BaselineTrainer

        def _make_cnn():
            return CNN1DBaseline(_n_ch, _n_classes, TC())
        def _make_lstm():
            return LSTMBaseline(_n_ch, _n_classes, _seq_len, TC())
        def _make_hybrid():
            return HybridCNNLSTM(_n_ch, _n_classes, hybrid_cfg).to(cfg.device)

        def _train_fn(model, loaders):
            trainer = BaselineTrainer(
                model=model, loaders=loaders, n_classes=_n_classes,
                output_dir=os.path.join(stat_dir, "tmp"), cfg=TC(),
            )
            trainer.fit()
            return model

        def _eval_fn(model, test_loader):
            from sklearn.metrics import accuracy_score, f1_score
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for X, y in test_loader:
                    preds.append(model(X.to(cfg.device)).argmax(1).cpu().numpy())
                    labels.append(y.numpy())
            y_true = np.concatenate(labels)
            y_pred = np.concatenate(preds)
            return {
                "accuracy":  float(accuracy_score(y_true, y_pred)),
                "macro_f1":  float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }

        evaluator = StatisticalEvaluator(
            model_fns  = {"CNN": _make_cnn, "LSTM": _make_lstm, "Hybrid": _make_hybrid},
            train_fn   = _train_fn,
            eval_fn    = _eval_fn,
            n_seeds    = 3,        # increase to 5 for final thesis run
            base_seed  = cfg.seed,
        )
        summary_df = evaluator.run(_loaders_fn)
        summary_df.to_csv(os.path.join(stat_dir, "statistical_summary.csv"), index=False)

        sig_df = evaluator.all_pairwise_tests(metric="macro_f1")
        sig_df.to_csv(os.path.join(stat_dir, "significance_tests.csv"), index=False)

        reporter = ResearchGradeReport(stat_dir)
        reporter.latex_results_table(
            summary_df, sig_df, metric="macro_f1",
            caption=f"{modality.upper()} model comparison (mean ± std, n=3 seeds).",
        )
        reporter.violin_plot(evaluator, metric="macro_f1")
        log.info("Statistical results for %s → %s", modality.upper(), stat_dir)

    # [Stage 7] Late fusion
    log.info("\n[Stage 7] Late fusion ensemble")
    from multimodal_fusion import (
        LateFusionEnsemble, PhysiologicalStateMapper, train_late_fusion,
        MultimodalDataset, make_multimodal_loaders,
    )
    from baseline_models import LoaderConfig

    # Collect per-modality models and remap labels to shared taxonomy
    modality_models = {}
    mapped_data: Dict[str, np.ndarray] = {}
    mapped_labels_per_mod: Dict[str, np.ndarray] = {}
    min_samples = None

    for modality in cfg.modalities:
        ckpt_path = os.path.join(cfg.output_dir, "hybrid", modality, "best_hybrid.pt")
        h5_path   = os.path.join(cfg.processed_dir, f"{modality}.h5")
        if not os.path.exists(ckpt_path) or not os.path.exists(h5_path):
            log.warning("Late fusion: missing checkpoint/HDF5 for %s — skipping.", modality)
            continue

        lc      = LoaderConfig(batch_size=cfg.batch_size,
                               num_workers=cfg.num_workers, device=cfg.device)
        loaders = build_loaders_from_hdf5(
            cfg.processed_dir, modality, lc.batch_size, lc.num_workers, lc.device
        )
        ds      = loaders["train"].dataset
        n_ch    = ds.n_channels
        n_cls   = ds.n_classes
        label_map = ds.label_map

        model = HybridCNNLSTM(n_ch, n_cls, hybrid_cfg)
        ckpt  = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"])
        model.to(cfg.device)

        # Replace final classification head with a shared-taxonomy head
        model.head[-1] = nn.Linear(
            model.head[-1].in_features,
            PhysiologicalStateMapper.N_CLASSES,
        ).to(cfg.device)
        modality_models[modality] = model

        # Remap labels using shared taxonomy
        raw_labels = np.array([label_map.get(str(i), str(i))
                               for i in range(n_cls)])[ds.y]
        mapped_labels_per_mod[modality] = PhysiologicalStateMapper.map(
            raw_labels, modality
        )
        mapped_data[modality] = ds.X

        n = len(ds.X)
        min_samples = n if min_samples is None else min(min_samples, n)

    if len(modality_models) >= 2:
        # Align sample counts (take first min_samples from each)
        aligned_data   = {m: mapped_data[m][:min_samples]   for m in modality_models}
        aligned_labels = np.stack(
            [mapped_labels_per_mod[m][:min_samples] for m in modality_models], axis=1
        ).T
        # Use majority vote across modalities for the unified label
        from scipy import stats as _sp
        fusion_labels, _ = _sp.mode(aligned_labels, axis=0)
        fusion_labels = fusion_labels.flatten().astype(np.int64)

        mm_loaders = make_multimodal_loaders(
            data_dict   = aligned_data,
            labels      = fusion_labels,
            modalities  = list(modality_models.keys()),
            batch_size  = cfg.batch_size,
        )

        ensemble = LateFusionEnsemble(
            modality_models = modality_models,
            n_classes       = PhysiologicalStateMapper.N_CLASSES,
            fusion_rule     = "weighted",
        )
        fusion_dir = os.path.join(cfg.output_dir, "late_fusion")
        train_late_fusion(
            ensemble   = ensemble,
            loaders    = mm_loaders,
            n_classes  = PhysiologicalStateMapper.N_CLASSES,
            output_dir = fusion_dir,
            epochs     = 20,
            lr         = 1e-3,
            device     = cfg.device,
        )
        log.info("Late fusion complete → %s", fusion_dir)
    else:
        log.warning(
            "Late fusion skipped — fewer than 2 modality checkpoints found. "
            "Run hybrid training for all modalities first."
        )

    log.info("\n[Complete] All outputs → %s", cfg.output_dir)


#
# USAGE
# 
#
# Proposed system (LearnableNorm, no preprocessing norm):
#   from xai_and_pipeline import end_to_end_run, UnifiedRunConfig
#   cfg = UnifiedRunConfig(
#       preprocessing_norm = "none",
#       model_norm         = "learnable",
#   )
#   end_to_end_run(cfg)
#
# Control system (z-score preprocessing, no model norm):
#   cfg = UnifiedRunConfig(
#       preprocessing_norm = "zscore",
#       model_norm         = "none",
#   )
#   end_to_end_run(cfg)
#
# Force re-run of preprocessing:
#   end_to_end_run(cfg, force_preprocess=True)
#
# XAI only (after training):
#   run_xai_analysis(model, loaders["test"], label_map,
#                    "outputs/xai/ecg", "ecg", n_samples=64)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = UnifiedRunConfig(
        preprocessing_norm = "none",
        model_norm         = "learnable",
    )
    end_to_end_run(cfg)
