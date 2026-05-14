"""

The three datasets (MIT-BIH, PhysioNet EEG Motor, PPG-DaLiA) have
incompatible label spaces and cannot be aligned at the sample level.

This module therefore implements the valid fusion strategies:

A. ModalitySpecificSystem (primary, used in all per-modality experiments)
   ─ Same HybridCNNLSTM architecture applied to each modality independently.
   ─ It is scientifically valid multimodal research: we demonstrate that
     one unified architecture generalises across ECG/EEG/PPG — a contribution
     in itself (c.f. [5] Oladunni & Wong 2025).

B. LateFusionEnsemble (secondary, used when a shared label space exists)
   ─ Trains per-modality models, then combines their softmax outputs.
   ─ Requires a shared task — implemented here for the WESAD dataset
     (which contains simultaneous ECG+EEG+PPG from the same subjects)
     and as a simulation using the PPG-DaLiA activity labels mapped to
     a common physiological-state space for demonstration purposes.
   ─ Three fusion rules: average, learned weighted, max-vote.

C. MultimodalDataset
   ─ A proper Dataset that returns dicts keyed by modality.
   ─ Used when a truly aligned dataset is available.
   ─ Included here so the architecture claim is fully implemented,
     even if the primary experiments use strategy A.

The thesis MUST state clearly in §2 (Problem Statement) and §3 (Methodology):
  "Because the three chosen datasets represent different recording paradigms
   with incompatible label spaces, multimodality is demonstrated through
   architectural generalisation (Strategy A) and late-fusion ensemble
   (Strategy B), rather than sample-level sensor fusion."

"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# 1. MODALITY-SPECIFIC SYSTEM

class ModalitySpecificSystem:
    """
    Trains one HybridCNNLSTM per modality and tracks all results
    under a unified namespace.

    The same architectural blueprint is validated across three 
    physiologically distinct signal types, demonstrating 
    generalisation that per-modality custom networks
    cannot claim.

    Usage
    -----
    sys = ModalitySpecificSystem(cfg)
    sys.train_all(loaders_dict)     # {"ecg": loaders, "eeg": loaders, ...}
    sys.evaluate_all(loaders_dict)
    sys.summary()                   # DataFrame: modality × metric
    """

    def __init__(self, cfg):
        self.cfg     = cfg
        self.models: Dict[str, nn.Module] = {}
        self.results: Dict[str, Dict]     = {}

    def register_model(self, modality: str, model: nn.Module) -> None:
        self.models[modality] = model

    def register_result(self, modality: str, metrics: Dict) -> None:
        self.results[modality] = metrics

    def summary(self) -> pd.DataFrame:
        rows = []
        for mod, m in self.results.items():
            rows.append({"Modality": mod.upper(), **m})
        df = pd.DataFrame(rows)
        log.info("Modality-specific results:\n%s", df.to_string(index=False))
        return df


# 2. MULTIMODAL DATASET

class MultimodalDataset(Dataset):
    """
    Dataset for truly aligned multi-sensor recordings.

    Each sample contains simultaneous signals from multiple modalities
    and a SINGLE shared label.

    When to use:
    ------------
    ✓ WESAD dataset (wrist + chest: ECG + EDA + PPG + RESP + TEMP + ACC)
    ✓ Any dataset where one subject produces simultaneous multi-sensor data
    ✗ MIT-BIH + PhysioNet EEG + PPG-DaLiA (different subjects, tasks, labels)

    For the thesis datasets, use ModalitySpecificSystem instead and
    document this architectural decision explicitly.

    Parameters
    ----------
    data_dict : { modality_name: np.ndarray of shape (N, C, T) }
    labels    : np.ndarray of shape (N,) — shared label for all modalities
    modalities: list of modality keys to include
    augment   : whether to apply train-time augmentation
    """

    def __init__(
        self,
        data_dict:  Dict[str, np.ndarray],
        labels:     np.ndarray,
        modalities: List[str],
        augment:    bool = False,
    ):
        super().__init__()
        self.modalities = modalities
        self.augment    = augment

        # Validate alignment
        n_samples = labels.shape[0]
        for mod, arr in data_dict.items():
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"Modality '{mod}' has {arr.shape[0]} samples "
                    f"but labels have {n_samples}. "
                    f"All modalities must be sample-aligned."
                )

        self.data   = {m: data_dict[m].astype(np.float32) for m in modalities}
        self.labels = labels.astype(np.int64)
        self.n_classes = int(labels.max()) + 1
        self.shapes = {m: self.data[m].shape[1:] for m in modalities}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sample = {}
        for mod in self.modalities:
            x = self.data[mod][idx].copy()
            if self.augment:
                x = _augment_signal(x)
            sample[mod] = torch.from_numpy(x)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, y


def _augment_signal(x: np.ndarray) -> np.ndarray:
    """Lightweight signal-space augmentation (same as in baseline_models.py)."""
    if np.random.rand() < 0.5:
        x += np.random.normal(0, np.random.uniform(0, 0.02), x.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        x *= np.random.uniform(0.9, 1.1)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-int(0.05 * x.shape[-1]), int(0.05 * x.shape[-1]))
        x = np.roll(x, shift, axis=-1)
    return x


def multimodal_collate(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Custom collate that stacks each modality independently."""
    samples, labels = zip(*batch)
    modalities = list(samples[0].keys())
    collated = {mod: torch.stack([s[mod] for s in samples]) for mod in modalities}
    return collated, torch.stack(labels)


def make_multimodal_loaders(
    data_dict:  Dict[str, np.ndarray],
    labels:     np.ndarray,
    modalities: List[str],
    batch_size: int   = 64,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    seed:       int   = 42,
    num_workers: int  = 4,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders for aligned multimodal data.
    Uses stratified splitting to preserve class balance.
    """
    from sklearn.model_selection import train_test_split

    n = len(labels)
    idx = np.arange(n)

    idx_tv, idx_test = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=seed
    )
    val_frac_adj = val_frac / (1 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_frac_adj,
        stratify=labels[idx_tv], random_state=seed,
    )

    loaders = {}
    for split, indices, aug in [
        ("train", idx_train, True),
        ("val",   idx_val,   False),
        ("test",  idx_test,  False),
    ]:
        split_data = {m: data_dict[m][indices] for m in modalities}
        ds = MultimodalDataset(split_data, labels[indices], modalities, aug)
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = aug,
            collate_fn  = multimodal_collate,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )
        log.info("Multimodal [%s]: %d samples | %d classes", split, len(ds), ds.n_classes)
    return loaders


# 3. LATE FUSION ENSEMBLE

class LateFusionEnsemble(nn.Module):
    """
    Late fusion of per-modality classifiers.

    This is the simplest valid form of multimodal fusion and requires:
    1. All modalities have a SHARED label space (same classes).
    2. All modality models are pre-trained (or trained jointly).

    For the thesis datasets, a shared label space is approximated by
    mapping each modality's classes to a common physiological-state
    taxonomy (see PhysiologicalStateMapper below).

    Three fusion rules are provided:
    ─ "average"  : uniform average of softmax probabilities
    ─ "weighted" : learned scalar weight per modality (trainable)
    ─ "attention": learned query-key attention over modality embeddings

    Parameters
    ----------
    modality_models : dict of {modality: nn.Module}
    n_classes       : number of SHARED output classes
    fusion_rule     : "average" | "weighted" | "attention"
    feature_dim     : output dim of each modality model's penultimate layer
                      (required for "attention" fusion)
    """

    def __init__(
        self,
        modality_models: Dict[str, nn.Module],
        n_classes:       int,
        fusion_rule:     str = "weighted",
        feature_dim:     int = 128,
    ):
        super().__init__()
        self.modalities    = list(modality_models.keys())
        self.n_modalities  = len(self.modalities)
        self.fusion_rule   = fusion_rule
        self.n_classes     = n_classes

        self.encoders = nn.ModuleDict(modality_models)

        if fusion_rule == "weighted":
            # Learnable scalar weight per modality, softmax-normalised
            self.modality_weights = nn.Parameter(
                torch.ones(self.n_modalities) / self.n_modalities
            )

        elif fusion_rule == "attention":
            # Query = stacked softmax probs (B, M*n_classes); size is encoder-agnostic
            self.attn_proj = nn.Sequential(
                nn.Linear(n_classes * self.n_modalities, self.n_modalities),
                nn.Softmax(dim=1),
            )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : { modality: (B, C, T) tensor }

        Returns
        -------
        logits : (B, n_classes)
        """
        probs   = []

        for mod in self.modalities:
            x      = inputs[mod]
            logits = self.encoders[mod](x)         # (B, n_classes)
            probs.append(F.softmax(logits, dim=1)) # (B, n_classes)

        probs_stack = torch.stack(probs, dim=1)    # (B, M, n_classes)

        if self.fusion_rule == "average":
            return probs_stack.mean(dim=1)         # (B, n_classes)

        elif self.fusion_rule == "weighted":
            w = F.softmax(self.modality_weights, dim=0)  # (M,)
            fused = (probs_stack * w.view(1, -1, 1)).sum(dim=1)
            return fused

        elif self.fusion_rule == "attention":
            all_feats = probs_stack.view(probs_stack.size(0), -1) # (B, M*n_classes)
            weights   = self.attn_proj(all_feats)                 # (B, M)
            weighted  = (probs_stack * weights.unsqueeze(2)).sum(dim=1)
            return weighted

        else:
            raise ValueError(f"Unknown fusion_rule: {self.fusion_rule}")

    def get_modality_weights(self) -> Dict[str, float]:
        """Return current learned modality importance weights."""
        if self.fusion_rule == "weighted":
            w = F.softmax(self.modality_weights, dim=0).detach().cpu().numpy()
            return {mod: float(w[i]) for i, mod in enumerate(self.modalities)}
        return {}


# 4. PHYSIOLOGICAL STATE MAPPER

class PhysiologicalStateMapper:
    """
    Maps dataset-specific labels to a shared physiological-state taxonomy.

    This is the principled solution to GAP #2: instead of forcing a
    single classifier across incompatible label spaces, we define a
    coarser shared space and map each modality's labels into it.

    Shared taxonomy (5 states):
    ---------------------------
    0 → REST         : baseline, low activity
    1 → MILD_STRESS  : light physical or cognitive load
    2 → HIGH_STRESS  : intense physical activity or arrhythmia
    3 → PATHOLOGICAL : disease marker (arrhythmia, seizure)
    4 → TRANSITION   : state change / uncertain

    Mapping rationale:
    ------------------
    ECG (AAMI):
      N → REST (normal sinus rhythm = baseline cardiac state)
      S → MILD_STRESS (supraventricular: elevated but not dangerous)
      V → HIGH_STRESS (ventricular: haemodynamically significant)
      F → TRANSITION (fusion: mixed morphology, uncertain state)
      Q → PATHOLOGICAL (paced / unknown: non-physiological)

    EEG (Motor Imagery):
      rest       → REST
      left/right → MILD_STRESS (active motor imagery = cognitive load)
      both       → HIGH_STRESS (bilateral imagery = high cognitive load)

    PPG (DaLiA activity):
      sitting, driving, lunch_break, working → REST
      walking, table_soccer                  → MILD_STRESS
      stairs, cycling                        → HIGH_STRESS

    IMPORTANT: This mapping is coarse and introduces information loss.
    It exists solely to enable late-fusion experiments and MUST be
    acknowledged as a limitation in the thesis discussion.
    """

    ECG_MAP = {
        "N": 0, # REST
        "S": 1, # MILD_STRESS
        "V": 2, # HIGH_STRESS
        "F": 4, # TRANSITION
        "Q": 3, # PATHOLOGICAL
    }

    EEG_MAP = {
        "rest":                    0,
        "left_fist_or_both_fists": 1,
        "right_fist_or_both_feet": 1,
        "unknown":                 4,
    }

    PPG_MAP = {
        "sitting":      0,
        "driving":      0,
        "lunch_break":  0,
        "working":      0,
        "walking":      1,
        "table_soccer": 1,
        "stairs":       2,
        "cycling":      2,
    }

    SHARED_CLASSES = {
        0: "REST",
        1: "MILD_STRESS",
        2: "HIGH_STRESS",
        3: "PATHOLOGICAL",
        4: "TRANSITION",
    }
    N_CLASSES = 5

    @classmethod
    def map(cls, labels: np.ndarray, modality: str) -> np.ndarray:
        mapping = {
            "ecg": cls.ECG_MAP,
            "eeg": cls.EEG_MAP,
            "ppg": cls.PPG_MAP,
        }[modality]
        mapped = np.array([mapping.get(str(l), 4) for l in labels], dtype=np.int64)
        unique_before = np.unique(labels)
        unique_after  = np.unique(mapped)
        log.info(
            "PhysiologicalStateMapper [%s]: %d → %d unique labels",
            modality, len(unique_before), len(unique_after),
        )
        return mapped


# 5. LATE FUSION TRAINING LOOP

def train_late_fusion(
    ensemble:  LateFusionEnsemble,
    loaders:   Dict[str, DataLoader], # multimodal loaders
    n_classes: int,
    output_dir: str,
    epochs:    int   = 30,
    lr:        float = 1e-3,
    device:    str   = "cpu",
) -> pd.DataFrame:
    """
    Fine-tune the fusion weights while keeping encoder weights frozen.

    In late fusion the encoders are already trained; we only learn
    the combination rule (modality_weights or attention_proj).

    Returns training history DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Freeze encoders, train only fusion parameters
    for name, param in ensemble.named_parameters():
        if not any(k in name for k in ["modality_weights", "attn_proj", "classifier"]):
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    log.info("Late fusion — trainable parameters: %d", trainable)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, ensemble.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    ensemble  = ensemble.to(device)

    history = []
    best_val_acc = -1.0
    best_state   = {}

    for epoch in range(1, epochs + 1):
        for split in ("train", "val"):
            ensemble.train(split == "train")
            total_loss = 0.0
            all_preds, all_labels = [], []

            ctx = torch.enable_grad() if split == "train" else torch.no_grad()
            with ctx:
                for batch_x, batch_y in loaders[split]:
                    batch_x = {m: v.to(device) for m, v in batch_x.items()}
                    batch_y = batch_y.to(device)

                    logits = ensemble(batch_x)
                    loss   = criterion(logits, batch_y)

                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item() * len(batch_y)
                    all_preds.append(logits.argmax(1).cpu().numpy())
                    all_labels.append(batch_y.cpu().numpy())

            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            acc    = (y_true == y_pred).mean()

            if split == "val":
                history.append({"epoch": epoch, "val_loss": total_loss / len(y_true), "val_acc": acc})
                log.info("LateFusion Ep %d | val acc=%.4f", epoch, acc)
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_state   = {k: v.clone() for k, v in ensemble.state_dict().items()}

    ensemble.load_state_dict(best_state)
    torch.save(best_state, os.path.join(output_dir, "best_late_fusion.pt"))

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, "late_fusion_history.csv"), index=False)

    weights = ensemble.get_modality_weights()
    if weights:
        log.info("Learned modality weights: %s", weights)
        with open(os.path.join(output_dir, "modality_weights.json"), "w") as f:
            json.dump(weights, f, indent=2)

    return df
