# Multimodal Biomedical Signal Classification
### Hybrid CNN–LSTM with Adaptive Normalization and Explainable AI

> Bachelor of Engineering Thesis — Electrical and Electronics Engineering  
> Datasets: MIT-BIH Arrhythmia (ECG) · PhysioNet EEG Motor Movement · PPG-DaLiA

---

## Overview

This repository implements a **hybrid CNN–LSTM deep-learning framework** for classifying biomedical signals across three physiologically distinct modalities.

The architecture addresses three limitations of prior single-modality work:

| Limitation | Solution |
|---|---|
| CNNs discard temporal context after pooling | CNN output is passed as a feature *sequence* to the LSTM |
| LSTMs receive raw samples as features | Multi-scale CNN pre-processes signals into morphological features first |
| Static normalization ignores inter-subject variability | `LearnableNorm` predicts per-window γ/β from signal statistics via a hyper-network |

A full **explainability module** (Grad-CAM, Integrated Gradients, LRP ε-rule) produces attribution maps over the time axis, and a **deployment module** measures latency, FLOPs, and quantization impact.

---

## Repository structure

```
.
├── main.py                        Entry point — CLI + config.yaml loader
├── config.yaml                    All hyperparameters (replaces CLI flags)
├── requirements.txt               Pinned dependencies
│
├── src/                           Source package
│   ├── preprocessing_pipeline.py  ECG, EEG, PPG loading, filtering, HDF5 export
│   ├── baseline_models.py         CNN-only and LSTM-only baselines
│   ├── hybrid_model.py            Multi-scale CNN–LSTM with LearnableNorm + gated fusion
│   ├── multimodal_fusion.py       Late-fusion ensemble + MultimodalDataset
│   ├── ablation_framework.py      10-config ablation matrix + AblationRunner
│   ├── xai_and_pipeline.py        Grad-CAM, IG, LRP + UnifiedRunConfig + end_to_end_run
│   └── evaluation_suite.py        Subject-wise splits, Wilcoxon tests, latency profiling
│
├── notebooks/
│   └── thesis_pipeline.ipynb      Jupyter walkthrough — run everything from one place
│
├── tests/
│   └── verify_patches.py          Three self-contained correctness checks (no framework)
│
├── scripts/
│   ├── download_data.sh           Automated dataset download via wfdb + wget
│   └── run_experiments.sh         Sequential proposed + control + deployment runs
│
├── docs/
│   ├── data_setup.md              Detailed data layout instructions with diagrams
│   └── architecture.md            Model architecture deep-dive
│
└── data/
    ├── raw/
    │   ├── mit-bih/               MIT-BIH .dat/.hea/.atr files (flat)
    │   ├── eeg-motor/             PhysioNet EEG S001/–S109/ subdirs
    │   └── ppg-dalia/             PPG-DaLiA S1.pkl–S15.pkl (flat)
    └── processed/                 HDF5 outputs from preprocessing (auto-created)
```

---

## Architecture

```
Input (B, C, T)
    │
    ├─── Multi-Scale CNN Encoder ──────────────────────────────────────────┐
    │     ├─ Branch k=3   [LearnableNorm → Conv → ResBlocks(SE) → Pool]    │
    │     ├─ Branch k=7   [LearnableNorm → Conv → ResBlocks(SE) → Pool]    │
    │     ├─ Branch k=15  [LearnableNorm → Conv → ResBlocks(SE) → Pool]    │
    │     └─ Branch k=31  [LearnableNorm → Conv → ResBlocks(SE) → Pool]    │
    │           ↓ concat → (B, 128, 64)                                    │
    │           │                                                          │
    │           ├──→ GlobalAvgPool → f_cnn (B, 128) ───────────────────────┤
    │           └──→ permute (B, 64, 128)                                  │
    │                     ↓                                                │
    ├─── BiLSTM Encoder ───────────────────────────────────────────────────┤
    │     ├─ BiLSTM (hidden=192, layers=2)                                 │
    │     └─ Multi-head temporal attention (heads=4)                       │
    │           ↓ → f_lstm (B, 384)                                        │
    │                                                                      │
    ├─── Gated Fusion ─────────────────────────────────────────────────────┘
    │     g = σ(W · [f_cnn ; f_lstm])
    │     f = g ⊙ proj(f_cnn) + (1−g) ⊙ proj(f_lstm)
    │           ↓ → f_fused (B, 256)
    │
    └─── Classification Head
          Linear(256→128) → BN → ReLU → Dropout → Linear(128→n_classes)
```

**LearnableNorm** (the core novel component):
```
μ, σ = per-channel mean and std of the current window
γ, β = MLP(concat[μ, σ])     # hyper-network, 2-layer
x_norm = (x − μ) / σ
output  = γ · x_norm + β
```
This makes normalization *input-conditioned* rather than globally fixed, adapting to inter-subject amplitude variation and sensor drift without requiring calibration data.

---

## Quickstart

### 1 · Install

```bash
git clone https://github.com/GiorgosZafeiratos/bachelor-thesis.git
cd bachelor-thesis
pip install -r requirements.txt
```

### 2 · Download data

See [`docs/data_setup.md`](docs/data_setup.md) for full instructions.  
The automated script handles MIT-BIH via `wfdb` and prints manual download links for EEG and PPG:

```bash
bash scripts/download_data.sh
```

### 3 · Verify setup

```bash
python tests/verify_patches.py
```

All three lines should print `PASS`.

### 4 · Run

**Via config file (recommended):**
```bash
python main.py --config config.yaml
```

**Via CLI flags:**
```bash
# Proposed system (LearnableNorm — novel contribution)
python main.py --norm-strategy proposed --modalities ecg eeg ppg

# Control system (z-score baseline for ablation A0 vs A1)
python main.py --norm-strategy control --output-dir outputs_control

# ECG-only fast-track (~2–3 hours on CPU, for pipeline validation)
python main.py --modalities ecg --epochs 30 --no-ablations --output-dir outputs_fast

# With deployment analysis
python main.py --config config.yaml --run-deployment
```

**Via Jupyter:**
```bash
jupyter notebook notebooks/thesis_pipeline.ipynb
```

---

## Ablation matrix

| Config | Preprocess norm | Model norm | LSTM | MS-CNN | SE | Gated fusion | Purpose |
|--------|----------------|------------|------|--------|-----|--------------|---------|
| A0 | z-score | none | ✓ | ✓ | ✓ | ✓ | Control baseline |
| A1 | none | learnable | ✓ | ✓ | ✓ | ✓ | **Proposed system** |
| A2 | none | layer | ✓ | ✓ | ✓ | ✓ | LayerNorm ablation |
| A3 | none | instance | ✓ | ✓ | ✓ | ✓ | InstanceNorm ablation |
| A4 | z-score | none | ✗ | ✓ | ✓ | ✓ | CNN-only |
| A5 | z-score | none | ✓ | ✗ | ✓ | ✓ | LSTM-only |
| A6 | z-score | none | ✓ | ✓ | ✗ | ✓ | No SE attention |
| A7 | z-score | none | ✓ | ✓ | ✓ | ✗ | No gated fusion |
| A8 | z-score | none | ✓ | single | ✓ | ✓ | Single-scale CNN |
| A9 | none | learnable | ✓ | ✓ | ✓ | ✓ | Full proposed (= A1) |

---

## Outputs

After a full run, `outputs/` contains:

```
outputs/
├── hybrid/
│   ├── final_comparison.csv          CNN, LSTM, Hybrid — all modalities
│   ├── final_comparison.png
│   └── {ecg,eeg,ppg}/
│       ├── best_hybrid.pt            Best checkpoint
│       ├── training_history.csv/.png
│       ├── confusion_matrix.png
│       ├── attention_profiles.png
│       └── test_metrics.json
├── baselines/
│   └── {ecg,eeg,ppg}/{cnn,lstm}/
│       ├── best_{cnn,lstm}.pt
│       └── test_metrics.json
├── ablations/
│   └── {ecg,eeg,ppg}/
│       ├── ablation_results.csv      Mean ± std for all 10 configs
│       └── ablation_results.png
├── statistical/
│   └── {ecg,eeg,ppg}/
│       ├── statistical_summary.csv   Mean ± std across seeds
│       ├── significance_tests.csv    Pairwise Wilcoxon tests
│       ├── results_table.tex         Ready-to-paste LaTeX
│       └── violin_macro_f1.png
├── xai/
│   └── {ecg,eeg,ppg}/
│       ├── *_class_profiles.png      Grad-CAM, IG, LRP per class
│       └── *_sample*.png             Individual sample attributions
├── late_fusion/
│   ├── modality_weights.json
│   └── late_fusion_history.csv
├── deployment/
│   └── {ecg,eeg,ppg}/
│       ├── deployment_report.json
│       ├── deployment_comparison.csv  Original, INT8, Pruned
│       └── latency_profile.png
└── run.log                            Full training log
```

---

## Multimodal framing

The three datasets have incompatible label spaces and cannot be aligned at the sample level. The thesis addresses multimodality in two valid ways:

1. **Architectural generalisation** — one unified `HybridCNNLSTM` trained and evaluated separately on ECG, EEG, and PPG, demonstrating that a single architecture generalises across physiologically distinct signal types.

2. **Late-fusion ensemble** — per-modality model outputs are combined via a learned weighted average after remapping labels to a shared five-class physiological-state taxonomy (REST / MILD_STRESS / HIGH_STRESS / PATHOLOGICAL / TRANSITION) defined in `PhysiologicalStateMapper`.

Both strategies are implemented, and the thesis explicitly states that sample-level sensor fusion is not possible with these three datasets.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| torch | ≥ 2.1 | Model training and inference |
| wfdb | ≥ 4.1 | MIT-BIH record loading |
| mne | ≥ 1.6 | EEG/EDF loading and epoching |
| neurokit2 | ≥ 0.2 | ECG R-peak detection (Pan-Tompkins) |
| h5py | ≥ 3.9 | HDF5 dataset storage |
| scikit-learn | ≥ 1.3 | Label encoding, metrics, splits |
| scipy | ≥ 1.11 | Signal filtering, Wilcoxon test |
| numpy | ≥ 1.24 | Array operations |
| pandas | ≥ 2.0 | Result tables |
| matplotlib | ≥ 3.7 | Figures |
| seaborn | ≥ 0.13 | Confusion matrices |
| tqdm | ≥ 4.66 | Progress bars |

---

## Citation

```bibtex
@thesis{zafeiratos2026multimodal,
  title   = {Multimodal Biomedical Signal Classification Using a Hybrid
             CNN--LSTM Architecture with Adaptive Normalization and
             Explainable AI},
  author  = {Author: Giorgos Zafeiratos},
  year    = {2025-2026},
  school  = {University of East London},
  type    = {Bachelor's Thesis},
}
```

---

## Licence

This code is released under the MIT licence. The three datasets are subject to their own licences:
- MIT-BIH: [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mitdb/1.0.0/)
- EEG Motor Movement: [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/eegmmidb/1.0.0/)
- PPG-DaLiA: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/495/ppg+dalia)
