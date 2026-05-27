# Architecture Reference

## File map

| File | What it implements |
|------|--------------------|
| `src/preprocessing_pipeline.py` | `ECGPreprocessor`, `EEGPreprocessor`, `PPGPreprocessor`, `MultimodalPreprocessingPipeline` |
| `src/baseline_models.py` | `CNN1DBaseline`, `LSTMBaseline`, `BaselineTrainer`, `evaluate` |
| `src/hybrid_model.py` | `LearnableNorm`, `MultiScaleCNNEncoder`, `MultiHeadTemporalAttention`, `GatedFusion`, `HybridCNNLSTM`, `HybridTrainer` |
| `src/multimodal_fusion.py` | `LateFusionEnsemble`, `MultimodalDataset`, `PhysiologicalStateMapper` |
| `src/ablation_framework.py` | `AblationConfig`, `NormalizationConfig`, `AblationHybrid`, `AblationRunner` |
| `src/xai_and_pipeline.py` | `GradCAM1D`, `IntegratedGradients1D`, `LRPExplainer`, `XAIVisualizer`, `UnifiedRunConfig`, `end_to_end_run` |
| `src/evaluation_suite.py` | `SubjectWiseSplitter`, `StatisticalEvaluator`, `LatencyProfiler`, `ModelSizeAnalyser`, `FLOPCounter`, `ModelCompressor` |

---

## LearnableNorm

Addresses the thesis gap: static normalization ignores inter-subject variability.

```
Input x: (B, C, T)
  ↓
μ = x.mean(dim=T)        # (B, C) — per-channel mean of THIS window
σ = x.std(dim=T)          # (B, C) — per-channel std  of THIS window
summary = concat[μ, σ]   # (B, 2C)
  ↓
[γ_offset, β] = MLP(summary)   # two-layer network, 2C → hidden → 2C
γ = 1 + γ_offset               # centred at 1 so identity is the default
  ↓
output = γ ⊙ ((x − μ) / σ) + β
```

The MLP is initialized with zero weights and zero biases, so at the start of training `LearnableNorm` is numerically identical to standard z-score normalization. Gradients then drive it toward input-conditioned adaptive scaling.

**Ablation**: Compare A0 (preprocessing z-score, no model norm) vs A1 (no preprocessing, `LearnableNorm`). The difference in test macro-F1 is the direct contribution of `LearnableNorm`.

---

## Multi-Scale CNN Encoder

Four parallel branches with kernel sizes 3, 7, 15, 31:

```
k=3  → detects sharp transients (QRS peaks, EEG spikes, PPG systolic upstroke)
k=7  → captures waveform morphology (P-wave, T-wave)
k=15 → mid-range oscillations (alpha bursts in EEG, BVP cycle shape)
k=31 → slow trends (baseline wander, respiration modulation)
```

Each branch: `LearnableNorm → Conv → (ResBlock × depth)` where each `ResBlock` contains two convolutions with a skip connection and an SE channel-attention block.

All branches are concatenated: `(B, 32×4, T_out) = (B, 128, 64)`.

**Ablation A8**: Replace all 4 branches with a single k=7 branch. The performance drop quantifies the contribution of multi-scale processing.

---

## Squeeze-and-Excitation (SE) blocks

Recalibrates channel responses:

```
z = GlobalAvgPool(x)     # (B, C)
s = σ(W₂ · ReLU(W₁ · z)) # (B, C), bottleneck reduction factor = 4
output = x ⊙ s          # element-wise channel scaling
```

SE blocks address the case where different EEG channels are differentially informative depending on the motor imagery task. **Ablation A6** removes them.

---

## Gated Fusion

Replaces simple concatenation with a learned mixture:

```
g = σ(W_g · [f_cnn ; f_lstm] + b) # (B, fusion_hidden)
f = g ⊙ proj(f_cnn) + (1−g) ⊙ proj(f_lstm)
```

When g→1, the model trusts the CNN (morphology-heavy classes, e.g. arrhythmia type).  
When g→0, the model trusts the LSTM (rhythm-heavy classes, e.g. motor imagery).  
**Ablation A7** replaces this with `concat + Linear`.

---

## Multi-Head Temporal Attention

Replaces the single additive attention head from the LSTM baseline:

```
K, V = BiLSTM_output projected
Q    = n_heads learnable global query vectors (one per head)

attn_weights = softmax(Q · Kᵀ / √d_head)   # (B, H, 1, T)
context      = attn_weights · V            # (B, H, 1, d_head)
output       = concat[heads] projected     # (B, d_model)
```

Each head specialises in a different temporal pattern simultaneously. The weights are exported via `model.get_attention_maps()` and visualized in the XAI module.

---

## XAI methods

Three complementary attribution methods are implemented:

| Method | What it answers | Key property |
|---|---|---|
| Grad-CAM | Which time region did the last CNN layer attend to? | Fast, coarse |
| Integrated Gradients | How much did each input point contribute? | Axiomatically faithful |
| LRP (ε-rule) | Layer-by-layer relevance redistribution through CNN | Conservation |

All three are run on the same 32 test samples and compared in a three-column figure per sample. Class-mean attribution profiles are produced for the thesis Chapter 5 figures.

---

## Evaluation design

**Subject-wise splitting** (implemented in `SubjectWiseSplitter`): subjects are partitioned into train/val/test *before* windowing, ensuring no window from a test subject appears in training. This is the clinically correct evaluation for generalisation to unseen patients.

**Statistical evaluation** (`StatisticalEvaluator`): all models are trained across 3–5 independent random seeds. Results are reported as mean ± std. Pairwise model comparisons use the **Wilcoxon signed-rank test** (non-parametric; appropriate for bounded, potentially non-Gaussian metric distributions) with Cohen's d effect size.
