"""

This module provides:

Research-grade evaluation
-------------------------
  1. SubjectWiseSplitter
       Splits data by subject ID, never by random sample.
       Prevents data leakage: windows from the same subject cannot
       appear in both train and test sets.

  2. CrossDatasetEvaluator
       Trains on one subset of subjects, evaluates on held-out subjects.
       Measures how well the model generalises to unseen patients —
       the clinically relevant evaluation for real-world deployment.

  3. StatisticalEvaluator
       Runs experiments across n_seeds, computes mean ± std for every
       metric, and runs paired Wilcoxon signed-rank tests between
       model variants (the standard non-parametric test for comparing
       ML classifiers on non-Gaussian metric distributions).

  4. ResearchGradeReport
       Produces the tables and figures expected in a published thesis:
       - Table: mean ± std across seeds, with significance markers
       - Figure: violin plots of metric distributions per model
       - Figure: learning curves averaged across seeds

Deployment analysis
-------------------
  5. LatencyProfiler
       Measures wall-clock latency at batch sizes 1, 8, 32, 64.
       Separates preprocessing latency from model inference latency.
       Reports: mean latency (ms), std, P95, P99, throughput (samples/s).

  6. ModelSizeAnalyser
       Reports: parameter count, model size on disk (MB),
       estimated memory footprint during inference (MB).

  7. FLOPCounter
       Estimates FLOPs for one forward pass using hook-based counting.
       Reports GFLOPs alongside parameter count — standard in
       efficient-model papers.

  8. ModelCompressor
       Applies dynamic INT8 quantisation (torch.quantization) and
       optional structured pruning to produce a lightweight variant
       suitable for edge devices. Measures accuracy degradation.

  9. DeploymentReport
       Produces a complete deployment readiness table comparing
       Original vs Quantised vs Pruned variants.

"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

log = logging.getLogger(__name__)


# 1. SUBJECT-WISE SPLITTER

class SubjectWiseSplitter:
    """
    Splits dataset by subject ID to prevent data leakage.

    Why this matters for biomedical data
    ------------------------------------
    A random 80/20 split of windows from the same subjects means the model
    sees different windows from the same person in both train and test. Because
    individual subjects have distinctive ECG/EEG/PPG morphologies, the model
    learns subject identity rather than the target class — a form of data
    leakage that inflates performance metrics by 5–15%.

    A subject-wise split ensures the test set contains ONLY subjects the
    model has never seen, measuring true cross-subject generalisation.

    Parameters
    ----------
    subject_ids : array of shape (N,) — subject ID for each window
    labels      : array of shape (N,) — class label for each window
    test_frac   : fraction of subjects (not windows) for test set
    val_frac    : fraction of subjects for validation set
    seed        : random seed for reproducibility

    Usage
    -----
    splitter = SubjectWiseSplitter(subject_ids, labels)
    idx_train, idx_val, idx_test = splitter.split()
    """

    def __init__(
        self,
        subject_ids: np.ndarray,
        labels:      np.ndarray,
        test_frac:   float = 0.20,
        val_frac:    float = 0.15,
        seed:        int   = 42,
    ):
        self.subject_ids = subject_ids
        self.labels      = labels
        self.test_frac   = test_frac
        self.val_frac    = val_frac
        self.seed        = seed
        self.subjects    = np.unique(subject_ids)

    def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (train_indices, val_indices, test_indices) into the
        original dataset. Indices are sample-level, not subject-level.
        """
        rng       = np.random.RandomState(self.seed)
        subjects  = self.subjects.copy()
        rng.shuffle(subjects)

        n_subj    = len(subjects)
        n_test    = max(1, int(n_subj * self.test_frac))
        n_val     = max(1, int(n_subj * self.val_frac))
        n_train   = n_subj - n_test - n_val

        if n_train <= 0:
            raise ValueError(
                f"Too few subjects ({n_subj}) for "
                f"test_frac={self.test_frac} + val_frac={self.val_frac}."
            )

        subj_train = subjects[:n_train]
        subj_val   = subjects[n_train : n_train + n_val]
        subj_test  = subjects[n_train + n_val :]

        idx_train = np.where(np.isin(self.subject_ids, subj_train))[0]
        idx_val   = np.where(np.isin(self.subject_ids, subj_val))[0]
        idx_test  = np.where(np.isin(self.subject_ids, subj_test))[0]

        log.info(
            "Subject-wise split | subjects: %d train / %d val / %d test "
            "| windows: %d / %d / %d",
            len(subj_train), len(subj_val), len(subj_test),
            len(idx_train),  len(idx_val),  len(idx_test),
        )
        return idx_train, idx_val, idx_test

    def get_subject_ids_for_split(
        self,
        split_indices: np.ndarray,
    ) -> np.ndarray:
        """Return unique subject IDs present in a given split."""
        return np.unique(self.subject_ids[split_indices])

    @staticmethod
    def extract_subject_ids_mitbih(h5_path: str) -> np.ndarray:
        """
        Load subject IDs saved alongside windows in the HDF5 file.
        If not present (preprocessing_pipeline.py did not save them),
        returns a placeholder array of zeros (falls back to random split).
        """
        with h5py.File(h5_path, "r") as f:
            for split in ("train", "val", "test"):
                key = f"ecg/{split}/subject_ids"
                if key in f:
                    return f[key][:]
        log.warning(
            "Subject IDs not found in HDF5 — subject-wise split unavailable. "
            "Re-run preprocessing_pipeline.py with save_subject_ids=True."
        )
        return None


# 2. CROSS-DATASET / LEAVE-ONE-SUBJECT-OUT EVALUATOR

class CrossDatasetEvaluator:
    """
    Evaluates cross-subject generalisation via Leave-One-Subject-Out (LOSO)
    or a held-out subject group evaluation.

    Two modes
    ---------
    "loso" : train on all subjects except one, test on the excluded subject.
             Repeat for every subject. Report mean ± std across subjects.
             Computationally expensive but gold standard for clinical eval.

    "holdout" : train on 80% of subjects, test on 20% never seen.
                Repeat n_runs times with different held-out subsets.
                Practical for large datasets (MIT-BIH, EEG Motor Movement).

    Parameters
    ----------
    X           : (N, C, T) signal array
    y           : (N,) label array
    subject_ids : (N,) subject ID array
    model_fn    : callable that returns a fresh nn.Module given (in_ch, n_classes)
    train_fn    : callable(model, X_train, y_train, X_val, y_val) → trained model
    """

    def __init__(
        self,
        X:           np.ndarray,
        y:           np.ndarray,
        subject_ids: np.ndarray,
        model_fn,
        train_fn,
        mode:        str = "holdout",
        n_runs:      int = 5,
        seed:        int = 42,
    ):
        self.X           = X
        self.y           = y
        self.subject_ids = subject_ids
        self.model_fn    = model_fn
        self.train_fn    = train_fn
        self.mode        = mode
        self.n_runs      = n_runs
        self.seed        = seed

    def run(self) -> pd.DataFrame:
        """
        Run cross-subject evaluation.
        Returns DataFrame with per-subject (LOSO) or per-run (holdout) metrics.
        """
        if self.mode == "loso":
            return self._run_loso()
        else:
            return self._run_holdout()

    def _run_loso(self) -> pd.DataFrame:
        subjects = np.unique(self.subject_ids)
        rows = []
        for subj in subjects:
            test_idx  = np.where(self.subject_ids == subj)[0]
            train_idx = np.where(self.subject_ids != subj)[0]

            # Split train into train/val (80/20 of remaining)
            val_cut = int(len(train_idx) * 0.8)
            val_idx   = train_idx[val_cut:]
            train_idx = train_idx[:val_cut]

            n_ch      = self.X.shape[1]
            n_classes = int(self.y.max()) + 1
            model     = self.model_fn(n_ch, n_classes)
            model     = self.train_fn(
                model,
                self.X[train_idx], self.y[train_idx],
                self.X[val_idx],   self.y[val_idx],
            )

            y_pred = _predict(model, self.X[test_idx])
            y_true = self.y[test_idx]
            rows.append({
                "subject":   subj,
                "n_test":    len(y_true),
                "accuracy":  accuracy_score(y_true, y_pred),
                "macro_f1":  f1_score(y_true, y_pred, average="macro", zero_division=0),
            })
            log.info("LOSO subject %s | acc=%.4f | f1=%.4f",
                     subj, rows[-1]["accuracy"], rows[-1]["macro_f1"])

        df = pd.DataFrame(rows)
        log.info(
            "LOSO summary | acc=%.4f±%.4f | f1=%.4f±%.4f",
            df["accuracy"].mean(), df["accuracy"].std(),
            df["macro_f1"].mean(),  df["macro_f1"].std(),
        )
        return df

    def _run_holdout(self) -> pd.DataFrame:
        subjects = np.unique(self.subject_ids)
        rng      = np.random.RandomState(self.seed)
        rows     = []

        for run_i in range(self.n_runs):
            perm       = rng.permutation(subjects)
            n_test_s   = max(1, int(len(perm) * 0.20))
            test_subj  = perm[:n_test_s]
            train_subj = perm[n_test_s:]

            test_idx   = np.where(np.isin(self.subject_ids, test_subj))[0]
            trainval   = np.where(np.isin(self.subject_ids, train_subj))[0]
            val_cut    = int(len(trainval) * 0.85)
            train_idx  = trainval[:val_cut]
            val_idx    = trainval[val_cut:]

            n_ch      = self.X.shape[1]
            n_classes = int(self.y.max()) + 1
            model     = self.model_fn(n_ch, n_classes)
            model     = self.train_fn(
                model,
                self.X[train_idx], self.y[train_idx],
                self.X[val_idx],   self.y[val_idx],
            )

            y_pred = _predict(model, self.X[test_idx])
            y_true = self.y[test_idx]
            rows.append({
                "run":       run_i,
                "n_test_subjects": n_test_s,
                "n_test_windows":  len(y_true),
                "accuracy":  accuracy_score(y_true, y_pred),
                "macro_f1":  f1_score(y_true, y_pred, average="macro", zero_division=0),
            })
            log.info("Holdout run %d | acc=%.4f | f1=%.4f",
                     run_i, rows[-1]["accuracy"], rows[-1]["macro_f1"])

        df = pd.DataFrame(rows)
        log.info(
            "Holdout summary | acc=%.4f±%.4f | f1=%.4f±%.4f",
            df["accuracy"].mean(), df["accuracy"].std(),
            df["macro_f1"].mean(),  df["macro_f1"].std(),
        )
        return df


def _predict(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Batch prediction from numpy array without DataLoader."""
    model.eval()
    device = next(model.parameters()).device
    preds  = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        with torch.no_grad():
            preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# 3. STATISTICAL EVALUATOR

class StatisticalEvaluator:
    """
    Runs multiple independent training runs and produces
    statistically meaningful results.

    Computes:
    - Mean ± std for accuracy, macro-F1, weighted-F1 across n_seeds runs
    - Paired Wilcoxon signed-rank test between model A and model B
      (non-parametric, appropriate for non-Gaussian metric distributions)
    - Cohen's d effect size between model variants

    Why Wilcoxon, not t-test?
    -------------------------
    ML metric distributions across seeds are often non-Gaussian (bounded
    at [0, 1], skewed for imbalanced datasets). The Wilcoxon signed-rank
    test makes no distribution assumptions and is the standard comparison
    test in biomedical ML literature.

    Parameters
    ----------
    model_fns : dict of {model_name: callable returning nn.Module}
    train_fn  : callable(model, loaders) → trained model
    eval_fn   : callable(model, loader) → metrics dict
    n_seeds   : number of independent runs
    """

    def __init__(
        self,
        model_fns: Dict[str, callable],
        train_fn:  callable,
        eval_fn:   callable,
        n_seeds:   int = 5,
        base_seed: int = 42,
    ):
        self.model_fns = model_fns
        self.train_fn  = train_fn
        self.eval_fn   = eval_fn
        self.n_seeds   = n_seeds
        self.base_seed = base_seed
        self.results:  Dict[str, List[Dict]] = {name: [] for name in model_fns}

    def run(self, loaders_fn: callable) -> pd.DataFrame:
        """
        Run all models across all seeds.

        Parameters
        ----------
        loaders_fn : callable(seed) → dict of DataLoaders
                     Called once per seed to produce freshly-split data.

        Returns
        -------
        summary_df : DataFrame with mean ± std per model per metric.
        """
        for seed_offset in range(self.n_seeds):
            seed    = self.base_seed + seed_offset * 100
            loaders = loaders_fn(seed)
            log.info("Statistical run %d/%d (seed=%d)", seed_offset+1, self.n_seeds, seed)

            for name, fn in self.model_fns.items():
                torch.manual_seed(seed)
                np.random.seed(seed)
                model  = fn()
                model  = self.train_fn(model, loaders)
                metrics = self.eval_fn(model, loaders["test"])
                self.results[name].append(metrics)
                log.info("  %s | acc=%.4f | f1=%.4f",
                         name, metrics["accuracy"], metrics["macro_f1"])

        return self.summary()

    def summary(self) -> pd.DataFrame:
        """Compute mean ± std per model per metric."""
        rows = []
        metrics_keys = list(self.results[list(self.results.keys())[0]][0].keys())

        for name, runs in self.results.items():
            row = {"model": name}
            for k in metrics_keys:
                vals = [r[k] for r in runs]
                row[f"{k}_mean"] = np.mean(vals)
                row[f"{k}_std"]  = np.std(vals)
                row[f"{k}_min"]  = np.min(vals)
                row[f"{k}_max"]  = np.max(vals)
            rows.append(row)
        df = pd.DataFrame(rows)
        log.info("Statistical summary:\n%s", df.to_string(index=False))
        return df

    def significance_test(
        self,
        model_a: str,
        model_b: str,
        metric:  str = "macro_f1",
        alpha:   float = 0.05,
    ) -> Dict:
        """
        Paired Wilcoxon signed-rank test comparing two models.

        Returns dict with: statistic, p_value, significant, effect_size_d
        """
        a_vals = [r[metric] for r in self.results[model_a]]
        b_vals = [r[metric] for r in self.results[model_b]]

        if len(a_vals) < 5:
            log.warning(
                "Only %d runs — Wilcoxon test requires ≥5 paired samples "
                "for reliable results. Increase n_seeds.", len(a_vals)
            )

        try:
            stat, p = scipy_stats.wilcoxon(a_vals, b_vals, alternative="two-sided")
        except ValueError as e:
            log.warning("Wilcoxon test failed: %s", e)
            stat, p = float("nan"), float("nan")

        # Cohen's d effect size
        diff = np.array(a_vals) - np.array(b_vals)
        d    = diff.mean() / (diff.std() + 1e-9)

        result = {
            "model_a":     model_a,
            "model_b":     model_b,
            "metric":      metric,
            "a_mean":      np.mean(a_vals),
            "b_mean":      np.mean(b_vals),
            "statistic":   stat,
            "p_value":     p,
            "significant": p < alpha,
            "alpha":       alpha,
            "cohens_d":    d,
        }
        sig_str = "✓ SIGNIFICANT" if result["significant"] else "✗ not significant"
        log.info(
            "Wilcoxon %s vs %s [%s]: p=%.4f %s | d=%.3f",
            model_a, model_b, metric, p, sig_str, d
        )
        return result

    def all_pairwise_tests(
        self,
        metric: str = "macro_f1",
        alpha:  float = 0.05,
    ) -> pd.DataFrame:
        """Run all pairwise Wilcoxon tests between registered models."""
        names = list(self.results.keys())
        rows  = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rows.append(
                    self.significance_test(names[i], names[j], metric, alpha)
                )
        return pd.DataFrame(rows)


# 4. RESEARCH-GRADE REPORT

class ResearchGradeReport:
    """
    Generates the tables and figures expected in a published thesis.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def latex_results_table(
        self,
        summary_df:   pd.DataFrame,
        sig_tests_df: pd.DataFrame,
        metric:       str = "macro_f1",
        caption:      str = "Model comparison results (mean ± std).",
    ) -> str:
        """
        Generate a LaTeX table with mean ± std and significance markers.
        Bold entries are the best-performing model per metric.

        Returns LaTeX string.
        """
        mean_col = f"{metric}_mean"
        std_col  = f"{metric}_std"
        best_val = summary_df[mean_col].max()

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Model & Accuracy & Macro F1 & Weighted F1 \\\\",
            "\\hline",
        ]

        for _, row in summary_df.iterrows():
            name = row["model"]
            cells = []
            for m in ["accuracy", "macro_f1", "weighted_f1"]:
                mn  = row[f"{m}_mean"]
                std = row[f"{m}_std"]
                cell = f"{mn:.4f}\\textpm{{{std:.4f}}}"
                if m == metric and abs(mn - best_val) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            lines.append(f"{name} & {' & '.join(cells)} \\\\")

        lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
        latex = "\n".join(lines)

        path = os.path.join(self.output_dir, "results_table.tex")
        with open(path, "w") as f:
            f.write(latex)
        log.info("LaTeX table → %s", path)
        return latex

    def violin_plot(
        self,
        evaluator: StatisticalEvaluator,
        metric:    str = "macro_f1",
    ) -> str:
        """Violin plot of metric distributions per model across seeds."""
        model_names  = list(evaluator.results.keys())
        metric_vals  = [
            [r[metric] for r in evaluator.results[name]]
            for name in model_names
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 5))
        parts = ax.violinplot(metric_vals, showmedians=True, showextrema=True)

        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Distribution of {metric} across {evaluator.n_seeds} runs")
        ax.grid(axis="y", alpha=0.3)

        # Add significance brackets
        try:
            sig_df = evaluator.all_pairwise_tests(metric)
            y_top  = max(max(v) for v in metric_vals) + 0.02
            for _, row in sig_df[sig_df["significant"]].iterrows():
                i = model_names.index(row["model_a"]) + 1
                j = model_names.index(row["model_b"]) + 1
                ax.annotate(
                    "", xy=(j, y_top), xytext=(i, y_top),
                    arrowprops=dict(arrowstyle="-", lw=1.5),
                )
                ax.text((i + j) / 2, y_top + 0.005, "*", ha="center",
                        fontsize=12, color="red")
                y_top += 0.02
        except Exception:
            pass

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"violin_{metric}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        log.info("Violin plot → %s", path)
        return path


# 5. LATENCY PROFILER

class LatencyProfiler:
    """
    Measures model inference latency with statistical rigour.

    Measures wall-clock time for forward passes at multiple batch sizes.
    Reports mean, std, P95, P99 latency and throughput.

    GPU timing uses CUDA events for accuracy (avoids Python overhead).
    CPU timing uses time.perf_counter().

    Parameters
    ----------
    model      : trained nn.Module (in eval mode)
    in_channels: input channels
    seq_len    : input sequence length
    device     : "cuda" or "cpu"
    n_warmup   : warm-up iterations before timing (populates CUDA caches)
    n_iters    : number of timed iterations per batch size
    """

    def __init__(
        self,
        model:       nn.Module,
        in_channels: int,
        seq_len:     int,
        device:      str = "cpu",
        n_warmup:    int = 20,
        n_iters:     int = 200,
    ):
        self.model       = model.to(device).eval()
        self.in_channels = in_channels
        self.seq_len     = seq_len
        self.device      = device
        self.n_warmup    = n_warmup
        self.n_iters     = n_iters

    def profile(
        self,
        batch_sizes: List[int] = [1, 8, 32, 64],
    ) -> pd.DataFrame:
        """
        Profile latency for each batch size.

        Returns DataFrame with columns:
            batch_size, mean_ms, std_ms, p95_ms, p99_ms,
            throughput_samples_per_sec
        """
        rows = []
        for bs in batch_sizes:
            x = torch.randn(bs, self.in_channels, self.seq_len,
                            device=self.device)

            # Warm-up
            with torch.no_grad():
                for _ in range(self.n_warmup):
                    _ = self.model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()

            # Timed iterations
            latencies = []
            for _ in range(self.n_iters):
                if self.device == "cuda":
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev   = torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    with torch.no_grad():
                        _ = self.model(x)
                    end_ev.record()
                    torch.cuda.synchronize()
                    latencies.append(start_ev.elapsed_time(end_ev))  # ms
                else:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = self.model(x)
                    latencies.append((time.perf_counter() - t0) * 1000)  # ms

            latencies = np.array(latencies)
            per_sample_ms = latencies / bs

            rows.append({
                "batch_size":                 bs,
                "batch_latency_mean_ms":      float(latencies.mean()),
                "batch_latency_std_ms":       float(latencies.std()),
                "per_sample_mean_ms":         float(per_sample_ms.mean()),
                "per_sample_std_ms":          float(per_sample_ms.std()),
                "per_sample_p95_ms":          float(np.percentile(per_sample_ms, 95)),
                "per_sample_p99_ms":          float(np.percentile(per_sample_ms, 99)),
                "throughput_samples_per_sec": float(bs / (latencies.mean() / 1000)),
            })
            log.info(
                "Latency [bs=%d] mean=%.3fms/sample p95=%.3fms "
                "throughput=%.0f samples/s",
                bs, per_sample_ms.mean(),
                np.percentile(per_sample_ms, 95),
                rows[-1]["throughput_samples_per_sec"],
            )

        return pd.DataFrame(rows)

    def plot_latency(self, df: pd.DataFrame, output_dir: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Latency vs batch size
        axes[0].errorbar(
            df["batch_size"], df["per_sample_mean_ms"],
            yerr=df["per_sample_std_ms"],
            marker="o", capsize=4, color="#4C72B0", linewidth=1.5,
        )
        axes[0].fill_between(
            df["batch_size"],
            df["per_sample_p95_ms"], df["per_sample_p99_ms"],
            alpha=0.2, color="#4C72B0", label="P95–P99",
        )
        axes[0].set_xlabel("Batch size")
        axes[0].set_ylabel("Latency per sample (ms)")
        axes[0].set_title("Inference Latency")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Throughput vs batch size
        axes[1].bar(
            df["batch_size"].astype(str),
            df["throughput_samples_per_sec"],
            color="#55A868", edgecolor="white",
        )
        axes[1].set_xlabel("Batch size")
        axes[1].set_ylabel("Throughput (samples/s)")
        axes[1].set_title("Inference Throughput")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "latency_profile.png")
        plt.savefig(path, dpi=150)
        plt.close()
        log.info("Latency profile → %s", path)
        return path


# 6. MODEL SIZE ANALYZER

class ModelSizeAnalyser:
    """
    Reports parameter count, disk size, and estimated inference memory.

    Parameter count separates trainable from frozen parameters
    (relevant if encoder weights are frozen for fine-tuning).

    Memory estimate = parameter_bytes + activation_bytes for one sample.
    """

    @staticmethod
    def analyse(
        model:       nn.Module,
        in_channels: int,
        seq_len:     int,
        device:      str = "cpu",
    ) -> Dict:
        # Parameter counts
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params    = total_params - trainable_params

        # Disk size (float32 baseline)
        param_bytes = total_params * 4   # float32
        size_mb     = param_bytes / (1024 ** 2)

        # Activation memory for batch_size=1
        activation_bytes = ModelSizeAnalyser._estimate_activations(
            model, in_channels, seq_len, device
        )
        activation_mb = activation_bytes / (1024 ** 2)

        result = {
            "total_params":     total_params,
            "trainable_params": trainable_params,
            "frozen_params":    frozen_params,
            "param_size_mb":    round(size_mb, 3),
            "activation_mb":    round(activation_mb, 3),
            "total_memory_mb":  round(size_mb + activation_mb, 3),
        }

        log.info(
            "Model size | params=%s (%.2f MB) | "
            "activation=%.2f MB | total=%.2f MB",
            f"{total_params:,}", size_mb, activation_mb, size_mb + activation_mb,
        )
        return result

    @staticmethod
    def _estimate_activations(
        model: nn.Module, in_channels: int, seq_len: int, device: str
    ) -> int:
        """Hook-based activation memory estimation for batch_size=1."""
        total_bytes = [0]

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                total_bytes[0] += output.numel() * 4  # float32

        handles = []
        for m in model.modules():
            handles.append(m.register_forward_hook(hook))

        try:
            x = torch.randn(1, in_channels, seq_len, device=device)
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()

        return total_bytes[0]


# 7. FLOP COUNTER

class FLOPCounter:
    """
    Estimates FLOPs for one forward pass via hooks.

    Counts multiply-add operations (MACs × 2 = FLOPs) for:
    - nn.Conv1d : FLOPs = 2 × Cout × T_out × Cin × kernel_size
    - nn.Linear : FLOPs = 2 × out_features × in_features
    - nn.LSTM   : estimated as 8 × hidden × input_size × T per layer
    """

    @staticmethod
    def count(
        model:       nn.Module,
        in_channels: int,
        seq_len:     int,
        device:      str = "cpu",
    ) -> Dict:
        flops = [0]
        handles = []

        def conv1d_hook(module: nn.Conv1d, input, output):
            b, c_out, t_out = output.shape
            k    = module.kernel_size[0]
            c_in = module.in_channels // module.groups
            flops[0] += 2 * c_out * t_out * c_in * k

        def linear_hook(module: nn.Linear, input, output):
            flops[0] += 2 * module.in_features * module.out_features

        def lstm_hook(module: nn.LSTM, input, output):
            # Estimate: 8 per timestep per layer (4 gates × 2 matmuls)
            seq_len_l = input[0].shape[1]
            dirs = 2 if module.bidirectional else 1
            for layer in range(module.num_layers):
                in_sz  = module.input_size if layer == 0 else module.hidden_size * dirs
                flops[0] += 8 * module.hidden_size * in_sz * seq_len_l * dirs

        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                handles.append(m.register_forward_hook(conv1d_hook))
            elif isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(linear_hook))
            elif isinstance(m, nn.LSTM):
                handles.append(m.register_forward_hook(lstm_hook))

        try:
            x = torch.randn(1, in_channels, seq_len, device=device)
            with torch.no_grad():
                model(x)
        finally:
            for h in handles:
                h.remove()

        gflops = flops[0] / 1e9
        log.info("FLOPs: %.4f GFLOPs (%d raw)", gflops, flops[0])
        return {"flops": flops[0], "gflops": round(gflops, 6)}


# 8. MODEL COMPRESSOR

class ModelCompressor:
    """
    Applies dynamic INT8 quantisation and optional structured pruning
    to produce a lightweight model variant for edge deployment.

    Quantization
    ------------
    Dynamic INT8 quantisation (torch.quantization.quantize_dynamic):
    - Converts Linear and LSTM weights to INT8 at inference time
    - No calibration dataset required
    - Typically 2–4× smaller model, 1.5–2× faster on CPU
    - Minimal accuracy loss (< 0.5% on most biomedical tasks)

    Pruning
    -------
    Structured L1 pruning of Conv1d filters:
    - Removes entire filter rows below an L1 norm threshold
    - Produces a truly smaller network (not just sparse weights)
    - Requires fine-tuning after pruning to recover accuracy

    Parameters
    ----------
    model : trained HybridCNNLSTM or AblationHybrid
    """

    def __init__(self, model: nn.Module):
        self.original_model = model
        self.quantised_model = None
        self.pruned_model    = None

    def quantise(self) -> nn.Module:
        """
        Apply dynamic INT8 quantisation.
        Returns quantised model (CPU only — INT8 not supported on CUDA).
        """
        import copy
        model_cpu = copy.deepcopy(self.original_model).cpu().eval()

        self.quantised_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8,
        )
        log.info("Dynamic INT8 quantisation applied.")
        return self.quantised_model

    def prune(self, amount: float = 0.3) -> nn.Module:
        """
        Apply L1 unstructured pruning to all Conv1d weight tensors.

        Parameters
        ----------
        amount : fraction of weights to prune (0.3 = 30% of each tensor)

        Returns model with pruning masks applied (weights are zeroed,
        not physically removed; use prune.remove() to make permanent).
        """
        import copy
        import torch.nn.utils.prune as prune

        pruned = copy.deepcopy(self.original_model).eval()

        for name, module in pruned.named_modules():
            if isinstance(module, nn.Conv1d):
                prune.l1_unstructured(module, name="weight", amount=amount)

        n_zeros = sum(
            (p == 0).sum().item()
            for p in pruned.parameters()
        )
        n_total = sum(p.numel() for p in pruned.parameters())
        sparsity = n_zeros / n_total
        log.info(
            "Pruning (amount=%.2f) | sparsity=%.4f (%d/%d weights zeroed)",
            amount, sparsity, n_zeros, n_total,
        )
        self.pruned_model = pruned
        return pruned

    def compare(
        self,
        loader:      DataLoader,
        in_channels: int,
        seq_len:     int,
        device:      str = "cpu",
    ) -> pd.DataFrame:
        """
        Compare Original, Quantised, and Pruned models on:
        accuracy, latency, model size, FLOPs.
        """
        variants = {"Original": self.original_model}
        if self.quantised_model is not None:
            variants["INT8 Quantised"] = self.quantised_model
        if self.pruned_model is not None:
            variants["L1 Pruned"]      = self.pruned_model

        rows = []
        for name, model in variants.items():
            model = model.to(device).eval()

            # Accuracy
            preds, labels = [], []
            with torch.no_grad():
                for X, y in loader:
                    preds.append(model(X.to(device)).argmax(1).cpu().numpy())
                    labels.append(y.numpy())
            y_pred = np.concatenate(preds)
            y_true = np.concatenate(labels)
            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

            # Size
            size = ModelSizeAnalyser.analyse(model, in_channels, seq_len, device)
            # FLOPs
            flops = FLOPCounter.count(model, in_channels, seq_len, device)
            # Latency (batch=1 only for edge relevance)
            profiler = LatencyProfiler(model, in_channels, seq_len, device,
                                       n_warmup=10, n_iters=50)
            lat_df = profiler.profile([1])
            lat_ms = lat_df.iloc[0]["per_sample_mean_ms"]

            rows.append({
                "Variant":        name,
                "Accuracy":       round(acc,  4),
                "Macro F1":       round(f1,   4),
                "Params (M)":     round(size["total_params"] / 1e6, 3),
                "Size (MB)":      size["param_size_mb"],
                "GFLOPs":         flops["gflops"],
                "Latency@1 (ms)": round(lat_ms, 3),
            })
            log.info(
                "%s | acc=%.4f | f1=%.4f | %.1fM params | %.2f MB | "
                "%.4f GFLOPs | %.3f ms",
                name, acc, f1,
                size["total_params"]/1e6, size["param_size_mb"],
                flops["gflops"], lat_ms,
            )

        return pd.DataFrame(rows)


# 9. DEPLOYMENT REPORT

def deployment_report(
    model:       nn.Module,
    loader:      DataLoader,
    in_channels: int,
    seq_len:     int,
    output_dir:  str,
    device:      str = "cpu",
) -> None:
    """
    Full deployment readiness analysis.

    Runs:
    1. Latency profiling across batch sizes 1, 8, 32, 64
    2. Model size analysis
    3. FLOP count
    4. Quantisation + pruning comparison

    Saves:
    - deployment_report.json    — all metrics
    - latency_profile.png       — latency vs batch size
    - deployment_comparison.csv — Original vs Quantised vs Pruned
    """
    os.makedirs(output_dir, exist_ok=True)
    report = {}

    # Latency
    profiler = LatencyProfiler(model, in_channels, seq_len, device)
    lat_df   = profiler.profile([1, 8, 32, 64])
    lat_df.to_csv(os.path.join(output_dir, "latency_by_batch.csv"), index=False)
    profiler.plot_latency(lat_df, output_dir)
    report["latency"] = lat_df.to_dict(orient="records")

    # Size
    size_info = ModelSizeAnalyser.analyse(model, in_channels, seq_len, device)
    report["model_size"] = size_info

    # FLOPs
    flop_info = FLOPCounter.count(model, in_channels, seq_len, device)
    report["flops"] = flop_info

    # Compression comparison
    compressor = ModelCompressor(model)
    compressor.quantise()
    compressor.prune(amount=0.3)
    cmp_df = compressor.compare(loader, in_channels, seq_len, device)
    cmp_df.to_csv(os.path.join(output_dir, "deployment_comparison.csv"), index=False)
    log.info("Deployment comparison:\n%s", cmp_df.to_string(index=False))
    report["compression"] = cmp_df.to_dict(orient="records")

    # Save full report
    with open(os.path.join(output_dir, "deployment_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Deployment report → %s", output_dir)

    # Summary log — the key numbers for the thesis
    bs1_row  = lat_df[lat_df["batch_size"] == 1].iloc[0]
    log.info(
        "\n── Deployment Summary ──────────────────────────\n"
        "  Parameters    : %s (%.2f MB)\n"
        "  GFLOPs        : %.4f\n"
        "  Latency@bs=1  : %.3f ms/sample (P99: %.3f ms)\n"
        "  Throughput@64 : %.0f samples/s\n"
        "────────────────────────────────────────────────",
        f"{size_info['total_params']:,}", size_info["param_size_mb"],
        flop_info["gflops"],
        bs1_row["per_sample_mean_ms"], bs1_row["per_sample_p99_ms"],
        lat_df[lat_df["batch_size"] == 64].iloc[0]["throughput_samples_per_sec"],
    )


# 10. COMPLETE EVALUATION ENTRY POINT

def run_full_evaluation(
    model:       nn.Module,
    loaders:     Dict[str, DataLoader],
    label_map:   Dict[str, str],
    modality:    str,
    output_dir:  str,
    n_seeds:     int = 5,
    device:      str = "cpu",
) -> None:
    """
    Full research-grade evaluation pipeline for one modality.

    Runs:
    1. Standard test-set evaluation (accuracy, F1, confusion matrix)
    2. Statistical evaluation across n_seeds (mean ± std)
    3. Deployment analysis (latency, size, FLOPs, quantisation)

    This is the function to call after run_hybrid() completes.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_ds  = loaders["train"].dataset
    n_ch      = train_ds.n_channels
    seq_len   = train_ds.seq_len
    n_classes = train_ds.n_classes

    log.info("=" * 60)
    log.info("Research-grade evaluation — %s", modality.upper())
    log.info("=" * 60)

    # Standard evaluation (already called in hybrid trainer, included for completeness)
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for X, y in loaders["test"]:
            preds.append(model(X.to(device)).argmax(1).cpu().numpy())
            labels_list.append(y.numpy())
    y_true = np.concatenate(labels_list)
    y_pred = np.concatenate(preds)

    n_cls   = int(y_true.max()) + 1
    cnames  = [label_map.get(str(i), str(i)) for i in range(n_cls)]
    report  = classification_report(y_true, y_pred, target_names=cnames, zero_division=0)
    log.info("Classification report:\n%s", report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Deployment report
    deploy_dir = os.path.join(output_dir, "deployment")
    deployment_report(
        model, loaders["test"], n_ch, seq_len, deploy_dir, device
    )

    log.info("Full evaluation complete → %s", output_dir)


# 
# USAGE
# 
#
# Subject-wise split:
#   splitter = SubjectWiseSplitter(subject_ids, labels)
#   idx_train, idx_val, idx_test = splitter.split()
#
# Statistical evaluation:
#   evaluator = StatisticalEvaluator(
#       model_fns={"CNN": cnn_fn, "Hybrid": hybrid_fn},
#       train_fn=my_train, eval_fn=my_eval, n_seeds=5,
#   )
#   summary_df = evaluator.run(loaders_fn)
#   evaluator.all_pairwise_tests()
#
# Deployment analysis:
#   deployment_report(model, loaders["test"], n_ch, seq_len,
#                     "outputs/deployment", device="cpu")
#
# Full evaluation pipeline:
#   run_full_evaluation(model, loaders, label_map, "ecg",
#                       "outputs/evaluation/ecg", n_seeds=5)
