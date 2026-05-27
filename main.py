"""
Single entry point for all thesis experiments

Thesis: Multimodal Biomedical Signal Classification Using a Hybrid CNN-LSTM
        Architecture with Adaptive Normalization and Explainable AI

Usage
-----
# Full pipeline (proposed system — LearnableNorm, no preprocessing norm):
    python main.py

# Full pipeline (control — z-score preprocessing, no model norm):
    python main.py --norm-strategy control

# Single modality only:
    python main.py --modalities ecg

# Skip slow stages:
    python main.py --no-ablations --no-xai

# Force re-run preprocessing even if HDF5 files already exist:
    python main.py --force-preprocess

# Dry run — print config and exit:
    python main.py --dry-run

All outputs land in:  outputs/
    outputs/baselines/          CNN and LSTM baseline results
    outputs/hybrid/             Hybrid CNN-LSTM results + final comparison
    outputs/ablations/          Ablation study results
    outputs/xai/                GradCAM, IG, LRP attribution figures
    outputs/statistical/        Mean ± std tables, Wilcoxon tests, violin plots
    outputs/late_fusion/        Late-fusion ensemble weights and metrics
    outputs/deployment/         Latency, FLOPs, quantisation comparison

"""

import argparse
import logging
import os
import sys
import time


# Ensure src/ is on the path so modules find each other
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import torch


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# Argument parser

def load_yaml_config(path: str) -> dict:
    """Load config.yaml and return a flat dict of overrides."""
    import yaml  # PyYAML — listed in requirements.txt
    with open(path) as f:
        data = yaml.safe_load(f)
    # yaml.safe_load returns None for an empty file
    return data if isinstance(data, dict) else {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multimodal Biomedical Signal Classification — Thesis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=None,
                   help="Path to config.yaml; values override argparse defaults, CLI flags override YAML.")

    # Paths
    p.add_argument("--raw-data-dir",  default="data/raw",       help="Root directory containing mit-bih/, eeg-motor/, ppg-dalia/ sub-folders")
    p.add_argument("--processed-dir", default="data/processed", help="Directory for preprocessed HDF5 outputs")
    p.add_argument("--output-dir",    default="outputs",         help="Root directory for all experiment outputs")

    # Modalities
    p.add_argument("--modalities", nargs="+", default=["ecg", "eeg", "ppg"],
                   choices=["ecg", "eeg", "ppg"],
                   help="Which modalities to process and train on")

    # Normalization strategy
    p.add_argument("--norm-strategy", default="proposed",
                   choices=["proposed", "control"],
                   help=(
                       "proposed → preprocessing_norm=none, model_norm=learnable  "
                       "(novel LearnableNorm, Gap #3 fix). "
                       "control  → preprocessing_norm=zscore, model_norm=none "
                       "(standard z-score baseline for comparison)."
                   ))

    # Training
    p.add_argument("--epochs",        type=int,   default=100,  help="Max training epochs")
    p.add_argument("--batch-size",    type=int,   default=64,   help="DataLoader batch size")
    p.add_argument("--lr",            type=float, default=5e-4, help="Learning rate")
    p.add_argument("--patience",      type=int,   default=20,   help="Early stopping patience")
    p.add_argument("--seed",          type=int,   default=42,   help="Global random seed")
    p.add_argument("--num-workers",   type=int,   default=4,    help="DataLoader worker processes")
    p.add_argument("--device",        default=None,
                   help="torch device string, e.g. 'cuda:0' or 'cpu'. Auto-detected if not set.")

    # Stage flags
    p.add_argument("--force-preprocess", action="store_true",
                   help="Re-run preprocessing even if HDF5 files already exist")
    p.add_argument("--no-ablations",     action="store_true",
                   help="Skip ablation study (saves ~3× the training time)")
    p.add_argument("--no-xai",           action="store_true",
                   help="Skip XAI analysis")
    p.add_argument("--n-xai-samples",    type=int, default=32,
                   help="Number of test samples to run XAI on per modality")
    p.add_argument("--ablation-runs",    type=int, default=3,
                   help="Independent training runs per ablation config (for mean±std)")
    p.add_argument("--stat-seeds",       type=int, default=3,
                   help="Independent seeds for statistical evaluation")

    # Deployment
    p.add_argument("--run-deployment", action="store_true",
                   help="Run deployment analysis (latency, FLOPs, quantisation) after training")

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config and exit without running anything")

    return p.parse_args()


# Config builder

def build_config(args: argparse.Namespace):
    """Convert CLI args to a UnifiedRunConfig, with YAML overrides applied first."""
    from xai_and_pipeline import UnifiedRunConfig

    # Apply YAML overrides: YAML sets base values, explicit CLI flags win
    if args.config is not None:
        overrides = load_yaml_config(args.config)
            explicit_cli_keys = {a.lstrip('-').replace('-', '_')
                                for a in sys.argv[1:] if a.startswitch('--')}
        for key, val in overrides.items():
                if key not in explicit_cli_keys:
                        setattr(args, key, val)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.norm_strategy == "proposed":
        preprocessing_norm = "none"
        model_norm         = "learnable"
    else:  # "control"
        preprocessing_norm = "zscore"
        model_norm         = "none"

    return UnifiedRunConfig(
        raw_data_dir       = args.raw_data_dir,
        processed_dir      = args.processed_dir,
        output_dir         = args.output_dir,
        modalities         = args.modalities,
        preprocessing_norm = preprocessing_norm,
        model_norm         = model_norm,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        learning_rate      = args.lr,
        patience           = args.patience,
        seed               = args.seed,
        num_workers        = args.num_workers,
        device             = device,
        run_xai            = not args.no_xai,
        n_xai_samples      = args.n_xai_samples,
        run_ablations      = not args.no_ablations,
        ablation_runs      = args.ablation_runs,
    )


# Deployment stage (optional, after training)

def run_deployment_stage(cfg, args: argparse.Namespace) -> None:
    """
    Post-training deployment analysis for every modality's hybrid model.
    Produces latency_profile.png, deployment_comparison.csv, deployment_report.json.
    """
    from evaluation_suite import deployment_report, run_full_evaluation
    from xai_and_pipeline import build_loaders_from_hdf5
    from hybrid_model import HybridCNNLSTM

    hybrid_cfg = cfg.to_hybrid_config()

    for modality in cfg.modalities:
        ckpt_path = os.path.join(cfg.output_dir, "hybrid", modality, "best_hybrid.pt")
        h5_path   = os.path.join(cfg.processed_dir, f"{modality}.h5")

        if not os.path.exists(ckpt_path):
            log.warning("Deployment: checkpoint not found for %s — skipping.", modality)
            continue
        if not os.path.exists(h5_path):
            log.warning("Deployment: HDF5 not found for %s — skipping.", modality)
            continue

        loaders  = build_loaders_from_hdf5(
            cfg.processed_dir, modality,
            cfg.batch_size, cfg.num_workers, cfg.device,
        )
        train_ds = loaders["train"].dataset
        n_ch     = train_ds.n_channels
        seq_len  = train_ds.seq_len
        n_classes= train_ds.n_classes
        label_map= train_ds.label_map

        model = HybridCNNLSTM(n_ch, n_classes, hybrid_cfg)
        ckpt  = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"])
        model.to(cfg.device)

        deploy_dir = os.path.join(cfg.output_dir, "deployment", modality)
        os.makedirs(deploy_dir, exist_ok=True)

        log.info("Deployment analysis — %s", modality.upper())
        deployment_report(
            model       = model,
            loader      = loaders["test"],
            in_channels = n_ch,
            seq_len     = seq_len,
            output_dir  = deploy_dir,
            device      = cfg.device,
        )

        run_full_evaluation(
            model     = model,
            loaders   = loaders,
            label_map = label_map,
            modality  = modality,
            output_dir= os.path.join(cfg.output_dir, "evaluation", modality),
            n_seeds   = args.stat_seeds,
            device    = cfg.device,
        )


# Summary printer

def print_final_summary(cfg, elapsed: float) -> None:
    """Print paths to all key output files after a complete run."""
    log.info("\n" + "=" * 70)
    log.info("RUN COMPLETE  (%.1f minutes)", elapsed / 60)
    log.info("=" * 70)

    key_outputs = [
        ("Final comparison table",    os.path.join(cfg.output_dir, "hybrid", "final_comparison.csv")),
        ("Final comparison figure",   os.path.join(cfg.output_dir, "hybrid", "final_comparison.png")),
        ("Dataset summary",           os.path.join(cfg.processed_dir, "dataset_summary.csv")),
    ]
    for modality in cfg.modalities:
        m = modality.upper()
        key_outputs += [
            (f"{m} ablation results",     os.path.join(cfg.output_dir, "ablations", modality, "ablation_results.csv")),
            (f"{m} ablation figure",      os.path.join(cfg.output_dir, "ablations", modality, "ablation_results.png")),
            (f"{m} statistical summary",  os.path.join(cfg.output_dir, "statistical", modality, "statistical_summary.csv")),
            (f"{m} significance tests",   os.path.join(cfg.output_dir, "statistical", modality, "significance_tests.csv")),
            (f"{m} LaTeX table",          os.path.join(cfg.output_dir, "statistical", modality, "results_table.tex")),
            (f"{m} violin plot",          os.path.join(cfg.output_dir, "statistical", modality, "violin_macro_f1.png")),
            (f"{m} XAI class profiles",   os.path.join(cfg.output_dir, "xai", modality)),
            (f"{m} confusion matrix",     os.path.join(cfg.output_dir, "hybrid", modality, "confusion_matrix.png")),
            (f"{m} hybrid checkpoint",    os.path.join(cfg.output_dir, "hybrid", modality, "best_hybrid.pt")),
        ]
    key_outputs += [
        ("Late fusion weights",       os.path.join(cfg.output_dir, "late_fusion", "modality_weights.json")),
        ("Run log",                   "run.log"),
    ]

    for label, path in key_outputs:
        exists = "✓" if os.path.exists(path) else "✗ (not produced)"
        log.info("  %-35s  %s  %s", label, exists, path)
    log.info("=" * 70)


# Main

def main() -> None:
    args = parse_args()

    # Build config
    cfg = build_config(args)

    if args.dry_run:
        import dataclasses
        log.info("DRY RUN — resolved config:")
        for k, v in dataclasses.asdict(cfg).items():
            log.info("  %-25s = %s", k, v)
        log.info("Exiting (--dry-run).")
        return

    # Log environment
    log.info("=" * 70)
    log.info("THESIS PIPELINE — START")
    log.info("  Python  : %s", sys.version.split()[0])
    log.info("  PyTorch : %s", torch.__version__)
    log.info("  CUDA    : %s", torch.version.cuda or "not available")
    log.info("  Device  : %s", cfg.device)
    log.info("  Seed    : %d", cfg.seed)
    log.info("  Modalities   : %s", cfg.modalities)
    log.info("  Norm strategy: preprocessing=%s  model=%s",
             cfg.preprocessing_norm, cfg.model_norm)
    log.info("  Output dir   : %s", cfg.output_dir)
    log.info("=" * 70)

    t_start = time.time()

    # Main pipeline
    from xai_and_pipeline import end_to_end_run
    end_to_end_run(cfg, force_preprocess=args.force_preprocess)

    # Optional deployment analysis
    if args.run_deployment:
        log.info("\n[Deployment Analysis]")
        run_deployment_stage(cfg, args)

    # Summary
    print_final_summary(cfg, elapsed=time.time() - t_start)


if __name__ == "__main__":
    main()
