#!/usr/bin/env bash
# 
# scripts/run_experiments.sh
# Runs the complete experimental sequence for the thesis.
# Logs each stage to a timestamped file under logs/.
#
# Usage:
#   bash scripts/run_experiments.sh            # all modalities, full run
#   bash scripts/run_experiments.sh --ecg-only # ECG fast-track (CPU-friendly)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

ECG_ONLY=false
if [[ "${1:-}" == "--ecg-only" ]]; then ECG_ONLY=true; fi

TS=$(date +"%Y%m%d_%H%M%S")

run_stage() {
    local stage="$1"; shift
    local logfile="logs/${TS}_${stage}.log"
    echo ""
    echo "========================================"
    echo "  Stage: $stage"
    echo "  Log  : $logfile"
    echo "========================================"
    python main.py "$@" 2>&1 | tee "$logfile"
    echo "  $stage complete."
}

if $ECG_ONLY; then
    # ECG fast-track
    run_stage "ecg_proposed" \
        --modalities ecg \
        --norm-strategy proposed \
        --epochs 30 \
        --no-ablations \
        --output-dir outputs_ecg_proposed

    run_stage "ecg_control" \
        --modalities ecg \
        --norm-strategy control \
        --epochs 30 \
        --no-ablations \
        --no-xai \
        --output-dir outputs_ecg_control

    run_stage "ecg_deployment" \
        --modalities ecg \
        --norm-strategy proposed \
        --no-ablations \
        --no-xai \
        --run-deployment \
        --output-dir outputs_ecg_proposed
else
    # Full run

    # Stage 1: Proposed system (LearnableNorm — novel contribution)
    run_stage "proposed_system" \
        --config config.yaml \
        --norm-strategy proposed \
        --output-dir outputs

    # Stage 2: Control system (z-score baseline — for A0 vs A1 ablation)
    run_stage "control_system" \
        --modalities ecg eeg ppg \
        --norm-strategy control \
        --no-xai \
        --no-ablations \
        --output-dir outputs_control

    # Stage 3: Deployment analysis
    run_stage "deployment" \
        --config config.yaml \
        --norm-strategy proposed \
        --no-ablations \
        --no-xai \
        --run-deployment \
        --output-dir outputs
fi

echo ""
echo "All stages complete. Key outputs:"
echo "  outputs/hybrid/final_comparison.csv"
echo "  outputs/ablations/*/ablation_results.csv"
echo "  outputs/statistical/*/statistical_summary.csv"
echo "  outputs/xai/*/"
echo "  outputs/deployment/*/"
echo "  logs/${TS}_*.log"
