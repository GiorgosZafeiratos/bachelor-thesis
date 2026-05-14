#!/usr/bin/env bash
# 
# scripts/download_data.sh
# Automated download for MIT-BIH and EEG Motor Movement via wfdb.
# PPG-DaLiA requires manual download from UCI (link printed below).
#
# Usage:
#   bash scripts/download_data.sh   # download all
#   bash scripts/download_data.sh --ecg # ECG only
#   bash scripts/download_data.sh --eeg # EEG only
#   bash scripts/download_data.sh --ppg # print PPG instructions only
# 
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ECG_DIR="$REPO_ROOT/data/raw/mit-bih"
EEG_DIR="$REPO_ROOT/data/raw/eeg-motor"
PPG_DIR="$REPO_ROOT/data/raw/ppg-dalia"

DO_ECG=false
DO_EEG=false
DO_PPG=false

if [[ $# -eq 0 ]]; then
    DO_ECG=true; DO_EEG=true; DO_PPG=true
else
    for arg in "$@"; do
        case "$arg" in
            --ecg) DO_ECG=true ;;
            --eeg) DO_EEG=true ;;
            --ppg) DO_PPG=true ;;
            *) echo "Unknown flag: $arg  (use --ecg / --eeg / --ppg)"; exit 1 ;;
        esac
    done
fi

# ECG: MIT-BIH via wfdb
if $DO_ECG; then
    echo ""
    echo "=== Downloading MIT-BIH Arrhythmia Database ==="
    mkdir -p "$ECG_DIR"

    n_existing=$(find "$ECG_DIR" -name "*.hea" 2>/dev/null | wc -l)
    if [[ "$n_existing" -ge 48 ]]; then
        echo "  Already present ($n_existing .hea files). Skipping."
    else
        python3 - << PYEOF
import wfdb, sys
print("  Downloading via wfdb.dl_database('mitdb', ...) — ~100 MB ...")
try:
    wfdb.dl_database('mitdb', '$ECG_DIR')
    import os
    count = len([f for f in os.listdir('$ECG_DIR') if f.endswith('.hea')])
    print(f"  Done. {count} records downloaded.")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  Manual download: https://physionet.org/content/mitdb/1.0.0/")
    sys.exit(1)
PYEOF
    fi
fi

# EEG: PhysioNet Motor Movement via wfdb
if $DO_EEG; then
    echo ""
    echo "=== Downloading PhysioNet EEG Motor Movement Dataset ==="
    mkdir -p "$EEG_DIR"

    n_existing=$(find "$EEG_DIR" -name "S[0-9][0-9][0-9]" -type d 2>/dev/null | wc -l)
    if [[ "$n_existing" -ge 109 ]]; then
        echo "  Already present ($n_existing subject directories). Skipping."
    else
        echo "  WARNING: This dataset is ~3 GB and may take 20–60 minutes."
        python3 - << PYEOF
import wfdb, sys, os, shutil
from pathlib import Path
print("  Downloading via wfdb.dl_database('eegmmidb', ...) ...")
try:
    wfdb.dl_database('eegmmidb', '$EEG_DIR')
    # wfdb sometimes creates a nested 'files/' dir — flatten it
    files_dir = Path('$EEG_DIR') / 'files'
    if files_dir.exists():
        for subj in files_dir.iterdir():
            if subj.is_dir():
                dest = Path('$EEG_DIR') / subj.name
                shutil.move(str(subj), str(dest))
        files_dir.rmdir()
    count = len([d for d in os.listdir('$EEG_DIR')
                 if d.startswith('S') and os.path.isdir(os.path.join('$EEG_DIR', d))])
    print(f"  Done. {count} subject directories.")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  Manual download: https://physionet.org/content/eegmmidb/1.0.0/")
    sys.exit(1)
PYEOF
    fi
fi

# PPG: Manual only (UCI does not support direct scripted download)
if $DO_PPG; then
    echo ""
    echo "=== PPG-DaLiA: manual download required ==="
    echo ""
    echo "  1. Open: https://archive.ics.uci.edu/dataset/495/ppg+dalia"
    echo "  2. Click 'Download' and save the ZIP."
    echo "  3. Unzip. Inside you will find PPG_FieldStudy/S1/S1.pkl, etc."
    echo "  4. Copy each S*.pkl file into:"
    echo "       $PPG_DIR/"
    echo "     (flatten — no subdirectories)"
    echo ""
    echo "  Expected result: 15 files named S1.pkl – S15.pkl"
    mkdir -p "$PPG_DIR"
fi

# Final verification
echo ""
echo "=== Verification ==="
python3 - << PYEOF
from pathlib import Path
checks = {
    "MIT-BIH .hea": (list(Path("$ECG_DIR").glob("*.hea")),  48),
    "EEG subject dirs": (list(Path("$EEG_DIR").glob("S[0-9][0-9][0-9]")), 109),
    "PPG .pkl": (list(Path("$PPG_DIR").glob("S*.pkl")), 15),
}
all_ok = True
for label, (found_list, expected) in checks.items():
    n = len(found_list)
    ok = n >= expected
    all_ok = all_ok and ok
    print(f"  {'OK' if ok else 'MISSING':8s}  {label}: {n}/{expected}")
if all_ok:
    print("\nAll datasets ready. Run: python main.py --config config.yaml")
else:
    print("\nSome datasets incomplete — check the paths above.")
PYEOF
