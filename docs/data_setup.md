# Data Setup

Three open-access datasets are required. Total disk space: ~4 GB.

---

## 1 · MIT-BIH Arrhythmia Database (ECG)

**Source**: https://physionet.org/content/mitdb/1.0.0/  
**Size**: ~100 MB  
**Format**: Binary WFDB records (`.dat` / `.hea` / `.atr`)

### Download options

**Option A — automated (recommended):**
```bash
bash scripts/download_data.sh --ecg
```

**Option B — via wfdb Python API:**
```python
import wfdb
wfdb.dl_database('mitdb', 'data/raw/mit-bih')
```

**Option C — manual:**
1. Go to https://physionet.org/content/mitdb/1.0.0/
2. Click **Download the ZIP file**
3. Unzip; move all `.dat` / `.hea` / `.atr` files into `data/raw/mit-bih/`

### Expected layout after setup
```
data/raw/mit-bih/
├── 100.dat   100.hea   100.atr
├── 101.dat   101.hea   101.atr
├── ...
└── 234.dat   234.hea   234.atr
```
48 records × 3 files = **144 files** flat in the directory, no subdirectories.

### Verification
```bash
ls data/raw/mit-bih/*.hea | wc -l   # should print 48
```

---

## 2 · PhysioNet EEG Motor Movement / Imagery Dataset

**Source**: https://physionet.org/content/eegmmidb/1.0.0/  
**Size**: ~3 GB  
**Format**: EDF files (`.edf`), one per run, grouped by subject

### Download options

**Option A — automated:**
```bash
bash scripts/download_data.sh --eeg
```

**Option B — via wfdb:**
```python
import wfdb
wfdb.dl_database('eegmmidb', 'data/raw/eeg-motor')
```

**Option C — manual:**
1. Go to https://physionet.org/content/eegmmidb/1.0.0/
2. Download the ZIP file
3. The archive contains a `files/` directory with `S001/`…`S109/` subdirectories
4. Move the `S001/`…`S109/` folders directly into `data/raw/eeg-motor/`
   (do **not** include the intermediate `files/` level)

### Expected layout after setup
```
data/raw/eeg-motor/
├── S001/
│   ├── S001R01.edf
│   ├── S001R02.edf
│   ├── ...
│   └── S001R14.edf
├── S002/
│   └── ...
└── S109/
    └── ...
```
109 subjects × 14 runs = **1526 EDF files**.

### Verification
```bash
ls data/raw/eeg-motor/ | wc -l          # should print 109
ls data/raw/eeg-motor/S001/*.edf | wc -l  # should print 14
```

> **Memory note**: Loading all 109 subjects uses ~12 GB RAM during preprocessing.  
> To run with limited RAM, set `max_subjects=20` in `config.yaml` under `MultimodalPreprocessingPipeline`.  
> Use at least S001–S020 for a valid experiment.

---

## 3 · PPG-DaLiA

**Source**: https://archive.ics.uci.edu/dataset/495/ppg+dalia  
**Size**: ~500 MB  
**Format**: Python pickle files (`.pkl`), one per subject

### Download options

**Option A — automated:**
```bash
bash scripts/download_data.sh --ppg
```

**Option B — manual:**
1. Go to https://archive.ics.uci.edu/dataset/495/ppg+dalia
2. Click **Download**
3. Unzip; the archive extracts to `PPG_FieldStudy/S1/S1.pkl`, `PPG_FieldStudy/S2/S2.pkl`, etc.
4. Copy **only the `.pkl` files** (not the subject folders) into `data/raw/ppg-dalia/`

### Expected layout after setup
```
data/raw/ppg-dalia/
├── S1.pkl
├── S2.pkl
├── ...
└── S15.pkl
```
**15 pickle files** flat in the directory.

### Verification
```bash
ls data/raw/ppg-dalia/*.pkl | wc -l   # should print 15
```

---

## Verify all three datasets

Run this once everything is in place:

```bash
python - << 'EOF'
from pathlib import Path
checks = {
    "MIT-BIH .hea":    (Path("data/raw/mit-bih").glob("*.hea"),  48),
    "EEG subject dirs":(Path("data/raw/eeg-motor").glob("S[0-9][0-9][0-9]"), 109),
    "PPG .pkl":        (Path("data/raw/ppg-dalia").glob("S*.pkl"), 15),
}
for label, (gen, expected) in checks.items():
    found = len(list(gen))
    status = "OK" if found >= expected else f"MISSING ({found}/{expected})"
    print(f"  {label:25s}: {status}")
EOF
```

---

## Dataset citations

```bibtex
@misc{mitbih,
  author = {Moody, George B. and Mark, Roger G.},
  title  = {{MIT-BIH Arrhythmia Database}},
  year   = {1992},
  doi    = {10.13026/C2F305},
}

@misc{eegmmidb,
  author = {Schalk, G. and McFarland, D.J. and Hinterberger, T.
            and Birbaumer, N. and Wolpaw, J.R.},
  title  = {{BCI2000: A General-Purpose Brain-Computer Interface (BCI) System}},
  year   = {2004},
  doi    = {10.13026/C28G6P},
}

@inproceedings{ppgdalia,
  author = {Reiss, Attila and Indlekofer, Ina and Schmidt, Philip and Van Laerhoven, Kristof},
  title  = {Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks},
  year   = {2019},
}
```
