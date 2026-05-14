"""

Supported datasets:
  - MIT-BIH Arrhythmia Database  (ECG) → wfdb
  - PhysioNet EEG Motor Movement       → mne
  - PPG-DaLiA                   (PPG)  → pickle / numpy

Pipeline stages per modality:
  1. Loading
  2. Bandpass / notch filtering
  3. Artifact rejection
  4. Segmentation / epoching
  5. Resampling to a common frequency
  6. Adaptive normalisation  (learnable OR context-aware)
  7. Label encoding & dataset export (numpy .npz)

Dependencies:
  pip install numpy scipy pandas wfdb mne neurokit2 torch scikit-learn tqdm
  
"""

# Imports
import os
import logging
import pickle
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# 1.  UTILITY FUNCTIONS

def bandpass_filter(
    sig: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    sig     : 1-D or 2-D array  (samples,) or (channels, samples)
    lowcut  : lower cut-off frequency in Hz
    highcut : upper cut-off frequency in Hz
    fs      : sampling frequency in Hz
    order   : filter order (default 4 → 8th-order zero-phase via filtfilt)

    Returns
    -------
    Filtered signal, same shape as input.
    """
    nyq = 0.5 * fs
    low = np.clip(lowcut / nyq, 1e-6, 1.0 - 1e-6)
    high = np.clip(highcut / nyq, 1e-6, 1.0 - 1e-6)
    b, a = butter(order, [low, high], btype="band")
    if sig.ndim == 1:
        return filtfilt(b, a, sig)
    return np.array([filtfilt(b, a, ch) for ch in sig])


def notch_filter(
    sig: np.ndarray,
    freq: float,
    fs: float,
    quality: float = 30.0,
) -> np.ndarray:
    """
    IIR notch filter to remove power-line interference (50 or 60 Hz).

    Parameters
    ----------
    freq    : notch frequency (50 or 60 Hz)
    quality : Q-factor; higher value → narrower notch band
    """
    b, a = iirnotch(freq, quality, fs)
    if sig.ndim == 1:
        return filtfilt(b, a, sig)
    return np.array([filtfilt(b, a, ch) for ch in sig])


def resample_signal(
    sig: np.ndarray,
    orig_fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Rational-factor resampling using polyphase filtering.
    Works for 1-D (samples,) and 2-D (channels, samples) arrays.
    """
    from math import gcd
    orig_fs_i = int(orig_fs)
    target_fs_i = int(target_fs)
    g = gcd(orig_fs_i, target_fs_i)
    up = target_fs_i // g
    down = orig_fs_i // g
    if sig.ndim == 1:
        return resample_poly(sig, up, down)
    return np.array([resample_poly(ch, up, down) for ch in sig])


def amplitude_artifact_mask(
    sig: np.ndarray,
    threshold_std: float = 5.0,
) -> np.ndarray:
    """
    Returns boolean mask (True = clean sample) based on z-score amplitude.
    """
    z = np.abs((sig - np.mean(sig)) / (np.std(sig) + 1e-8))
    return z < threshold_std


def segment_signal(
    sig: np.ndarray,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """
    Sliding-window segmentation.

    Parameters
    ----------
    sig         : 1-D (samples,) or 2-D (channels, samples)
    window_size : number of samples per window
    step_size   : hop between window starts

    Returns
    -------
    segments : (n_segments, channels, window_size)
    """
    if sig.ndim == 1:
        sig = sig[np.newaxis, :]
    n_channels, n_samples = sig.shape
    starts = range(0, n_samples - window_size + 1, step_size)
    return np.array([sig[:, s: s + window_size] for s in starts])


# 2.  ECG PREPROCESSOR  —  MIT-BIH Arrhythmia Database

class ECGPreprocessor:
    """
    Preprocessing pipeline for the MIT-BIH Arrhythmia Database.

    Workflow
    --------
    load → bandpass → baseline-wander removal → R-peak detection
    → beat segmentation → artifact rejection → resample → label encode

    MIT-BIH annotations are mapped to the 5-class AAMI standard:
        N  Normal / LBBB / RBBB / Atrial escape / Nodal escape
        S  Atrial premature / Aberrant atrial premature / ...
        V  Premature ventricular contraction / Ventricular escape
        F  Fusion of ventricular and normal beat
        Q  Paced / Fusion of paced and normal / Unclassifiable
    """

    AAMI_MAP: Dict[str, str] = {
        # Normal
        "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        # Supraventricular ectopic
        "A": "S", "a": "S", "J": "S", "S": "S",
        # Ventricular ectopic
        "V": "V", "E": "V",
        # Fusion
        "F": "F",
        # Unknown / paced
        "P": "Q", "/": "Q", "f": "Q", "Q": "Q",
    }

    def __init__(
        self,
        data_dir: str,
        target_fs: float = 360.0,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        pre_r_samples: int = 90,   # 250 ms before R-peak at 360 Hz
        post_r_samples: int = 270, # 750 ms after  R-peak at 360 Hz
        max_records: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.pre_r = pre_r_samples
        self.post_r = post_r_samples
        self.window_samples = pre_r_samples + post_r_samples  # 360
        self.max_records = max_records

    # ------------------------------------------------------------------
    def _load_record(self, record_id: str):
        try:
            import wfdb
        except ImportError:
            raise ImportError("Install wfdb: pip install wfdb")
        record_path = str(self.data_dir / record_id)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")
        return record, annotation

    # ------------------------------------------------------------------
    def _baseline_wander_removal(
        self, sig: np.ndarray, fs: float
    ) -> np.ndarray:
        """
        Cascade of two median filters (200 ms + 600 ms windows) is used
        to estimate and subtract the baseline — standard clinical approach.
        """
        from scipy.ndimage import median_filter
        w1 = int(0.2 * fs)
        w2 = int(0.6 * fs)
        baseline = median_filter(median_filter(sig, size=w1, mode="nearest"),
                                 size=w2, mode="nearest")
        return sig - baseline

    # ------------------------------------------------------------------
    def _detect_r_peaks(self, sig: np.ndarray, fs: float) -> np.ndarray:
        """
        R-peak detection using neurokit2 (Pan-Tompkins algorithm).
        Falls back to scipy peak detection when neurokit2 is unavailable.
        """
        try:
            import neurokit2 as nk
            _, info = nk.ecg_peaks(sig, sampling_rate=int(fs))
            return info["ECG_R_Peaks"]
        except ImportError:
            log.warning("neurokit2 not found — falling back to scipy peaks.")
            min_distance = int(0.3 * fs)
            peaks, _ = sp_signal.find_peaks(
                sig, distance=min_distance, height=np.mean(sig)
            )
            return peaks

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_or_trim(sig: np.ndarray, target_len: int) -> np.ndarray:
        if len(sig) >= target_len:
            return sig[:target_len]
        return np.pad(sig, (0, target_len - len(sig)), mode="edge")

    @staticmethod
    def _is_valid_beat(beat: np.ndarray, z_thresh: float = 6.0) -> bool:
        z = np.abs((beat - np.mean(beat)) / (np.std(beat) + 1e-8))
        return bool(np.max(z) < z_thresh)

    # ------------------------------------------------------------------
    def process_record(
        self, record_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single MIT-BIH record.

        Returns
        -------
        beats  : (n_beats, 1, window_samples)
        labels : (n_beats,) string AAMI labels
        """
        record, annotation = self._load_record(record_id)
        fs = record.fs
        sig = record.p_signal[:, 0].astype(np.float32) # MLII lead

        # Filtering
        sig = bandpass_filter(sig, self.lowcut, self.highcut, fs)
        sig = self._baseline_wander_removal(sig, fs)

        # Resample to target_fs
        if fs != self.target_fs:
            scale = self.target_fs / fs
            ann_samples = (annotation.sample * scale).astype(int)
            sig = resample_signal(sig, fs, self.target_fs)
        else:
            ann_samples = annotation.sample
        fs = self.target_fs
        n_samples = len(sig)

        # Beat segmentation
        beats, labels = [], []
        for sample, symbol in zip(ann_samples, annotation.symbol):
            aami = self.AAMI_MAP.get(symbol)
            if aami is None:
                continue

            start = sample - self.pre_r
            end = sample + self.post_r
            if start < 0 or end > n_samples:
                continue

            beat = self._pad_or_trim(sig[start:end], self.window_samples)

            if not self._is_valid_beat(beat):
                continue

            beats.append(beat[np.newaxis, :])  # (1, window)
            labels.append(aami)

        if not beats:
            return np.empty((0, 1, self.window_samples)), np.array([])

        return np.array(beats, dtype=np.float32), np.array(labels)

    # ------------------------------------------------------------------
    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process all records in data_dir.

        Returns
        -------
        X           : (total_beats, 1, window_samples)
        y           : (total_beats,) integer-encoded AAMI labels
        classes     : class name array from LabelEncoder
        subject_ids : (total_beats,) integer record index per beat
        """
        try:
            import wfdb
            if not any(self.data_dir.glob("*.hea")):
                record_list = wfdb.get_record_list("mitdb")
            else:
                record_list = [p.stem for p in self.data_dir.glob("*.hea")]
        except Exception:
            record_list = [p.stem for p in self.data_dir.glob("*.hea")]

        if self.max_records:
            record_list = record_list[: self.max_records]

        all_beats, all_labels, all_subject_ids = [], [], []
        for rec_idx, rid in enumerate(tqdm(record_list, desc="ECG records")):
            try:
                beats, labels = self.process_record(rid)
                if len(beats) > 0:
                    all_beats.append(beats)
                    all_labels.append(labels)
                    all_subject_ids.append(
                        np.full(len(beats), rec_idx, dtype=np.int64)
                    )
            except Exception as exc:
                log.warning(f"Skipping record {rid}: {exc}")

        X = np.concatenate(all_beats, axis=0)
        y_str = np.concatenate(all_labels, axis=0)
        subject_ids = np.concatenate(all_subject_ids, axis=0)
        le = LabelEncoder()
        y = le.fit_transform(y_str).astype(np.int64)
        log.info(f"ECG dataset: {X.shape} | classes: {le.classes_}")
        return X, y, le.classes_, subject_ids


# 3.  EEG PREPROCESSOR  —  PhysioNet EEG Motor Movement/Imagery Dataset

class EEGPreprocessor:
    """
    Preprocessing pipeline for the PhysioNet EEG Motor Movement Dataset
    (eegmmidb), sampled at 160 Hz across 64 EEG channels.

    Task / run mapping:
        Runs 3, 7, 11  →  T1 = left fist       /  T2 = right fist
        Runs 4, 8, 12  →  T1 = both fists       /  T2 = both feet
        Run  1         →  baseline (eyes open)
        Run  2         →  baseline (eyes closed)

    Workflow
    --------
    load EDF → notch filter → bandpass filter → epoch around annotations
    → amplitude artifact rejection → resample → label encode
    """

    LABEL_MAP: Dict[str, int] = {"T0": 0, "T1": 1, "T2": 2}

    def __init__(
        self,
        data_dir: str,
        target_fs: float = 160.0,
        lowcut: float = 1.0,
        highcut: float = 50.0,
        notch_freq: float = 60.0,            # US power-line frequency
        tmin: float = 0.0,
        tmax: float = 4.0,                   # 4-second epoch
        amplitude_threshold: float = 100e-6, # 100 µV artifact threshold
        n_channels: int = 64,
        max_subjects: Optional[int] = None,
        runs: Optional[List[int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.tmin = tmin
        self.tmax = tmax
        self.amplitude_threshold = amplitude_threshold
        self.n_channels = n_channels
        self.max_subjects = max_subjects
        self.runs = runs or [3, 4, 7, 8, 11, 12] # motor imagery runs only

    # ------------------------------------------------------------------
    def process_subject(
        self, subject_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all selected runs for one subject.

        Returns
        -------
        epochs : (n_epochs, n_channels, n_times)
        labels : (n_epochs,) integer labels  {0=T0, 1=T1, 2=T2}
        """
        try:
            import mne
            mne.set_log_level("WARNING")
        except ImportError:
            raise ImportError("Install MNE: pip install mne")

        subject_str = f"S{subject_id:03d}"
        subject_dir = self.data_dir / subject_str

        all_epochs, all_labels = [], []

        for run in self.runs:
            edf_path = subject_dir / f"{subject_str}R{run:02d}.edf"
            if not edf_path.exists():
                continue

            try:
                raw = mne.io.read_raw_edf(
                    str(edf_path), preload=True, verbose=False
                )
            except Exception as exc:
                log.warning(f"Cannot read {edf_path}: {exc}")
                continue

            fs = raw.info["sfreq"]

            # Filtering
            raw.notch_filter(self.notch_freq, verbose=False)
            raw.filter(self.lowcut, self.highcut, verbose=False)

            # Channel selection (keep first n_channels)
            if len(raw.ch_names) > self.n_channels:
                raw.pick(raw.ch_names[: self.n_channels])

            # Epoching
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if len(events) == 0:
                continue

            valid_ids = {k: v for k, v in event_id.items()
                         if k in ("T0", "T1", "T2")}
            if not valid_ids:
                continue

            try:
                epochs = mne.Epochs(
                    raw, events, event_id=valid_ids,
                    tmin=self.tmin, tmax=self.tmax,
                    baseline=None, preload=True, verbose=False,
                )
            except Exception:
                continue

            epoch_data = epochs.get_data() # (n, ch, time)
            inv_id = {v: k for k, v in valid_ids.items()}
            epoch_labels_str = [inv_id[e] for e in epochs.events[:, 2]]

            clean_data, clean_labels = [], []
            for ep, lb in zip(epoch_data, epoch_labels_str):
                if np.max(np.abs(ep)) <= self.amplitude_threshold:
                    clean_data.append(ep)
                    clean_labels.append(self.LABEL_MAP.get(lb, -1))

            if clean_data:
                ep_arr = np.array(clean_data, dtype=np.float32)
                if fs != self.target_fs:
                    ep_arr = np.array([
                        resample_signal(e, fs, self.target_fs)
                        for e in ep_arr
                    ])
                all_epochs.append(ep_arr)
                all_labels.extend(clean_labels)

        if not all_epochs:
            return np.empty((0, self.n_channels, 1)), np.array([])

        return (
            np.concatenate(all_epochs, axis=0),
            np.array(all_labels, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process all subjects found in data_dir.

        Returns
        -------
        X           : (total_epochs, n_channels, n_times)
        y           : (total_epochs,) integer labels
        subject_ids : (total_epochs,) integer subject index per epoch
        """
        subject_dirs = sorted(self.data_dir.glob("S[0-9][0-9][0-9]"))
        subject_ids_list = [int(p.name[1:]) for p in subject_dirs]
        if self.max_subjects:
            subject_ids_list = subject_ids_list[: self.max_subjects]

        all_X, all_y, all_subject_ids = [], [], []
        for sid in tqdm(subject_ids_list, desc="EEG subjects"):
            try:
                X_sub, y_sub = self.process_subject(sid)
                if len(X_sub) > 0:
                    all_X.append(X_sub)
                    all_y.append(y_sub)
                    all_subject_ids.append(
                        np.full(len(X_sub), sid, dtype=np.int64)
                    )
            except Exception as exc:
                log.warning(f"Skipping subject S{sid:03d}: {exc}")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        subject_ids = np.concatenate(all_subject_ids, axis=0)
        valid = y >= 0
        X, y, subject_ids = X[valid], y[valid], subject_ids[valid]
        log.info(f"EEG dataset: {X.shape} | classes: {np.unique(y)}")
        return X, y, subject_ids


# 4.  PPG PREPROCESSOR  —  PPG-DaLiA Dataset

class PPGPreprocessor:
    """
    Preprocessing pipeline for the PPG-DaLiA dataset.
    Wrist BVP (PPG) sampled at 64 Hz.  Activity labels at 4 Hz.

    Activity classes:
        1=sitting  2=ascending stairs  3=descending stairs
        4=table soccer  5=cycling  6=driving
        7=lunch break  8=walking  9=working

    Workflow
    --------
    load pickle → bandpass → per-window amplitude artifact rejection
    → sliding window segmentation → majority-vote labelling
    → resample → label encode
    """

    def __init__(
        self,
        data_dir: str,
        target_fs: float = 64.0,
        lowcut: float = 0.5,
        highcut: float = 8.0,
        window_sec: float = 8.0,
        step_sec: float = 2.0,
        amplitude_thresh_std: float = 4.0,
        max_subjects: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.amplitude_thresh_std = amplitude_thresh_std
        self.max_subjects = max_subjects

    # ------------------------------------------------------------------
    def _load_subject(self, pkl_path: Path) -> Dict:
        with open(pkl_path, "rb") as f:
            return pickle.load(f, encoding="latin1")

    # ------------------------------------------------------------------
    def process_subject(
        self, pkl_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single PPG-DaLiA subject pickle.

        Returns
        -------
        segments : (n_segments, 1, window_samples)
        labels   : (n_segments,) integer activity labels
        """
        data = self._load_subject(pkl_path)
        ppg = data["signal"]["wrist"]["BVP"].flatten().astype(np.float32)
        activity = data["activity"].flatten().astype(np.int64)     # 4 Hz
        fs = 64.0 # fixed in PPG-DaLiA

        # Filtering
        ppg = bandpass_filter(ppg, self.lowcut, self.highcut, fs)

        # Resample if necessary
        if fs != self.target_fs:
            ppg = resample_signal(ppg, fs, self.target_fs)
            fs = self.target_fs

        window_samples = int(self.window_sec * fs)
        step_samples = int(self.step_sec * fs)

        # Upsample activity labels to match PPG sample rate (4 Hz → fs)
        label_upsample = int(fs / 4.0)
        activity_full = np.repeat(activity, label_upsample)

        min_len = min(len(ppg), len(activity_full))
        ppg = ppg[:min_len]
        activity_full = activity_full[:min_len]

        # Sliding window segmentation
        segments, labels = [], []
        for s in range(0, len(ppg) - window_samples + 1, step_samples):
            seg = ppg[s: s + window_samples]
            lbl_window = activity_full[s: s + window_samples]

            # Artifact rejection: reject window if >10 % samples are outliers
            bad_frac = np.mean(
                ~amplitude_artifact_mask(seg, self.amplitude_thresh_std)
            )
            if bad_frac > 0.10:
                continue

            label = int(np.bincount(lbl_window).argmax())
            if label == 0:
                continue # skip unlabelled transitions

            segments.append(seg[np.newaxis, :])
            labels.append(label)

        if not segments:
            return np.empty((0, 1, window_samples)), np.array([])

        return (
            np.array(segments, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process all subject pickle files in data_dir.

        Returns
        -------
        X           : (total_segments, 1, window_samples)
        y           : (total_segments,) integer-encoded labels
        classes     : class name array from LabelEncoder
        subject_ids : (total_segments,) integer subject index per segment
        """
        pkl_files = sorted(self.data_dir.glob("S*.pkl"))
        if self.max_subjects:
            pkl_files = pkl_files[: self.max_subjects]

        all_X, all_y, all_subject_ids = [], [], []
        for subj_idx, pkl_path in enumerate(tqdm(pkl_files, desc="PPG subjects")):
            try:
                X_sub, y_sub = self.process_subject(pkl_path)
                if len(X_sub) > 0:
                    all_X.append(X_sub)
                    all_y.append(y_sub)
                    all_subject_ids.append(
                        np.full(len(X_sub), subj_idx, dtype=np.int64)
                    )
            except Exception as exc:
                log.warning(f"Skipping {pkl_path.name}: {exc}")

        X = np.concatenate(all_X, axis=0)
        y_raw = np.concatenate(all_y, axis=0)
        subject_ids = np.concatenate(all_subject_ids, axis=0)
        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(np.int64)
        log.info(f"PPG dataset: {X.shape} | classes: {le.classes_}")
        return X, y, le.classes_, subject_ids


# 5.  ADAPTIVE NORMALIZATION

class ContextAwareNormalizer:
    """
    Offline context-aware normalization (NumPy).

    For each segment, statistics are computed from a sliding context window
    centred on that segment rather than from global dataset statistics.
    This adapts to local amplitude drifts, patient-specific baselines
    and cross-sensor variability — addressing the gap identified in
    the thesis proposal regarding generic static normalization methods.

    Modes
    -----
    'zscore'  : (x - mu_local) / sigma_local
    'minmax'  : (x - min_local) / (max_local - min_local)
    'robust'  : (x - median_local) / IQR_local
    """

    def __init__(
        self,
        mode: str = "zscore",
        context_window: int = 10,
    ):
        assert mode in ("zscore", "minmax", "robust"), \
            "mode must be 'zscore', 'minmax', or 'robust'"
        self.mode = mode
        self.context_window = context_window

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n_segments, n_channels, n_times)

        Returns
        -------
        X_norm : same shape, context-normalised float32 array
        """
        n = X.shape[0]
        X_norm = np.empty_like(X)
        hw = self.context_window // 2

        for i in range(n):
            lo = max(0, i - hw)
            hi = min(n, i + hw + 1)
            ctx = X[lo:hi]          # (ctx_len, ch, time)

            if self.mode == "zscore":
                # Per-channel statistics over context segments and time
                flat = ctx.reshape(ctx.shape[1], -1)   # (ch, ctx*time)
                mu = flat.mean(axis=1)                 # (ch,)
                sigma = flat.std(axis=1) + 1e-8
                X_norm[i] = (X[i] - mu[:, np.newaxis]) / sigma[:, np.newaxis]

            elif self.mode == "minmax":
                flat = ctx.reshape(ctx.shape[1], -1)
                mn = flat.min(axis=1)
                mx = flat.max(axis=1)
                rng = (mx - mn) + 1e-8
                X_norm[i] = (X[i] - mn[:, np.newaxis]) / rng[:, np.newaxis]

            elif self.mode == "robust":
                flat = ctx.reshape(ctx.shape[1], -1)
                med = np.median(flat, axis=1)
                q75 = np.percentile(flat, 75, axis=1)
                q25 = np.percentile(flat, 25, axis=1)
                iqr = (q75 - q25) + 1e-8
                X_norm[i] = (X[i] - med[:, np.newaxis]) / iqr[:, np.newaxis]

        return X_norm.astype(np.float32)


class LearnableNormLayer(nn.Module):
    """
    Learnable adaptive normalisation layer (PyTorch module).

    Combines instance normalisation with a context-projection network
    that produces per-sample adaptive scale (gamma) and shift (beta)
    corrections, allowing the model to condition normalisation behaviour
    on the local signal statistics at training time.

    Architecture
    ------------
    Given x of shape (B, C, T):

        mu, sigma  = per-channel mean and std over T   → (B, C)
        x_norm     = (x - mu) / sigma                  → (B, C, T)
        stats      = concat([mu, sigma])               → (B, 2C)
        g_a, b_a   = context_proj(stats)               → each (B, C)
        out        = (gamma + g_a) * x_norm + (beta + b_a)

    gamma and beta are global learnable parameters (C,).
    g_a and b_a are input-dependent corrections.
    """

    def __init__(
        self,
        n_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps

        if affine:
            self.gamma = nn.Parameter(torch.ones(n_channels))
            self.beta = nn.Parameter(torch.zeros(n_channels))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        # Context projection: (B, 2C) → (B, 2C) adaptive corrections
        self.context_proj = nn.Sequential(
            nn.Linear(2 * n_channels, n_channels),
            nn.Tanh(),
            nn.Linear(n_channels, 2 * n_channels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.context_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        out : (B, C, T)
        """
        mu = x.mean(dim=2)                    # (B, C)
        sigma = x.std(dim=2) + self.eps       # (B, C)
        x_norm = (x - mu.unsqueeze(2)) / sigma.unsqueeze(2)

        stats = torch.cat([mu, sigma], dim=1) # (B, 2C)
        adapt = self.context_proj(stats)      # (B, 2C)
        g_a, b_a = adapt.chunk(2, dim=1)      # each (B, C)

        g = (self.gamma.unsqueeze(0) + g_a).unsqueeze(2) # (B, C, 1)
        b = (self.beta.unsqueeze(0) + b_a).unsqueeze(2)  # (B, C, 1)

        return g * x_norm + b


# 6.  UNIFIED MULTIMODAL PIPELINE

class MultimodalPreprocessingPipeline:
    """
    Orchestrates the full preprocessing pipeline across all three
    modalities and exports analysis-ready compressed numpy archives.

    Output files
    ------------
    <output_dir>/
        ecg_preprocessed.npz   →  X (n, 1,  360), y (n,), classes
        eeg_preprocessed.npz   →  X (n, 64, 641), y (n,)
        ppg_preprocessed.npz   →  X (n, 1,  512), y (n,), classes
        dataset_summary.csv    →  per-class sample counts

    Usage
    -----
        pipeline = MultimodalPreprocessingPipeline(
            ecg_dir    = "data/mitdb",
            eeg_dir    = "data/eegmmidb",
            ppg_dir    = "data/ppg-dalia",
            output_dir = "data/processed",
            norm_mode  = "context",
        )
        pipeline.run()
    """

    def __init__(
        self,
        ecg_dir: Optional[str] = None,
        eeg_dir: Optional[str] = None,
        ppg_dir: Optional[str] = None,
        output_dir: str = "data/processed",
        norm_mode: str = "context",          # 'context' | 'zscore' | 'learnable'
        context_norm_method: str = "zscore", # 'zscore'  | 'minmax' | 'robust'
        context_window: int = 10,
        target_fs: float = 128.0,
        max_records: Optional[int] = None,
        random_seed: int = 42,
    ):
        self.ecg_dir = ecg_dir
        self.eeg_dir = eeg_dir
        self.ppg_dir = ppg_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.norm_mode = norm_mode
        self.context_norm_method = context_norm_method
        self.context_window = context_window
        self.target_fs = target_fs
        self.max_records = max_records
        np.random.seed(random_seed)

        self._normalizer = ContextAwareNormalizer(
            mode=context_norm_method,
            context_window=context_window,
        )

    # ------------------------------------------------------------------
    def _apply_normalisation(self, X: np.ndarray) -> np.ndarray:
        if self.norm_mode == "none":
            # Pass-through: model-level LearnableNorm handles normalisation.
            # This is the correct path for the proposed system
            # (preprocessing_norm="none", model_norm="learnable").
            log.info("  Normalisation mode='none' — raw filtered signals passed through.")
            return X.astype(np.float32)
        elif self.norm_mode == "context":
            log.info("  Applying context-aware normalisation …")
            return self._normalizer.fit_transform(X)
        elif self.norm_mode in ("zscore", "learnable"):
            # For 'learnable', we export z-score offline;
            # LearnableNormLayer handles adaptation inside the model.
            log.info("  Applying global z-score normalisation …")
            mu = X.mean(axis=(0, 2), keepdims=True)
            sigma = X.std(axis=(0, 2), keepdims=True) + 1e-8
            return ((X - mu) / sigma).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown norm_mode: '{self.norm_mode}'. "
                f"Valid options: 'none' | 'context' | 'zscore' | 'learnable'"
            )

    # ------------------------------------------------------------------
    def _save(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ):
        out_path = self.output_dir / f"{name}_preprocessed.npz"
        save_dict = {"X": X, "y": y}
        if classes is not None:
            save_dict["classes"] = classes
        np.savez_compressed(str(out_path), **save_dict)
        log.info(f"  Saved NPZ → {out_path}  shape={X.shape}")

    # ------------------------------------------------------------------
    def _save_hdf5(
        self,
        name:        str,
        X:           np.ndarray,
        y:           np.ndarray,
        subject_ids: np.ndarray,
        classes:     Optional[np.ndarray] = None,
        val_frac:    float = 0.15,
        test_frac:   float = 0.15,
        seed:        int   = 42,
    ):
        """
        Subject-wise train/val/test split + HDF5 export.

        Saves:  <output_dir>/<name>.h5
        Schema: /<name>/train/X, /<name>/train/y, /<name>/train/subject_ids
                /<name>/val/...   /<name>/test/...
                /<name>.attrs["label_map"]  — JSON {int: class_name}
                /<name>.attrs["n_classes"]

        Using subject-wise splitting here (at preprocessing time) ensures
        ALL downstream training — baselines, hybrid, and ablations — use
        the same split, making results directly comparable.
        """
        try:
            import h5py
            from sklearn.model_selection import train_test_split
            import json
        except ImportError as e:
            log.error("HDF5 export requires h5py and scikit-learn: %s", e)
            return

        unique_subjects = np.unique(subject_ids)
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_subjects)

        n_subj  = len(unique_subjects)
        n_test  = max(1, int(n_subj * test_frac))
        n_val   = max(1, int(n_subj * val_frac))
        n_train = n_subj - n_test - n_val

        if n_train <= 0:
            log.warning(
                "Too few subjects (%d) for subject-wise split — "
                "falling back to random sample split.", n_subj
            )
            idx = np.arange(len(X))
            idx_tv, idx_test = train_test_split(
                idx, test_size=test_frac, stratify=y, random_state=seed
            )
            val_adj = val_frac / (1.0 - test_frac)
            idx_train, idx_val = train_test_split(
                idx_tv, test_size=val_adj, stratify=y[idx_tv], random_state=seed
            )
        else:
            subj_train = unique_subjects[:n_train]
            subj_val   = unique_subjects[n_train:n_train + n_val]
            subj_test  = unique_subjects[n_train + n_val:]
            idx_train  = np.where(np.isin(subject_ids, subj_train))[0]
            idx_val    = np.where(np.isin(subject_ids, subj_val))[0]
            idx_test   = np.where(np.isin(subject_ids, subj_test))[0]

        # Build label map: int → class_name string
        if classes is not None:
            label_map = {str(i): str(c) for i, c in enumerate(classes)}
        else:
            label_map = {str(i): str(i) for i in np.unique(y)}

        out_path = str(self.output_dir / f"{name}.h5")
        with h5py.File(out_path, "w") as f:
            grp = f.create_group(name)
            grp.attrs["label_map"] = json.dumps(label_map)
            grp.attrs["n_classes"] = int(y.max()) + 1

            for split_name, indices in [
                ("train", idx_train),
                ("val",   idx_val),
                ("test",  idx_test),
            ]:
                sg = grp.create_group(split_name)
                sg.create_dataset("X",           data=X[indices].astype(np.float32),
                                  compression="gzip", compression_opts=4)
                sg.create_dataset("y",           data=y[indices].astype(np.int32))
                sg.create_dataset("subject_ids", data=subject_ids[indices].astype(np.int64))
                log.info(
                    "  HDF5 [%s/%s]: %d samples | %d subjects",
                    name, split_name, len(indices),
                    len(np.unique(subject_ids[indices])),
                )

        log.info(f"  Saved HDF5 → {out_path}  shape={X.shape}")

    # ------------------------------------------------------------------
    def _summarise(
        self,
        name: str,
        y: np.ndarray,
        classes: Optional[np.ndarray],
    ) -> pd.DataFrame:
        unique, counts = np.unique(y, return_counts=True)
        label_names = (
            classes[unique] if classes is not None else unique.astype(str)
        )
        return pd.DataFrame({
            "modality": name,
            "class": label_names,
            "n_samples": counts,
        })

    # ------------------------------------------------------------------
    def run(self):
        """Execute the full preprocessing pipeline for all modalities."""
        summary_frames = []

        # ECG
        if self.ecg_dir:
            log.info("=" * 60)
            log.info("Processing ECG — MIT-BIH Arrhythmia Database …")
            ecg_pp = ECGPreprocessor(
                data_dir=self.ecg_dir,
                target_fs=self.target_fs,
                max_records=self.max_records,
            )
            X, y, classes, subject_ids = ecg_pp.process_dataset()
            X = self._apply_normalisation(X)
            self._save("ecg", X, y, classes)
            self._save_hdf5("ecg", X, y, subject_ids, classes)
            summary_frames.append(self._summarise("ECG", y, classes))

        # EEG
        if self.eeg_dir:
            log.info("=" * 60)
            log.info("Processing EEG — PhysioNet Motor Movement Dataset …")
            eeg_pp = EEGPreprocessor(
                data_dir=self.eeg_dir,
                target_fs=self.target_fs,
                max_subjects=self.max_records,
            )
            X, y, subject_ids = eeg_pp.process_dataset()
            X = self._apply_normalisation(X)
            self._save("eeg", X, y)
            classes_eeg = np.array(["T0 (rest)", "T1 (left/both fists)",
                                     "T2 (right fist/both feet)"])
            self._save_hdf5("eeg", X, y, subject_ids, classes_eeg)
            summary_frames.append(self._summarise("EEG", y, classes_eeg))

        # PPG
        if self.ppg_dir:
            log.info("=" * 60)
            log.info("Processing PPG — PPG-DaLiA Dataset …")
            ppg_pp = PPGPreprocessor(
                data_dir=self.ppg_dir,
                target_fs=self.target_fs,
                max_subjects=self.max_records,
            )
            X, y, classes, subject_ids = ppg_pp.process_dataset()
            X = self._apply_normalisation(X)
            self._save("ppg", X, y, classes)
            self._save_hdf5("ppg", X, y, subject_ids, classes)
            summary_frames.append(self._summarise("PPG", y, classes))

        # Dataset summary
        if summary_frames:
            summary = pd.concat(summary_frames, ignore_index=True)
            csv_path = self.output_dir / "dataset_summary.csv"
            summary.to_csv(str(csv_path), index=False)
            log.info("=" * 60)
            log.info(f"Summary saved → {csv_path}")
            log.info("\n" + summary.to_string(index=False))

        log.info("=" * 60)
        log.info("Preprocessing pipeline complete.")


# 7.  ENTRY POINT

if __name__ == "__main__":
    """
    Edit paths below to match your local dataset directories.

    Expected layout:
        data/
          mitdb/       ← *.hea  *.dat  *.atr  (MIT-BIH records)
          eegmmidb/    ← S001/R01.edf … S109/R14.edf
          ppg-dalia/   ← S1.pkl … S15.pkl
          processed/   ← created automatically
    """
    pipeline = MultimodalPreprocessingPipeline(
        ecg_dir="data/mitdb",
        eeg_dir="data/eegmmidb",
        ppg_dir="data/ppg-dalia",
        output_dir="data/processed",
        norm_mode="context",          # 'context' | 'zscore' | 'learnable'
        context_norm_method="zscore", # 'zscore'  | 'minmax' | 'robust'
        context_window=10,
        target_fs=128.0,
    )
    pipeline.run()
