import torch
from torch.utils.data import Dataset
import mne
import numpy as np
from pathlib import Path
from preprocessing.filters import Filters
from preprocessing.segmentation import Segmentation
from preprocessing.synchronization import Synchronization

class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG Motor Movement/Imagery recordings.
    Returns (C, T) tensors with task labels.
    """
    def __init__(self, data_dir: str, channels: list = None, fs: int = 256,
                 bandpass: tuple = (1, 40), window_size: int = 512, overlap: float = 0.5,
                 preload: bool = True):
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.fs = fs
        self.bandpass = bandpass
        self.window_size = window_size
        self.overlap = overlap
        self.preload = preload
        self.filters = Filters()
        self.segmenter = Segmentation(window_size, overlap)
        self.sync = Synchronization()
        self.segments = []
        self.labels = []
        if self.preload:
            self._preload_data()

    def _bandpass_filter(self, signal):
        return self.filters.bandpass(signal, self.fs, self.bandpass[0], self.bandpass[1])

    def _extract_labels(self, events: np.ndarray, n_segments: int):
        """
        Map EEG event markers to segment labels.
        """
        step = int(self.window_size * (1 - self.overlap))
        segment_labels = []
        for start in range(0, len(events) - self.window_size + 1, step):
            window = events[start:start + self.window_size]
            values, counts = np.unique(window, return_counts=True)
            segment_labels.append(values[np.argmax(counts)])
        if len(segment_labels) < n_segments:
            segment_labels.extend([0]*(n_segments - len(segment_labels)))
        return np.array(segment_labels[:n_segments])

    def _preload_data(self):
        for edf_file in self.data_dir.glob("*.edf"):
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            if self.channels:
                raw.pick_channels(self.channels)
            sig = raw.get_data().T
            sig_resampled = self.sync.resample_signal(sig, raw.info['sfreq'], self.fs)
            sig_filtered = self._bandpass_filter(sig_resampled)
            segments = self.segmenter.segment(sig_filtered)
            # Extract event labels
            events = np.zeros(sig_filtered.shape[0], dtype=int)
            if raw.annotations:
                for annot in raw.annotations:
                    onset_idx = int(annot['onset'] * self.fs)
                    events[onset_idx] = int(annot['description']) if annot['description'].isdigit() else 0
            labels = self._extract_labels(events, segments.shape[0])
            self.segments.append(segments)
            self.labels.append(labels)
        self.segments = np.concatenate(self.segments, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.segments[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
