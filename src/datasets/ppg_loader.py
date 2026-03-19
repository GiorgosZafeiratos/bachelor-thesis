import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from preprocessing.filters import Filters
from preprocessing.segmentation import Segmentation
from preprocessing.synchronization import Synchronization

class PPGDataset(Dataset):
    """
    PyTorch Dataset for PPG signals.
    Returns (C, T) tensors with activity or HR labels.
    """
    def __init__(self, data_dir: str, channels: list = [0], fs: int = 64,
                 bandpass: tuple = (0.5, 8), window_size: int = 128, overlap: float = 0.5,
                 label_column: str = 'label', preload: bool = True):
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.fs = fs
        self.bandpass = bandpass
        self.window_size = window_size
        self.overlap = overlap
        self.label_column = label_column
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

    def _extract_labels(self, labels_arr: np.ndarray, n_segments: int):
        step = int(self.window_size * (1 - self.overlap))
        segment_labels = []
        for start in range(0, len(labels_arr) - self.window_size + 1, step):
            window = labels_arr[start:start+self.window_size]
            values, counts = np.unique(window, return_counts=True)
            segment_labels.append(values[np.argmax(counts)])
        if len(segment_labels) < n_segments:
            segment_labels.extend([0]*(n_segments - len(segment_labels)))
        return np.array(segment_labels[:n_segments])

    def _preload_data(self):
        for file in self.data_dir.glob("*.csv"):
            df = pd.read_csv(file)
            sig = df.iloc[:, self.channels].values
            labels_arr = df[self.label_column].values if self.label_column in df else np.zeros(sig.shape[0])
            sig_resampled = self.sync.resample_signal(sig, 1/(df.iloc[1,0]-df.iloc[0,0]), self.fs)
            sig_filtered = self._bandpass_filter(sig_resampled)
            segments = self.segmenter.segment(sig_filtered)
            labels = self._extract_labels(labels_arr, segments.shape[0])
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
