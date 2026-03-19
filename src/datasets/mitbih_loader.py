import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
from pathlib import Path
from preprocessing.filters import Filters
from preprocessing.segmentation import Segmentation

class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH Arrhythmia recordings.
    Returns (C, T) tensors with labels per segment.
    """
    def __init__(self, data_dir: str, channels: list = [0], fs: int = 360,
                 bandpass: tuple = (0.5, 100), window_size: int = 512,
                 overlap: float = 0.5, label_mapping: dict = None, preload: bool = True):
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.fs = fs
        self.bandpass = bandpass
        self.window_size = window_size
        self.overlap = overlap
        self.label_mapping = label_mapping or {
            'N': 0, 'L':1, 'R':2, 'A':3, 'V':4, 'F':5, 'E':6, 'j':7, '/':8
        }
        self.preload = preload
        self.filters = Filters()
        self.segmenter = Segmentation(window_size, overlap)
        self.segments = []
        self.labels = []
        if self.preload:
            self._preload_data()

    def _bandpass_filter(self, signal):
        return self.filters.bandpass(signal, self.fs, self.bandpass[0], self.bandpass[1])

    def _extract_labels(self, record_name: str, n_segments: int) -> np.ndarray:
        annot_path = self.data_dir / record_name
        annot = wfdb.rdann(str(annot_path), 'atr')
        beat_labels = [self.label_mapping.get(l, 0) for l in annot.symbol]
        # Map to segments
        step = int(self.window_size * (1 - self.overlap))
        segment_labels = []
        for start in range(0, len(annot.symbol) - self.window_size + 1, step):
            window_beats = beat_labels[start:start + self.window_size]
            # Use majority label in segment
            values, counts = np.unique(window_beats, return_counts=True)
            segment_labels.append(values[np.argmax(counts)])
        # Pad if fewer segments than signal
        if len(segment_labels) < n_segments:
            segment_labels.extend([0] * (n_segments - len(segment_labels)))
        return np.array(segment_labels[:n_segments])

    def _preload_data(self):
        for rec_file in self.data_dir.glob("*.dat"):
            record_name = rec_file.stem
            record = wfdb.rdrecord(str(rec_file))
            sig = record.p_signal[:, self.channels]
            sig = self._bandpass_filter(sig)
            segs = self.segmenter.segment(sig)  # (num_segments, channels, window_size)
            lbls = self._extract_labels(record_name, segs.shape[0])
            self.segments.append(segs)
            self.labels.append(lbls)
        self.segments = np.concatenate(self.segments, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.labels) if self.preload else sum(
            self.segmenter.segment(wfdb.rdrecord(str(f)).p_signal[:, self.channels]).shape[0]
            for f in self.data_dir.glob("*.dat")
        )

    def __getitem__(self, idx):
        if self.preload:
            x = torch.tensor(self.segments[idx], dtype=torch.float32)
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        else:
            # Lazy loading (optional, not common)
            raise NotImplementedError("Lazy loading not implemented in this version")
