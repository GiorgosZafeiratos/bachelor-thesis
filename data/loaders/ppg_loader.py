import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
from typing import Tuple

class PPGLoader:
    def __init__(self, data_dir: str, channels: list = [0], fs: int = 64,
                 bandpass: Tuple[float, float] = (0.5, 8), window_size: int = 128,
                 overlap: float = 0.5):
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.fs = fs
        self.bandpass = bandpass
        self.window_size = window_size
        self.overlap = overlap

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        nyq = 0.5 * self.fs
        low = self.bandpass[0] / nyq
        high = self.bandpass[1] / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def _segment_signal(self, signal: np.ndarray) -> np.ndarray:
        step = int(self.window_size * (1 - self.overlap))
        segments = []
        for start in range(0, signal.shape[0] - self.window_size + 1, step):
            segments.append(signal[start:start + self.window_size])
        return np.stack(segments, axis=0)

    def load_recording(self, file_name: str) -> np.ndarray:
        df = pd.read_csv(self.data_dir / file_name)
        sig = df.iloc[:, self.channels].values
        sig_resampled = resample(sig, int(sig.shape[0] * self.fs / (1 / (df.iloc[1,0]-df.iloc[0,0]))), axis=0)
        sig_filtered = np.stack([self._bandpass_filter(sig_resampled[:, i]) for i in range(sig_resampled.shape[1])], axis=1)
        segments = np.stack([self._segment_signal(sig_filtered[:, i]) for i in range(sig_filtered.shape[1])], axis=2)
        segments = segments.transpose(0, 2, 1)  # (num_segments, channels, window_size)
        return segments

    def load_all(self) -> np.ndarray:
        all_segments = []
        for file in self.data_dir.glob("*.csv"):
            seg = self.load_recording(file.name)
            all_segments.append(seg)
        return np.concatenate(all_segments, axis=0)
