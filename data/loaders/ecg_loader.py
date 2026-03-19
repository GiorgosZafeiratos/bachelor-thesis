import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
from typing import List, Tuple

class ECGLoader:
    def __init__(self, data_dir: str, channels: List[int] = [0], fs: int = 360,
                 bandpass: Tuple[float, float] = (0.5, 100), window_size: int = 512,
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

    def load_recording(self, record_name: str) -> np.ndarray:
        record_path = self.data_dir / record_name
        record = wfdb.rdrecord(str(record_path))
        sig = record.p_signal[:, self.channels]
        sig_filtered = np.stack([self._bandpass_filter(sig[:, i]) for i in range(sig.shape[1])], axis=1)
        segments = np.stack([self._segment_signal(sig_filtered[:, i]) for i in range(sig_filtered.shape[1])], axis=2)
        # Output shape: (num_segments, window_size, channels)
        segments = segments.transpose(0, 2, 1)
        return segments

    def load_all(self) -> np.ndarray:
        all_segments = []
        for rec_file in self.data_dir.glob("*.dat"):
            name = rec_file.stem
            seg = self.load_recording(name)
            all_segments.append(seg)
        return np.concatenate(all_segments, axis=0)
