import mne
import numpy as np
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
from typing import List, Tuple

class EEGLoader:
    def __init__(self, data_dir: str, channels: List[str] = None, fs: int = 256,
                 bandpass: Tuple[float, float] = (1, 40), window_size: int = 512,
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

    def load_recording(self, edf_file: str) -> np.ndarray:
        raw = mne.io.read_raw_edf(str(self.data_dir / edf_file), preload=True, verbose=False)
        if self.channels:
            raw.pick_channels(self.channels)
        sig = raw.get_data().T  # shape: (samples, channels)
        sig_resampled = resample(sig, int(sig.shape[0] * self.fs / raw.info['sfreq']), axis=0)
        sig_filtered = np.stack([self._bandpass_filter(sig_resampled[:, i]) for i in range(sig_resampled.shape[1])], axis=1)
        segments = np.stack([self._segment_signal(sig_filtered[:, i]) for i in range(sig_filtered.shape[1])], axis=2)
        segments = segments.transpose(0, 2, 1)  # (num_segments, channels, window_size)
        return segments

    def load_all(self) -> np.ndarray:
        all_segments = []
        for edf_file in self.data_dir.glob("*.edf"):
            seg = self.load_recording(edf_file.name)
            all_segments.append(seg)
        return np.concatenate(all_segments, axis=0)
