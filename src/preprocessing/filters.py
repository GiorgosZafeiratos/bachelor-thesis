import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

class Filters:
    @staticmethod
    def bandpass(signal: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
        """
        Apply a zero-phase Butterworth bandpass filter.
        :param signal: (samples, channels)
        :param fs: Sampling frequency
        :param lowcut: Low cutoff frequency
        :param highcut: High cutoff frequency
        :param order: Filter order
        :return: Filtered signal
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal, axis=0)

    @staticmethod
    def lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, signal, axis=0)

    @staticmethod
    def highpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        return filtfilt(b, a, signal, axis=0)

    @staticmethod
    def notch(signal: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
        """
        Remove powerline noise using a notch filter.
        """
        b, a = iirnotch(freq / (fs / 2), quality)
        return filtfilt(b, a, signal, axis=0)

    @staticmethod
    def remove_artifacts(signal: np.ndarray, method: str = "zscore", threshold: float = 3.0) -> np.ndarray:
        """
        Remove artifacts based on statistical thresholds.
        """
        if method == "zscore":
            z = (signal - np.mean(signal, axis=0)) / np.std(signal, axis=0)
            signal_clean = np.where(np.abs(z) > threshold, np.median(signal, axis=0), signal)
            return signal_clean
        elif method == "clipping":
            return np.clip(signal, -threshold, threshold)
        else:
            raise ValueError(f"Unknown artifact removal method {method}")
