import numpy as np
from scipy.interpolate import interp1d

class Synchronization:
    @staticmethod
    def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
        """
        Resample a 2D signal (samples x channels) to target_fs using linear interpolation.
        """
        n_samples, n_channels = signal.shape
        duration = n_samples / orig_fs
        t_orig = np.linspace(0, duration, n_samples)
        n_target = int(duration * target_fs)
        t_new = np.linspace(0, duration, n_target)
        resampled = np.zeros((n_target, n_channels))
        for ch in range(n_channels):
            f = interp1d(t_orig, signal[:, ch], kind='linear', fill_value="extrapolate")
            resampled[:, ch] = f(t_new)
        return resampled

    @staticmethod
    def align_modalities(signals: list, fs_list: list, target_fs: float) -> list:
        """
        Align multiple modalities to the same sampling frequency.
        """
        aligned = []
        for sig, fs in zip(signals, fs_list):
            aligned_sig = Synchronization.resample_signal(sig, fs, target_fs)
            aligned.append(aligned_sig)
        # Ensure same number of samples
        min_samples = min(sig.shape[0] for sig in aligned)
        aligned = [sig[:min_samples] for sig in aligned]
        return aligned
