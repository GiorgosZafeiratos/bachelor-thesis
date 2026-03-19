import numpy as np
from typing import List, Tuple

class Segmentation:
    def __init__(self, window_size: int = 512, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap

    def segment(self, signal: np.ndarray) -> np.ndarray:
        """
        Sliding window segmentation.
        :param signal: (samples, channels)
        :return: segments (num_segments, channels, window_size)
        """
        step = int(self.window_size * (1 - self.overlap))
        segments = []
        for start in range(0, signal.shape[0] - self.window_size + 1, step):
            segment = signal[start:start + self.window_size, :]
            segments.append(segment.T)  # (channels, window_size)
        return np.stack(segments, axis=0)

    def multi_channel_segment(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        Segment multiple signals simultaneously and stack them.
        """
        segmented_list = [self.segment(sig) for sig in signals]
        # Ensure all signals have the same number of segments
        min_len = min(seg.shape[0] for seg in segmented_list)
        segmented_list = [seg[:min_len] for seg in segmented_list]
        return np.stack(segmented_list, axis=1)  # (num_segments, modalities, channels, window_size)
