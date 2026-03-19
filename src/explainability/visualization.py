import matplotlib.pyplot as plt
import numpy as np

class ExplainVisualizer:
    """
    Visualization tools for LRP relevance and attention.
    """

    @staticmethod
    def plot_signal_with_relevance(signal: np.ndarray, relevance: np.ndarray,
                                   channel: int = 0, figsize=(12, 4)):
        """
        Overlay signal with relevance scores.
        signal: (C, T)
        relevance: (C, T)
        """
        plt.figure(figsize=figsize)
        plt.plot(signal[channel], label="Signal", color='blue', alpha=0.7)
        plt.plot(relevance[channel], label="Relevance", color='red', alpha=0.6)
        plt.fill_between(np.arange(signal.shape[1]), 0, relevance[channel],
                         color='red', alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel("Amplitude / Relevance")
        plt.title(f"Channel {channel} - Signal + Relevance Overlay")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_attention_weights(attn_weights: np.ndarray, figsize=(10, 4)):
        """
        Plot attention weights across sequence.
        attn_weights: (batch, seq_len, seq_len) or (seq_len, seq_len)
        """
        if attn_weights.ndim == 3:
            attn_weights = attn_weights[0]  # first sample
        plt.figure(figsize=figsize)
        plt.imshow(attn_weights, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.title("Attention Weight Matrix")
        plt.show()

    @staticmethod
    def overlay_signal_attention(signal: np.ndarray, attn_weights: np.ndarray, channel: int = 0):
        """
        Overlay signal amplitude with attention summed over keys.
        """
        attn_sum = attn_weights.sum(axis=0)  # sum across queries
        attn_norm = attn_sum / attn_sum.max()
        plt.figure(figsize=(12,4))
        plt.plot(signal[channel], label='Signal', color='blue')
        plt.plot(attn_norm * signal[channel].max(), label='Attention overlay', color='orange', alpha=0.7)
        plt.fill_between(np.arange(len(signal[channel])), 0, attn_norm*signal[channel].max(),
                         color='orange', alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel('Amplitude / Attention')
        plt.title(f"Channel {channel} - Signal + Attention Overlay")
        plt.legend()
        plt.show()
