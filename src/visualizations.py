"""
Visualization Module for EEG Signals, Spectral Analysis, and Model Predictions.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Representative channels across scalp regions
REPRESENTATIVE_CHANNELS = [
    (21, 'Fp1 (Frontal Polar)'),
    (31, 'Fz (Frontal Midline)'),
    (10, 'Cz (Central Vertex)'),
    (50, 'Pz (Parietal Midline)'),
    (60, 'O1 (Left Occipital)'),
    (61, 'Oz (Occipital Midline)'),
    (62, 'O2 (Right Occipital)')
]


def plot_eeg_waveforms(
    segment: np.ndarray,
    sfreq: float = 160.0,
    channels: Optional[List[int]] = None,
    channel_labels: Optional[List[str]] = None,
    title: str = "Multichannel EEG Traces (4-Second Window)"
) -> plt.Figure:
    """
    Plot stacked EEG signal traces for selected channels.

    Args:
        segment: shape (640, 64) or (time, channels)
        sfreq: sampling frequency (Hz)
    """
    if segment.ndim == 3:
        segment = segment[0]

    time = np.arange(segment.shape[0]) / sfreq

    if channels is None:
        channels = [idx for idx, _ in REPRESENTATIVE_CHANNELS]
        channel_labels = [lbl for _, lbl in REPRESENTATIVE_CHANNELS]
    elif channel_labels is None:
        channel_labels = [f"Ch {c+1}" for c in channels]

    n_plots = len(channels)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 1.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_plots))

    for i, (ch_idx, label) in enumerate(zip(channels, channel_labels)):
        ch_data = segment[:, ch_idx]
        axes[i].plot(time, ch_data, color=colors[i], linewidth=1.2)
        axes[i].set_ylabel(label, fontsize=9, fontweight='medium', rotation=0, ha='right', va='center')
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].set_xlim(time[0], time[-1])
        axes[i].tick_params(axis='both', which='major', labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout()
    return fig


def plot_power_spectral_density(
    segment: np.ndarray,
    sfreq: float = 160.0,
    channel_idx: int = 61,  # Default Oz (Midline Occipital, crucial for Alpha)
    channel_name: str = "Oz (Occipital Midline)"
) -> plt.Figure:
    """
    Compute and plot Power Spectral Density (PSD) via Welch's method.
    Highlights Delta, Theta, Alpha, Beta, and Gamma frequency bands.
    """
    if segment.ndim == 3:
        segment = segment[0]

    ch_data = segment[:, channel_idx]

    # Welch PSD computation
    freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=min(len(ch_data), 256))

    # Mask up to 45 Hz
    mask = freqs <= 45.0
    freqs = freqs[mask]
    psd = psd[mask]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, psd, color='#1f77b4', linewidth=2, label=f'PSD ({channel_name})')

    # Frequency band shading
    bands = [
        ('Delta (1-4 Hz)', 1, 4, '#e1f5fe'),
        ('Theta (4-8 Hz)', 4, 8, '#e8f5e9'),
        ('Alpha (8-13 Hz)*', 8, 13, '#fff9c4'),  # Highlight Alpha
        ('Beta (13-30 Hz)', 13, 30, '#fff3e0'),
        ('Gamma (30-45 Hz)', 30, 45, '#f3e5f5')
    ]

    for band_name, low, high, color in bands:
        ax.axvspan(low, high, color=color, alpha=0.55, label=band_name)

    ax.set_title(f"Power Spectral Density (PSD) — Frequency Band Decomposition\nChannel: {channel_name}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Power Spectral Density (V²/Hz)", fontsize=10, fontweight='bold')
    ax.set_xlim(0, 45)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_prediction_gauge(prob_class_0: float, prob_class_1: float) -> plt.Figure:
    """
    Plot modern horizontal probability breakdown bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 2.2))

    categories = ['Non-Visual Learner (Class 0)', 'Visual Learner (Class 1)']
    probs = [prob_class_0, prob_class_1]
    colors = ['#3498db', '#2ecc71'] if prob_class_1 > prob_class_0 else ['#2ecc71', '#3498db']

    bars = ax.barh(categories, probs, color=colors, height=0.5, edgecolor='black', linewidth=0.5)

    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.2f}%",
                ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Softmax Probability", fontsize=10, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def plot_segment_timeline(segment_probs: List[List[float]]) -> Optional[plt.Figure]:
    """
    Plot segment-by-segment probability timeline.
    """
    if len(segment_probs) <= 1:
        return None

    probs_0 = [p[0] for p in segment_probs]
    probs_1 = [p[1] for p in segment_probs]
    indices = list(range(1, len(segment_probs) + 1))

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(indices, probs_0, marker='o', color='#3498db', linewidth=1.8, label='Non-Visual (Class 0)')
    ax.plot(indices, probs_1, marker='s', color='#2ecc71', linewidth=1.8, label='Visual Learner (Class 1)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary (0.5)')

    ax.set_title("Segment-by-Segment Prediction Evolution", fontsize=11, fontweight='bold')
    ax.set_xlabel("Window Index (4s per segment)", fontsize=9, fontweight='bold')
    ax.set_ylabel("Softmax Probability", fontsize=9, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(indices)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig
