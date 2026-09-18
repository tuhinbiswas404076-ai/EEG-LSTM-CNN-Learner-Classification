"""
EEG Preprocessing Module for LSTM-CNN Classification.

Preserves the exact research preprocessing pipeline from the training notebook:
  1. IIR Butterworth 4th order bandpass filtering (1.0 - 45.0 Hz)
  2. Notch filtering at 60.0 Hz (power-line noise suppression)
  3. Frequency resampling to 160 Hz
  4. 64-channel verification and extraction
  5. Extreme artifact threshold clipping (±500 µV / 500e-6 V)
  6. Window segmentation (4 seconds @ 160 Hz = 640 samples)
  7. Robust Z-score normalization: (X - mu) / (sigma + 1e-8)
"""

import io
import os
import tempfile
from typing import Tuple, List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from scipy import signal

# Optional MNE import with fallback
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# Constants matching training notebook exactly
N_CHANNELS = 64
SAMPLE_RATE = 160
SEGMENT_SEC = 4
N_SAMPLES = SEGMENT_SEC * SAMPLE_RATE  # 640 samples
CLIP_THRESHOLD = 500e-6                # 500 µV (in Volts)
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 60.0

# Standard 64-channel names used in EEGMMIDB (10-10 system)
STANDARD_64_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..',
    'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.',
    'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..',
    'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.',
    'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
    'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.',
    'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..'
]


class EEGPreprocessingError(Exception):
    """Custom exception raised when EEG preprocessing fails."""
    pass


def apply_scipy_filters(
    data: np.ndarray,
    sfreq: float = SAMPLE_RATE,
    low_freq: float = BANDPASS_LOW,
    high_freq: float = BANDPASS_HIGH,
    notch_freq: float = NOTCH_FREQ
) -> np.ndarray:
    """
    Apply bandpass (1-45 Hz) and notch (60 Hz) filters using SciPy Butterworth filter.
    Used as an ultra-fast fallback or direct array filter.
    
    Args:
        data: shape (time, channels) or (channels, time)
    Returns:
        filtered array of same shape
    """
    # Design 4th order Butterworth bandpass
    nyq = 0.5 * sfreq
    low = low_freq / nyq
    high = min(high_freq / nyq, 0.99)
    b_band, a_band = signal.butter(4, [low, high], btype='bandpass')

    # Design 60Hz notch filter (Q=30)
    w0 = notch_freq / nyq
    if w0 < 1.0:
        b_notch, a_notch = signal.iirnotch(w0, 30.0)
    else:
        b_notch, a_notch = None, None

    # Apply along time axis (axis 0 if data is (time, channels))
    axis = 0 if data.shape[0] > data.shape[1] else 1
    filtered = signal.filtfilt(b_band, a_band, data, axis=axis)
    if b_notch is not None and a_notch is not None:
        filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=axis)
    return filtered


def preprocess_raw_mne(
    raw: "mne.io.BaseRaw",
    segment_sec: int = SEGMENT_SEC,
    n_channels: int = N_CHANNELS,
    sample_rate: int = SAMPLE_RATE,
    overlap: float = 0.0
) -> np.ndarray:
    """
    Exact reproduction of the notebook's _preprocess_and_segment function using MNE.
    
    Pipeline:
      1. Bandpass 1.0-45.0 Hz (4th order Butterworth IIR)
      2. Notch filter at 60.0 Hz
      3. Resampling to target rate (160 Hz) if sfreq differs
      4. Channel validation (at least 64 channels)
      5. Clipping extreme values to [-500uV, +500uV]
      6. Slicing into segments of length (segment_sec * sample_rate)
    """
    raw = raw.copy()

    # 1. Bandpass filter 1-45 Hz
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method='iir',
        iir_params=dict(order=4, ftype='butter'),
        verbose=False
    )

    # 2. Notch filter at 60 Hz
    raw.notch_filter(
        freqs=NOTCH_FREQ,
        method='iir',
        verbose=False
    )

    # 3. Resample if needed
    if abs(raw.info['sfreq'] - sample_rate) > 0.01:
        raw.resample(sample_rate, verbose=False)

    # 4. Extract data: (channels, time) -> (time, channels)
    data = raw.get_data()
    if data.shape[0] < n_channels:
        raise EEGPreprocessingError(
            f"Expected at least {n_channels} channels, but the input file only contains {data.shape[0]} channels."
        )

    data = data[:n_channels].T  # Shape: (time, 64)

    # 5. Clip extreme values (artifact rejection, 500 uV = 500e-6 V)
    data = np.clip(data, -CLIP_THRESHOLD, CLIP_THRESHOLD)

    # 6. Segment with overlap
    n_samples = int(segment_sec * sample_rate)
    step = max(1, int(n_samples * (1 - overlap)))

    segments = []
    start = 0
    while start + n_samples <= data.shape[0]:
        segments.append(data[start:start + n_samples])
        start += step

    if not segments:
        raise EEGPreprocessingError(
            f"The EEG recording is too short for a {segment_sec}-second segment. "
            f"Total available samples: {data.shape[0]} (need at least {n_samples})."
        )

    return np.stack(segments).astype(np.float32)


def segment_eeg(
    data: np.ndarray,
    segment_sec: int = SEGMENT_SEC,
    sample_rate: int = SAMPLE_RATE,
    overlap: float = 0.0
) -> np.ndarray:
    """
    Slice continuous (time, channels) EEG array into fixed-length segments.
    """
    n_samples = int(segment_sec * sample_rate)
    if data.shape[0] < n_samples:
        raise EEGPreprocessingError(
            f"Data length ({data.shape[0]} samples) is shorter than required segment length ({n_samples} samples)."
        )

    step = max(1, int(n_samples * (1 - overlap)))
    segments = []
    start = 0
    while start + n_samples <= data.shape[0]:
        segments.append(data[start:start + n_samples])
        start += step

    return np.stack(segments).astype(np.float32)


def normalize_eeg(
    segments: np.ndarray,
    training_mu: Optional[np.ndarray] = None,
    training_sigma: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply Z-score standardization matching training:
      X_norm = (X - mu) / (sigma + 1e-8)
      
    If training_mu/sigma are provided, uses those statistics.
    Otherwise, applies robust per-sample z-score standardization.
    """
    if segments.ndim == 2:
        # Single segment (640, 64) -> expand to (1, 640, 64)
        segments = np.expand_dims(segments, axis=0)

    if training_mu is not None and training_sigma is not None:
        norm_data = (segments - training_mu) / (training_sigma + 1e-8)
    else:
        # Per-segment normalization across timesteps and channels
        mu = segments.mean(axis=(1, 2), keepdims=True)
        sigma = segments.std(axis=(1, 2), keepdims=True) + 1e-8
        norm_data = (segments - mu) / sigma

    return norm_data.astype(np.float32)


def preprocess_edf_file(file_obj_or_path: Union[str, io.BytesIO], overlap: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess an uploaded .edf file or file path.
    """
    if not MNE_AVAILABLE:
        raise EEGPreprocessingError("MNE-Python is required to read EDF files.")

    tmp_path = None
    try:
        if isinstance(file_obj_or_path, str):
            path = file_obj_or_path
        else:
            # Write bytes to a temporary file because MNE read_raw_edf requires a filename
            suffix = ".edf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_obj_or_path.read())
                tmp_path = tmp.name
            path = tmp_path

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        info_summary = {
            "sfreq": raw.info['sfreq'],
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names[:N_CHANNELS],
            "duration_sec": raw.n_times / raw.info['sfreq']
        }

        segments = preprocess_raw_mne(raw, overlap=overlap)
        normalized = normalize_eeg(segments)

        return normalized, info_summary

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def preprocess_array(
    arr: np.ndarray,
    sfreq: float = SAMPLE_RATE,
    overlap: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess a raw NumPy array of shape (time, 64), (64, time), or pre-segmented (N, 640, 64).
    """
    # Case 1: Pre-segmented array (N, 640, 64)
    if arr.ndim == 3 and arr.shape[1] == N_SAMPLES and arr.shape[2] == N_CHANNELS:
        normalized = normalize_eeg(arr)
        info = {
            "sfreq": sfreq,
            "n_channels": N_CHANNELS,
            "segments_count": arr.shape[0],
            "duration_sec": arr.shape[0] * SEGMENT_SEC
        }
        return normalized, info

    # Case 2: Single segment (640, 64)
    if arr.ndim == 2 and arr.shape[0] == N_SAMPLES and arr.shape[1] == N_CHANNELS:
        arr_3d = np.expand_dims(arr, axis=0)
        normalized = normalize_eeg(arr_3d)
        info = {
            "sfreq": sfreq,
            "n_channels": N_CHANNELS,
            "segments_count": 1,
            "duration_sec": SEGMENT_SEC
        }
        return normalized, info

    # Case 3: Continuous 2D array
    if arr.ndim == 2:
        # Check if transposed (64, time)
        if arr.shape[0] == N_CHANNELS and arr.shape[1] > N_CHANNELS:
            arr = arr.T
        elif arr.shape[1] < N_CHANNELS:
            raise EEGPreprocessingError(
                f"Array has only {arr.shape[1]} channels. Minimum {N_CHANNELS} channels required."
            )
        else:
            arr = arr[:, :N_CHANNELS]

        # Artifact clipping
        arr = np.clip(arr, -CLIP_THRESHOLD, CLIP_THRESHOLD)
        # Apply bandpass + notch
        arr = apply_scipy_filters(arr, sfreq=sfreq)
        # Segment
        segments = segment_eeg(arr, segment_sec=SEGMENT_SEC, sample_rate=int(sfreq), overlap=overlap)
        normalized = normalize_eeg(segments)

        info = {
            "sfreq": sfreq,
            "n_channels": N_CHANNELS,
            "segments_count": segments.shape[0],
            "duration_sec": arr.shape[0] / sfreq
        }
        return normalized, info

    raise EEGPreprocessingError(
        f"Unsupported array shape {arr.shape}. Expected (time, 64) or pre-segmented (N, 640, 64)."
    )


def preprocess_csv_file(file_obj_or_path: Union[str, io.BytesIO], overlap: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess a CSV file containing EEG time series data.
    """
    try:
        df = pd.read_csv(file_obj_or_path)
    except Exception as e:
        raise EEGPreprocessingError(f"Failed to read CSV file: {e}")

    # Drop non-numeric columns if any (like timestamp or index)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < N_CHANNELS:
        raise EEGPreprocessingError(
            f"CSV file contains {numeric_df.shape[1]} numeric columns. Expected at least {N_CHANNELS} channels."
        )

    data = numeric_df.iloc[:, :N_CHANNELS].values.astype(np.float32)
    return preprocess_array(data, overlap=overlap)


def preprocess_eeg(
    file_or_data: Any,
    file_type: str = "edf",
    overlap: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified entry point for preprocessing EEG files or arrays into model-ready tensor.
    
    Returns:
        normalized_segments: (N, 640, 64) float32 array
        metadata: dictionary with signal information
    """
    file_type = file_type.lower().strip().replace(".", "")

    if file_type == "edf":
        return preprocess_edf_file(file_or_data, overlap=overlap)
    elif file_type in ["npy", "npz"]:
        if isinstance(file_or_data, (str, io.BytesIO)):
            arr = np.load(file_or_data)
            if isinstance(arr, np.lib.npyio.NpzFile):
                # Take first array in npz
                key = list(arr.keys())[0]
                arr = arr[key]
        elif isinstance(file_or_data, np.ndarray):
            arr = file_or_data
        else:
            raise EEGPreprocessingError("Invalid input for NumPy array.")
        return preprocess_array(arr, overlap=overlap)
    elif file_type == "csv":
        return preprocess_csv_file(file_or_data, overlap=overlap)
    else:
        raise EEGPreprocessingError(f"Unsupported file format '.{file_type}'. Supported: .edf, .npy, .npz, .csv")
