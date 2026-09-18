"""
EEG-LSTM-CNN-Learner-Classification
Streamlit Web Application for Visual Learner Identification from Raw EEG.

Research Reference:
Jawed, Faye & Malik — IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024.
"""

import os
import io
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import (
    preprocess_eeg,
    EEGPreprocessingError,
    N_CHANNELS,
    SAMPLE_RATE,
    SEGMENT_SEC,
    N_SAMPLES,
    STANDARD_64_CHANNELS
)
from src.inference import EEGInferenceEngine, EEGInferenceError
from src.visualizations import (
    plot_eeg_waveforms,
    plot_power_spectral_density,
    plot_prediction_gauge,
    plot_segment_timeline,
    REPRESENTATIVE_CHANNELS
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EEG Visual Learner Classification | LSTM-CNN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-prediction {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #F3F4F6;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Inference Engine Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-trained LSTM-CNN model...")
def load_engine() -> EEGInferenceEngine:
    model_path = os.path.join(os.path.dirname(__file__), "model", "lstm_cnn_model.keras")
    config_path = os.path.join(os.path.dirname(__file__), "model", "model_config.json")
    return EEGInferenceEngine(model_path=model_path, config_path=config_path)


# -----------------------------------------------------------------------------
# Main Application Header & Disclaimer
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 EEG Visual Learner Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Deep Learning-Based Assessment Model using Raw 64-Channel EEG (LSTM-CNN Architecture)</div>',
    unsafe_allow_html=True
)

# Mandatory Academic & Medical Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Research & Educational Disclaimer:</strong><br>
    This web platform is designed strictly for academic, scientific research, and educational demonstration purposes. 
    It is <strong>NOT a medical diagnostic system</strong>, clinical device, or psychological evaluation tool. 
    Predictions are based on algorithmic pattern analysis of resting-state electroencephalographic rhythms.
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Configuration & Input Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-LSTM--CNN-blue?style=flat-square&logo=tensorflow", use_container_width=False)
    st.image("https://img.shields.io/badge/Dataset-PhysioNet_EEGMMIDB-green?style=flat-square", use_container_width=False)
    st.image("https://img.shields.io/badge/Sampling_Rate-160_Hz-orange?style=flat-square", use_container_width=False)

    st.markdown("---")
    st.subheader("📁 Select EEG Input")

    input_mode = st.radio(
        "Choose Data Source:",
        ["Instant Sample Demo", "Upload Custom EEG File"],
        index=0,
        help="Use built-in benchmark recordings or upload your own 64-channel EEG data."
    )

    selected_sample = None
    uploaded_file = None

    if input_mode == "Instant Sample Demo":
        sample_choice = st.selectbox(
            "Select Benchmark Sample:",
            [
                "Subject 001 — Eyes-Open (Baseline 01 / Non-Visual Learner)",
                "Subject 001 — Eyes-Closed (Baseline 02 / Visual Learner)"
            ]
        )
        if "Eyes-Open" in sample_choice:
            sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample_eyes_open_non_visual.npy")
            expected_class = "Non-Visual Learner (Class 0)"
        else:
            sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample_eyes_closed_visual.npy")
            expected_class = "Visual Learner (Class 1)"
    else:
        uploaded_file = st.file_uploader(
            "Upload EEG Recording (.edf, .npy, .npz, .csv):",
            type=["edf", "npy", "npz", "csv"],
            help="Upload a standard 64-channel EEG recording sampled at or resampleable to 160 Hz."
        )

    st.markdown("---")
    st.subheader("⚙️ Pipeline Specifications")
    st.markdown("""
    - **Channels**: 64 (10-10 Montage)
    - **Sampling Rate**: 160 Hz
    - **Window Length**: 4.0 sec (640 pts)
    - **Bandpass Filter**: 1.0 – 45.0 Hz
    - **Notch Filter**: 60.0 Hz
    - **Artifact Clip**: ±500 µV
    - **Normalization**: Z-score Standardization
    """)


# -----------------------------------------------------------------------------
# Pipeline Execution & Processing
# -----------------------------------------------------------------------------
engine = load_engine()

data_to_process = None
file_format = None

if input_mode == "Instant Sample Demo":
    if os.path.exists(sample_path):
        data_to_process = np.load(sample_path)
        file_format = "npy"
        st.info(f"Loaded benchmark sample: **{sample_choice}** (Expected Ground Truth: `{expected_class}`)")
    else:
        st.error(f"Sample file not found at '{sample_path}'. Please upload an EEG file instead.")
elif uploaded_file is not None:
    data_to_process = io.BytesIO(uploaded_file.getvalue())
    file_format = uploaded_file.name.split(".")[-1].lower()
    st.success(f"File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")


if data_to_process is not None:
    with st.spinner("Executing EEG preprocessing pipeline (filtering, resampling, segmentation, z-scoring)..."):
        try:
            segments, meta = preprocess_eeg(data_to_process, file_type=file_format)
            st.session_state["preprocessed_segments"] = segments
            st.session_state["meta"] = meta
        except EEGPreprocessingError as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error during preprocessing: {e}")
            st.stop()

    with st.spinner("Running LSTM-CNN neural inference..."):
        try:
            results = engine.predict(segments)
            st.session_state["results"] = results
        except EEGInferenceError as e:
            st.error(f"Inference Error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    # -------------------------------------------------------------------------
    # Display Results & Tabs
    # -------------------------------------------------------------------------
    results = st.session_state["results"]
    segments = st.session_state["preprocessed_segments"]
    meta = st.session_state["meta"]

    # Top Metric Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Predicted Class", value=f"Class {results['predicted_class_id']}")
    with col2:
        st.metric(label="Confidence Level", value=f"{results['confidence_percent']}%")
    with col3:
        st.metric(label="Total Windows Processed", value=f"{results['num_segments']} segment(s)")
    with col4:
        st.metric(label="Segment Agreement", value=f"{results['segment_consensus']}%")

    # High-impact Prediction Banner
    pred_label = results["predicted_label"]
    pred_class = results["predicted_class_id"]
    color_banner = "#10B981" if pred_class == 1 else "#3B82F6"

    st.markdown(f"""
    <div style="background-color: {color_banner}; color: white; padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h2 style="margin: 0; color: white;">🎯 Outcome: {pred_label}</h2>
        <p style="margin: 8px 0 0 0; font-size: 1.05rem;">
            Confidence: <strong>{results['confidence_percent']}%</strong> &bull; 
            {results['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Multi-tab Dashboard View
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Classification Analysis",
        "🌊 EEG Signal Traces",
        "⚡ Spectral & Frequency Analysis",
        "🔬 Methodology & Architecture"
    ])

    # Tab 1: Classification Analysis
    with tab1:
        st.subheader("Model Probability Breakdown")
        gauge_fig = plot_prediction_gauge(results["prob_class_0"], results["prob_class_1"])
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close(gauge_fig)

        if results["num_segments"] > 1:
            st.markdown("---")
            st.subheader("Temporal Segment Evolution")
            st.markdown("Softmax probabilities computed for each 4-second window across the recording:")
            timeline_fig = plot_segment_timeline(results["segment_probabilities"])
            if timeline_fig is not None:
                st.pyplot(timeline_fig, use_container_width=True)
                plt.close(timeline_fig)

    # Tab 2: EEG Signal Traces
    with tab2:
        st.subheader("Multichannel EEG Waveforms (First 4-Second Window)")
        st.markdown("Displaying representative Frontal (Fp1, Fz), Central (Cz), Parietal (Pz), and Occipital (O1, Oz, O2) channels:")
        wave_fig = plot_eeg_waveforms(segments[0], sfreq=SAMPLE_RATE)
        st.pyplot(wave_fig, use_container_width=True)
        plt.close(wave_fig)

    # Tab 3: Spectral & Frequency Analysis
    with tab3:
        st.subheader("Power Spectral Density (PSD) & Alpha Oscillations")
        st.markdown("""
        Visual learner classification strongly correlates with occipital **Alpha band synchronization (8–13 Hz)**.
        - **Visual Learners (Eyes-Closed)** exhibit high-amplitude, synchronized alpha peaks.
        - **Non-Visual Learners (Eyes-Open)** exhibit desynchronized alpha attenuation.
        """)
        psd_fig = plot_power_spectral_density(segments[0], sfreq=SAMPLE_RATE, channel_idx=61, channel_name="Oz (Occipital Midline)")
        st.pyplot(psd_fig, use_container_width=True)
        plt.close(psd_fig)

    # Tab 4: Methodology & Architecture
    with tab4:
        st.subheader("Research Foundation & Deep Learning Architecture")
        st.markdown("""
        ### Paper Reference
        - **Title:** *Deep Learning-Based Assessment Model for Visual Learner Identification using Raw EEG*
        - **Authors:** Jawed, Faye & Malik
        - **Publication:** *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*, Vol. 32, 2024.
        
        ### Neural Network Architecture (LSTM-CNN)
        The hybrid architecture is structured to extract both long-term temporal dependencies and spatial/local convolutional feature representations:
        1. **Input Layer**: `(Batch, 640 timesteps, 64 EEG channels)`
        2. **LSTM Layer**: 64 units (`return_sequences=True`) to model temporal signal dynamics
        3. **Dropout**: 30% rate for regularization
        4. **Conv1D Block 1**: 64 filters, kernel size 3, ReLU activation, same padding
        5. **MaxPooling1D**: Pool size 2, stride 2 (temporal downsampling)
        6. **Conv1D Block 2**: 32 filters, kernel size 3, ReLU activation, same padding
        7. **GlobalAveragePooling1D**: Sequence aggregation to invariant feature vectors
        8. **Dense Layer**: 64 hidden units with ReLU activation
        9. **Dropout**: 30% rate
        10. **Softmax Output**: 2 classes (Non-Visual vs Visual Learner)

        ### PhysioNet EEGMMIDB Benchmark Results
        | Metric | Baseline LSTM-CNN | CNN-BiLSTM-Attention |
        | :--- | :---: | :---: |
        | **Validation Accuracy** | **87.12%** | 92.42% |
        | **Validation Balanced Acc** | **87.12%** | 92.42% |
        | **Validation ROC-AUC** | **0.9738** | 0.9490 |
        | **Test Accuracy** | **80.45%** | 85.91% |
        | **3-Fold CV Mean Acc** | **82.9 ± 4.0%** | 87.9 ± 3.9% |
        """)
else:
    # Empty State Instructions
    st.info("👈 Select a demo sample or upload a 64-channel EEG file in the sidebar to begin analysis.")
    st.markdown("""
    ### Getting Started
    1. **Test with 1-Click Samples**: Use the sidebar radio button to toggle between sample recordings (Eyes-Open vs Eyes-Closed).
    2. **Upload Your Data**: Upload a `.edf` (European Data Format), `.npy`, `.npz`, or `.csv` EEG recording with 64 channels.
    3. **Automated Analysis**: The app automatically applies 1–45 Hz bandpass filtering, 60 Hz notch filtering, artifact thresholding, and window segmentation before evaluating with the pre-trained neural network.
    """)
