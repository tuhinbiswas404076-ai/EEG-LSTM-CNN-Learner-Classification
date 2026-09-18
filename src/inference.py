"""
EEG Inference Module for LSTM-CNN Model.

Handles model loading, batch prediction, probability calculation,
and segment-level aggregation.
"""

import os
import json
from typing import Dict, Any, Optional, Union
import numpy as np

# Suppress verbose TF logging during inference
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    try:
        from tensorflow import keras
        KERAS_AVAILABLE = True
    except ImportError:
        KERAS_AVAILABLE = False


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "lstm_cnn_model.keras")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model_config.json")


class EEGInferenceError(Exception):
    """Exception raised during model inference."""
    pass


class EEGInferenceEngine:
    """
    Production-grade inference engine for the LSTM-CNN EEG model.
    """

    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.model_path = model_path or os.path.abspath(DEFAULT_MODEL_PATH)
        self.config_path = config_path or os.path.abspath(DEFAULT_CONFIG_PATH)
        self.model = None
        self.config = self._load_config()
        self._load_model()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration metadata."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "classes": {
                "0": {"name": "Non-Visual Learner (Eyes-Open State)", "description": "Characterized by low occipital alpha power."},
                "1": {"name": "Visual Learner (Eyes-Closed State)", "description": "Characterized by prominent synchronized alpha oscillations (8-13 Hz)."}
            },
            "signal_parameters": {
                "input_shape": [640, 64],
                "sampling_rate_hz": 160,
                "num_channels": 64
            }
        }

    def _load_model(self):
        """Load the pre-trained Keras model once."""
        if not KERAS_AVAILABLE:
            raise EEGInferenceError("TensorFlow / Keras is not available in the current environment.")

        if not os.path.exists(self.model_path):
            raise EEGInferenceError(
                f"Trained model not found at '{self.model_path}'. "
                "Ensure 'lstm_cnn_model.keras' is present in the model/ directory."
            )

        try:
            self.model = keras.models.load_model(self.model_path, compile=False)
        except Exception as e:
            raise EEGInferenceError(f"Failed to load Keras model from '{self.model_path}': {e}")

    def predict(self, segments: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on preprocessed and normalized segments.

        Args:
            segments: np.ndarray of shape (N_segments, 640, 64)

        Returns:
            Dictionary containing class probabilities, predicted label,
            confidence score, and per-segment metrics.
        """
        if self.model is None:
            raise EEGInferenceError("Model is not initialized.")

        if segments.ndim == 2:
            segments = np.expand_dims(segments, axis=0)

        if segments.ndim != 3 or segments.shape[1] != 640 or segments.shape[2] != 64:
            raise EEGInferenceError(
                f"Incompatible input shape {segments.shape}. "
                f"Expected shape (N, 640, 64) for [Segments, Timesteps, Channels]."
            )

        # Batch prediction
        raw_probs = self.model.predict(segments, verbose=0)  # Shape: (N, 2)
        n_segments = len(segments)

        # Aggregate across all segments
        mean_probs = np.mean(raw_probs, axis=0)
        pred_class = int(np.argmax(mean_probs))
        confidence = float(mean_probs[pred_class])

        class_0_meta = self.config.get("classes", {}).get("0", {})
        class_1_meta = self.config.get("classes", {}).get("1", {})

        class_0_name = class_0_meta.get("name", "Non-Visual Learner")
        class_1_name = class_1_meta.get("name", "Visual Learner")

        pred_label = class_1_name if pred_class == 1 else class_0_name
        pred_description = class_1_meta.get("description", "") if pred_class == 1 else class_0_meta.get("description", "")

        # Segment level agreement / consensus
        segment_classes = np.argmax(raw_probs, axis=1)
        agreement = float(np.mean(segment_classes == pred_class))

        return {
            "predicted_class_id": pred_class,
            "predicted_label": pred_label,
            "description": pred_description,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "probabilities": {
                "Non-Visual Learner (Class 0)": float(mean_probs[0]),
                "Visual Learner (Class 1)": float(mean_probs[1])
            },
            "prob_class_0": float(mean_probs[0]),
            "prob_class_1": float(mean_probs[1]),
            "num_segments": n_segments,
            "segment_probabilities": raw_probs.tolist(),
            "segment_predictions": segment_classes.tolist(),
            "segment_consensus": round(agreement * 100, 1),
            "input_shape": list(segments.shape)
        }


_GLOBAL_ENGINE: Optional[EEGInferenceEngine] = None


def get_inference_engine(model_path: Optional[str] = None) -> EEGInferenceEngine:
    """Singleton getter for the inference engine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = EEGInferenceEngine(model_path=model_path)
    return _GLOBAL_ENGINE


def load_model_cached(model_path: Optional[str] = None) -> EEGInferenceEngine:
    """Helper alias for cached inference engine loading."""
    return get_inference_engine(model_path=model_path)
