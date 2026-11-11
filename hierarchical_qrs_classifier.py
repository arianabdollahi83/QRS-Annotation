import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional

from features import bessel_bandpass_filter, extract_all_features, CLASS_NAMES


class HierarchicalQRSClassifier:
    """
    Inference classifier for unseen continuous ECG signals.
    Processes full signals, retains all boundary beats via median padding,
    and returns binary and 5-class predictions.
    """

    def __init__(self, model_path: str = "ecg_hierarchical_v4.keras", threshold: float = 0.30):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold

    def predict(self, raw_signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> Dict[str, Any]:
        """
        Inputs:
            raw_signal : 1D raw ECG array
            r_peaks    : 1D array of detected R-peak sample indices
            fs         : sampling rate (Hz)

        Returns:
            Dictionary containing binary predictions, class labels, and probabilities
        """
        r_peaks = np.asarray(r_peaks, dtype=np.int64)
        n_beats = len(r_peaks)
        if n_beats == 0:
            return {"predictions": [], "probabilities": [], "classes": []}

        # 1. Zero-phase Bessel bandpass filtering (0.5 - 40 Hz)
        filt_sig = bessel_bandpass_filter(raw_signal, fs, lowcut=0.5, highcut=40.0)

        # 2. Extract features in inference mode (boundary beats are padded with record medians)
        feats = extract_all_features(filt_sig, r_peaks, fs, mode="inference")

        inputs = {
            "beat_input": feats["beat"][..., None],
            "diff_input": feats["diff"][..., None],
            "rr_raw_input": feats["rr_raw"][..., None],
            "rr_norm_input": feats["rr_norm"][..., None],
            "rr_z_input": feats["rr_z"][..., None],
            "p_wave_input": feats["p_wave"]
        }

        # 3. Run Inference
        preds = self.model.predict(inputs, batch_size=256, verbose=0)
        abn_prob = np.asarray(preds[0]).ravel()
        cls_prob = np.asarray(preds[1])

        pred_binary = (abn_prob >= self.threshold).astype(int)
        pred_class_idx = np.argmax(cls_prob, axis=-1)
        pred_class_names = [CLASS_NAMES[i] for i in pred_class_idx]

        return {
            "r_peaks": r_peaks,
            "abnormal_probability": abn_prob,
            "is_abnormal": pred_binary,
            "class_probabilities": cls_prob,
            "class_predictions": pred_class_names,
            "total_evaluated_beats": n_beats
        }
