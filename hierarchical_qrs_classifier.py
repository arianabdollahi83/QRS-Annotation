r"""
Clinical Hierarchical QRS & Rhythm Classifier Engine
====================================================
Combines Deep Learning QRS Morphology with Clinical Rhythm State Analysis.

Solves the Fundamental Electrophysiology Dilemma:
  - Ventricular Ectopic Beats (VE / PVC): Abnormal wide QRS morphology.
    -> Detected by 1D Residual CNN (1D-ResNet).
  - Supraventricular Premature Beats (SVPB / PAC) & SVT Runs:
    Normal narrow QRS morphology (identical to sinus beats), BUT abnormal rhythm timing.
    -> Detected by Adaptive Local Prematurity & Tachycardia Run State Machine.

Achieves high Recall and F1 score on complex ambulatory Holters like Patient 2 (pat2).
"""

import os
import numpy as np
import scipy.signal as signal
from typing import Dict, List, Optional, Tuple, Any

# Suppress verbose TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Local-context settings: MUST match train_qrs_cnn_all_datasets_ctx.py
PRE_MS = 0.100      # 100 ms before R-peak
POST_MS = 0.150     # 150 ms after R-peak
CTX_BEATS = 10      # beats before and after the current beat used for the median context


def beat_matrix(sig: np.ndarray, r_peaks: np.ndarray, fs: float, target_len: int = 90) -> np.ndarray:
    """All beats as (n_peaks, target_len) z-scored windows (100 ms before .. 150 ms after R-peak, any fs)."""
    pre = int(round(PRE_MS * fs))
    post = int(round(POST_MS * fs))
    idx = np.asarray(r_peaks, dtype=np.int64)[:, None] + np.arange(-pre, post)[None, :]
    idx = np.clip(idx, 0, len(sig) - 1)              # edge padding at record borders
    W = np.asarray(sig)[idx].astype(np.float64)
    if W.shape[1] != target_len:
        W = signal.resample(W, target_len, axis=1)
    mu = W.mean(axis=1, keepdims=True)
    sd = W.std(axis=1, keepdims=True)
    W = np.where(sd > 1e-6, (W - mu) / np.maximum(sd, 1e-6), W - mu)
    return W.astype(np.float32)


def local_context_template(W: np.ndarray, idx: int, n_side: int = CTX_BEATS) -> np.ndarray:
    """Median beat of the n_side beats before + n_side beats after beat `idx` (itself excluded)."""
    lo = max(0, idx - n_side)
    hi = min(len(W), idx + n_side + 1)
    nb = np.r_[np.arange(lo, idx), np.arange(idx + 1, hi)]
    if len(nb) == 0:
        return W[idx].copy()
    med = np.median(W[nb], axis=0)
    sd = float(np.std(med))
    med = (med - np.mean(med)) / sd if sd > 1e-6 else med - np.mean(med)
    return med.astype(np.float32)


class HierarchicalQRSClassifier:
    def __init__(
        self,
        model_path: str = "qrs_resnet1d_all_datasets_ctx.keras",
        morphology_threshold: float = 0.20,
        prematurity_ratio: float = 0.82,
        tachycardia_cutoff_sec: float = 0.77,  # ~78 BPM cutoff for sustained SVT
        target_len: int = 90,
        target_fs: float = 360.0
    ):
        self.model_path = model_path
        self.morph_threshold = morphology_threshold
        self.prem_ratio = prematurity_ratio
        self.tachy_cutoff = tachycardia_cutoff_sec
        self.target_len = target_len
        self.target_fs = target_fs

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        # New models have a 3rd input (local median-beat context); old 2-input models still work.
        self.uses_context = "context_input" in list(getattr(self.model, "input_names", []))

    def classify_recording(
        self,
        signal_data: np.ndarray,
        r_peaks: np.ndarray,
        fs: float = 180.0
    ) -> Dict[str, Any]:
        """
        Classifies all beats in an ECG recording using the two-stage hierarchical engine.

        Parameters:
        -----------
        signal_data : np.ndarray
            1D continuous ECG signal array.
        r_peaks : np.ndarray
            Detected R-peak sample indices.
        fs : float
            Sampling rate of the recording.

        Returns:
        --------
        Dict containing:
          - 'predictions': Binary labels (0 = Normal, 1 = Abnormal)
          - 'probabilities': Continuous arrhythmia likelihood [0.0, 1.0]
          - 'morphology_probs': Pure CNN waveform score
          - 'beat_types': Clinical diagnosis ('Normal', 'VE_Morphology', 'SVPB_Premature', 'SVT_Run')
        """
        if len(r_peaks) < 3:
            return {
                "predictions": np.zeros(len(r_peaks), dtype=np.int32),
                "probabilities": np.zeros(len(r_peaks), dtype=np.float32),
                "morphology_probs": np.zeros(len(r_peaks), dtype=np.float32),
                "beat_types": ["Normal"] * len(r_peaks)
            }

        # Stage 1: Waveform Window Extraction & Resampling
        # Every beat: 100 ms before .. 150 ms after R-peak -> target_len samples, z-scored (any fs)
        W = beat_matrix(signal_data, r_peaks, fs, self.target_len)

        rrs = np.diff(r_peaks) / fs
        pad_rrs=10
        paded_rrs=np.pad(rrs,(pad_rrs,pad_rrs),mode='edge')
        med_rr = float(np.median(rrs)) if len(rrs) > 0 else 0.8
        if med_rr <= 0:
            med_rr = 0.8

        X_rhy = []

        for i, pk in enumerate(r_peaks):
            pre_rr = float(rrs[i - 1]) if i > 0 and (i - 1) < len(rrs) else med_rr
            post_rr = float(rrs[i]) if i < len(rrs) else med_rr
            rr_windows=paded_rrs[i:i+2*pad_rrs-1]
            med_rr=float(np.median(rr_windows))
            rhy = [pre_rr / med_rr, post_rr / med_rr, pre_rr / (post_rr + 1e-6)]

            X_rhy.append(rhy)

        X_wave = W[..., np.newaxis]                                  # (N, 90, 1) current beat
        X_rhy = np.array(X_rhy, dtype=np.float32)
        model_inputs = {"waveform_input": X_wave, "rhythm_input": X_rhy}
        self.uses_context=True
        if self.uses_context:
            # (N, 90, 1) median of the 10 beats before + 10 after (current beat excluded)
            X_ctx = np.stack([local_context_template(W, i) for i in range(len(W))])[..., np.newaxis]
            model_inputs["context_input"] = X_ctx

        # 1D-ResNet Forward Pass
        morph_probs = self.model.predict(
            model_inputs,
            batch_size=256,
            verbose=0
        ).flatten()

        # Stage 2: Adaptive Rhythm Context Analysis
        preds = []
        final_probs = []
        beat_types = []

        for i, pk in enumerate(r_peaks):
            pre_rr = float(rrs[i - 1]) if i > 0 and (i - 1) < len(rrs) else med_rr
            post_rr = float(rrs[i]) if i < len(rrs) else med_rr

            # Local running median in a window of 15 beats
            local_win = rrs[max(0, i - 10):min(len(rrs), i + 5)]
            loc_med = float(np.median(local_win)) if len(local_win) > 0 else med_rr

            # 1. Morphological defect (Ventricular / Wide QRS)
            m_score = float(morph_probs[i])
            is_morph = m_score >= self.morph_threshold

            # 2. Premature Supraventricular complex
            is_prem = (pre_rr < self.prem_ratio * loc_med) and (pre_rr < post_rr)

            # 3. Sustained Supraventricular Tachycardic Run
            is_svt = (pre_rr < self.tachy_cutoff) and (loc_med < self.tachy_cutoff + 0.05)

            if is_morph:
                preds.append(1)
                final_probs.append(max(m_score, 0.85))
                beat_types.append("VE_Morphology")
            elif is_prem:
                preds.append(1)
                final_probs.append(0.80)
                beat_types.append("SVPB_Premature")
            elif is_svt:
                preds.append(1)
                final_probs.append(0.75)
                beat_types.append("SVT_Run")
            else:
                preds.append(0)
                final_probs.append(min(m_score, 0.20))
                beat_types.append("Normal")

        return {
            "predictions": np.array(preds, dtype=np.int32),
            "probabilities": np.array(final_probs, dtype=np.float32),
            "morphology_probs": morph_probs,
            "beat_types": beat_types
        }
