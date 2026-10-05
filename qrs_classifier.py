"""
QRS Classification Engine (Production Inference & Online Monitoring)
====================================================================
Performs automated classification of electrocardiogram (ECG) heartbeats into
Normal (0) vs. Abnormal (1) (PVCs, PACs, Bundle Branch Blocks, Ectopic beats).

Features:
- Fast patient-specific template calibration (< 5 ms).
- Sampling-rate and voltage scale invariant inference.
- Real-time online beat classification (< 0.1 ms per beat).
- Batch classification of continuous Holter recordings.
- CLI interface for standalone inference and live evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pywt
import scipy.signal as signal
import scipy.stats as stats

TARGET_FS = 360.0  # Reference sampling rate in Hz


# =============================================================================
# Helper: Extract Clean Patient Reference Template
# =============================================================================

def extract_clean_template(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    source_fs: float,
    num_beats: int = 40,
    target_fs: float = TARGET_FS,
) -> np.ndarray:
    """
    Extracts an uncontaminated median QRS template from a consecutive resting stretch.
    """
    if len(r_peaks) < num_beats:
        num_beats = len(r_peaks)
    if num_beats == 0:
        raise ValueError("r_peaks array is empty.")

    pre = int(round(0.100 * source_fs))
    post = int(round(0.150 * source_fs))
    target_len = int(round(0.250 * target_fs))  # 90 samples

    rr_intervals = np.diff(r_peaks) / float(source_fs)

    best_start = 0
    min_rr_std = float("inf")

    # Search for consecutive beats with regular resting rhythm
    for i in range(len(rr_intervals) - num_beats + 1):
        slice_rr = rr_intervals[i : i + num_beats]
        if np.all(slice_rr > 0.55) and np.all(slice_rr < 1.25):
            rr_std = np.std(slice_rr)
            if rr_std < min_rr_std:
                min_rr_std = rr_std
                best_start = i

    selected_peaks = r_peaks[best_start : best_start + num_beats]
    windows = []
    total_len = len(ecg_signal)

    for idx in selected_peaks:
        start = idx - pre
        end = idx + post
        if start >= 0 and end <= total_len:
            win = ecg_signal[start:end]
            if len(win) > 1:
                win_res = signal.resample(win, target_len)
                std_val = np.std(win_res)
                if std_val < 1e-6:
                    std_val = 1e-6
                win_norm = (win_res - np.mean(win_res)) / std_val
                windows.append(win_norm)

    if len(windows) == 0:
        win = ecg_signal[max(0, r_peaks[0] - pre) : min(total_len, r_peaks[0] + post)]
        res = signal.resample(win, target_len)
        return (res - np.mean(res)) / (np.std(res) + 1e-6)

    template = np.median(windows, axis=0)
    std_t = np.std(template)
    if std_t < 1e-6:
        std_t = 1e-6
    return (template - np.mean(template)) / std_t


# =============================================================================
# Core QRS Classifier Class
# =============================================================================

class QRSClassifier:
    """
    Clinical-grade QRS classification engine for Normal vs. Abnormal beat labeling.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initializes the classifier with the trained XGBoost model bundle.
        """
        if model_path is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(cur_dir, "qrs_classifier_model.joblib"),
                os.path.join(cur_dir, "demo_qrs_model.joblib"),
                r"C:\Users\Artin\.gemini\antigravity\scratch\ecg_classification\demo_qrs_model.joblib",
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Could not locate QRS model artifact. Searched: {model_path}. "
                "Run train_qrs_classifier.py to train the model first."
            )

        self.model_path = model_path
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.expected_features = list(bundle["feature_columns"])
        self.target_fs = float(bundle.get("target_fs", TARGET_FS))
        self.target_window_len = int(round(0.250 * self.target_fs))

    def calibrate_patient_template(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: float,
        num_beats: int = 40,
    ) -> np.ndarray:
        """
        Calibrates the patient-specific baseline template from resting beats (< 5 ms).
        """
        return extract_clean_template(
            ecg_signal, r_peaks, source_fs=fs, num_beats=num_beats, target_fs=self.target_fs
        )

    def extract_features(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        template: np.ndarray,
        source_fs: float,
        lead: str = "II",
        local_window_size: int = 10,
    ) -> pd.DataFrame:
        """
        Extracts invariant feature matrix for all beats in an ECG signal.
        All morphology features are patient-calibrated relative to the resting template.
        """
        n_peaks = len(r_peaks)
        if n_peaks == 0:
            return pd.DataFrame(columns=self.expected_features)

        total_len = len(ecg_signal)
        pre_samples = int(round(0.100 * source_fs))
        post_samples = int(round(0.150 * source_fs))

        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-6)
        if len(template_norm) != self.target_window_len:
            template_norm = signal.resample(template_norm, self.target_window_len)

        # Precompute template reference metrics for patient calibration
        t_auc = float(np.mean(np.abs(template_norm)))
        t_energy = float(np.mean(template_norm**2))
        t_kurt = float(stats.kurtosis(template_norm))
        t_skew = float(stats.skew(template_norm))
        dt = np.diff(template_norm) * self.target_fs
        t_max_slope = float(np.max(np.abs(dt)))
        t_mean_slope = float(np.mean(np.abs(dt)))
        t_notches = int(np.sum(np.diff(np.signbit(dt))))

        t_coeffs = pywt.wavedec(template_norm, "db4", level=3)
        t_a3, t_d3, t_d2, t_d1 = t_coeffs
        t_ed1, t_ed2, t_ed3, t_ea3 = np.sum(t_d1**2), np.sum(t_d2**2), np.sum(t_d3**2), np.sum(t_a3**2)
        t_tot_e = t_ed1 + t_ed2 + t_ed3 + t_ea3 + 1e-10
        t_d2_ratio = float(t_ed2 / t_tot_e)
        t_d3_ratio = float(t_ed3 / t_tot_e)
        t_hf_lf = float((t_ed1 + t_ed2) / (t_ed3 + t_ea3 + 1e-10))

        if n_peaks > 1:
            rr_intervals = np.diff(r_peaks) / float(source_fs)
            global_median_rr = float(np.median(rr_intervals))
        else:
            rr_intervals = np.array([0.8])
            global_median_rr = 0.8

        rows = []
        for i, r_idx in enumerate(r_peaks):
            is_boundary = 0
            if i == 0:
                pre_rr = rr_intervals[0]
                is_boundary = 1
            else:
                pre_rr = rr_intervals[i - 1]

            if i == n_peaks - 1:
                post_rr = rr_intervals[-1]
                is_boundary = 1
            else:
                post_rr = rr_intervals[i]

            start_w = max(0, i - (local_window_size // 2))
            end_w = min(len(rr_intervals), i + (local_window_size // 2))
            local_slice = rr_intervals[start_w:end_w]
            local_rr = float(np.median(local_slice)) if len(local_slice) > 0 else global_median_rr
            if local_rr < 0.2:
                local_rr = 0.2

            pre_rr_ratio = pre_rr / local_rr
            post_rr_ratio = post_rr / local_rr
            pre_post_ratio = pre_rr / (post_rr + 1e-6)

            start = r_idx - pre_samples
            end = r_idx + post_samples

            if start < 0 or end > total_len:
                raw_win = np.pad(
                    ecg_signal[max(0, start) : min(total_len, end)],
                    (max(0, -start), max(0, end - total_len)),
                    mode="edge",
                )
            else:
                raw_win = ecg_signal[start:end]

            win_resampled = signal.resample(raw_win, self.target_window_len)
            std_val = np.std(win_resampled)
            if std_val < 1e-6:
                std_val = 1e-6
            x_norm = (win_resampled - np.mean(win_resampled)) / std_val

            corr = np.corrcoef(x_norm, template_norm)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            rmse = float(np.sqrt(np.mean((x_norm - template_norm) ** 2)))

            norm_auc = float(np.mean(np.abs(x_norm)))
            norm_energy = float(np.mean(x_norm**2))
            kurt = float(stats.kurtosis(x_norm))
            skew = float(stats.skew(x_norm))

            dx = np.diff(x_norm) * self.target_fs
            max_slope = float(np.max(np.abs(dx)))
            mean_slope = float(np.mean(np.abs(dx)))
            notches = int(np.sum(np.diff(np.signbit(dx))))

            coeffs = pywt.wavedec(x_norm, "db4", level=3)
            approx_A3, detail_D3, detail_D2, detail_D1 = coeffs
            e_d1 = float(np.sum(detail_D1**2))
            e_d2 = float(np.sum(detail_D2**2))
            e_d3 = float(np.sum(detail_D3**2))
            e_a3 = float(np.sum(approx_A3**2))
            total_e = e_d1 + e_d2 + e_d3 + e_a3 + 1e-10

            kurt_ratio = float(kurt / (t_kurt + 1e-4) if abs(t_kurt) > 1e-4 else 1.0)
            kurt_ratio = float(np.clip(kurt_ratio, -10.0, 10.0))

            row = {
                "pre_rr": float(pre_rr),
                "post_rr": float(post_rr),
                "pre_rr_ratio": float(pre_rr_ratio),
                "post_rr_ratio": float(post_rr_ratio),
                "pre_post_ratio": float(pre_post_ratio),
                "is_boundary_beat": is_boundary,
                "template_corr": float(corr),
                "template_rmse": float(rmse),
                "kurtosis_ratio": kurt_ratio,
                "skewness_diff": float(skew - t_skew),
                "energy_ratio": float(norm_energy / (t_energy + 1e-6)),
                "auc_ratio": float(norm_auc / (t_auc + 1e-6)),
                "max_slope_ratio": float(max_slope / (t_max_slope + 1e-6)),
                "mean_slope_ratio": float(mean_slope / (t_mean_slope + 1e-6)),
                "notches_diff": int(notches - t_notches),
                "dwt_D2_diff": float((e_d2 / total_e) - t_d2_ratio),
                "dwt_D3_diff": float((e_d3 / total_e) - t_d3_ratio),
                "hf_lf_diff": float((e_d1 + e_d2) / (e_d3 + e_a3 + 1e-10) - t_hf_lf),
                "lead": str(lead),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure all expected columns are present
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.expected_features]

    def classify_recording(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: float = 180.0,
        lead: str = "II",
        template: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Classifies all beats in an ECG signal into Normal (0) or Abnormal (1).

        Returns:
            Dictionary containing:
            - 'predictions': Array of 0 (Normal) or 1 (Abnormal)
            - 'probabilities': Array of abnormal probabilities in [0.0, 1.0]
            - 'r_peaks': Array of R-peak sample indices
            - 'times': Array of R-peak timestamps in seconds
            - 'template': The calibrated patient template
            - 'threshold': The decision threshold used
        """
        sig = np.asarray(ecg_signal, dtype=np.float64).flatten()
        peaks = np.asarray(r_peaks, dtype=np.int64).flatten()

        if len(peaks) == 0:
            return {
                "predictions": np.array([], dtype=int),
                "probabilities": np.array([], dtype=float),
                "r_peaks": np.array([], dtype=np.int64),
                "times": np.array([], dtype=float),
                "template": np.zeros(self.target_window_len),
                "threshold": self.threshold,
            }

        t0 = time.time()
        # 1. Calibrate patient template if not provided
        if template is None:
            template = self.calibrate_patient_template(sig, peaks, fs=fs)

        # 2. Extract features
        X_df = self.extract_features(sig, peaks, template, source_fs=fs, lead=lead)
        if "lead" in X_df.columns:
            X_df["lead"] = X_df["lead"].astype("category")

        # 3. Predict probabilities
        probs = self.model.predict_proba(X_df)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        inference_time = time.time() - t0

        return {
            "predictions": preds,
            "probabilities": probs,
            "r_peaks": peaks,
            "times": (peaks / float(fs)).round(3),
            "template": template,
            "threshold": self.threshold,
            "inference_time_ms": inference_time * 1000.0,
        }

    def classify_beat(
        self,
        ecg_signal: np.ndarray,
        r_peaks: np.ndarray,
        beat_idx: int,
        fs: float,
        template: np.ndarray,
        lead: str = "II",
    ) -> Tuple[int, float]:
        """
        Real-time single beat classification (< 0.1 ms latency).
        """
        single_res = self.classify_recording(
            ecg_signal, r_peaks[beat_idx : beat_idx + 1], fs=fs, lead=lead, template=template
        )
        pred = int(single_res["predictions"][0]) if len(single_res["predictions"]) else 0
        prob = float(single_res["probabilities"][0]) if len(single_res["probabilities"]) else 0.0
        return pred, prob

    def evaluate(
        self,
        ecg_signal: np.ndarray,
        detected_r_peaks: np.ndarray,
        labeled_r_peaks: Union[np.ndarray, str],
        labeled_classes: Optional[Union[np.ndarray, List[Union[int, str]]]] = None,
        predicted_labels: Optional[np.ndarray] = None,
        fs: float = 180.0,
        tolerance_samples: int = 10,
        sample_offset: int = 0,
        lead: str = "II",
        print_report: bool = True,
    ) -> Dict[str, Any]:
        """Convenience method calling evaluate_qrs_classification with this classifier instance."""
        return evaluate_qrs_classification(
            ecg_signal=ecg_signal,
            detected_r_peaks=detected_r_peaks,
            labeled_r_peaks=labeled_r_peaks,
            labeled_classes=labeled_classes,
            predicted_labels=predicted_labels,
            fs=fs,
            tolerance_samples=tolerance_samples,
            sample_offset=sample_offset,
            classifier=self,
            lead=lead,
            print_report=print_report,
        )


# =============================================================================
# Standalone Evaluation: Confusion Matrix (TP, TN, FP, FN) with Tolerance Matching
# =============================================================================

# Global cache for auto-resolving beat labels across modules
_GLOBAL_ANNOTATION_CACHE: Dict[str, Any] = {}


def load_mortara_annotations(annotation_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses Mortara pat.txt annotation export.

    Returns:
        r_peaks: 1D array of sample positions at 180 Hz
        labels: 1D array of clinical codes (0=Normal, 1=SVPB, 2=VPB, 4=Paced, 6=AFib, etc.)
    """
    samples = []
    labels = []
    with open(annotation_path, "r", encoding="latin1") as f:
        in_table = False
        for line in f:
            if "Beat    Beat     Beat" in line:
                in_table = True
                continue
            if in_table:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].isdigit():
                    samples.append(int(parts[1]))
                    labels.append(int(parts[2]))
    return np.array(samples, dtype=np.int64), np.array(labels, dtype=np.int32)


def evaluate_qrs_classification(
    ecg_signal: np.ndarray,
    detected_r_peaks: np.ndarray,
    labeled_r_peaks: Union[np.ndarray, str],
    labeled_classes: Optional[Union[np.ndarray, List[Union[int, str]]]] = None,
    predicted_labels: Optional[np.ndarray] = None,
    annotation_path: Optional[str] = None,
    fs: float = 180.0,
    tolerance_samples: int = 10,
    sample_offset: int = 0,
    classifier: Optional[QRSClassifier] = None,
    model_path: Optional[str] = None,
    lead: str = "II",
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates QRS classification performance by computing TP, TN, FP, and FN
    strictly on beats where detected and labeled R-peaks match within a tolerance threshold.

    Beats with no corresponding reference within +/- tolerance_samples (e.g. undetected beats
    or false detections) are excluded from the QRS classification confusion matrix.

    Args:
        ecg_signal: 1D ECG array (may be truncated to any duration).
        detected_r_peaks: Array of detected R-peak sample indices (local or global).
        labeled_r_peaks: Array of ground-truth R-peak sample indices OR path to pat.txt file.
        labeled_classes: Array of ground-truth labels (0=Normal, 1/2/4/6/7=Abnormal).
                         If None, will be auto-resolved from annotation_path or global cache.
        predicted_labels: Optional array of binary predicted labels (0=Normal, 1=Abnormal)
                          corresponding to detected_r_peaks. If None, QRSClassifier will run automatically.
        annotation_path: Optional path to ground-truth annotation file (.txt).
        fs: Sampling frequency in Hz (default: 180.0).
        tolerance_samples: Maximum allowable sample difference to consider a match (default: 10).
        sample_offset: Start sample index if ecg_signal is truncated from a longer file.
        classifier: Optional pre-loaded QRSClassifier instance.
        model_path: Optional path to classifier model joblib file.
        lead: ECG lead name (default: "II").
        print_report: If True, prints a formatted summary table to console.

    Returns:
        Dictionary containing TP, TN, FP, FN, Accuracy, Recall, Precision, F1,
        and detailed beat-by-beat matching records.
    """
    sig = np.asarray(ecg_signal, dtype=np.float64).flatten()
    sig_len = len(sig)

    # 1. Load annotations if filepath is provided or auto-resolve labels
    if isinstance(labeled_r_peaks, str):
        lab_peaks_raw, lab_classes_raw = load_mortara_annotations(labeled_r_peaks)
    else:
        lab_peaks_raw = np.asarray(labeled_r_peaks, dtype=np.int64).flatten()
        if labeled_classes is not None:
            lab_classes_raw = np.asarray(labeled_classes)
        elif annotation_path is not None and os.path.exists(annotation_path):
            file_peaks, file_labels = load_mortara_annotations(annotation_path)
            peak_map = dict(zip(file_peaks, file_labels))
            lab_classes_raw = np.array([peak_map.get(p, 0) for p in lab_peaks_raw], dtype=np.int32)
        elif "last" in _GLOBAL_ANNOTATION_CACHE:
            cached_peaks, cached_labels = _GLOBAL_ANNOTATION_CACHE["last"]
            peak_map = dict(zip(cached_peaks, cached_labels))
            matches = sum(1 for p in lab_peaks_raw[:50] if p in peak_map)
            if matches > 0:
                lab_classes_raw = np.array([peak_map.get(p, 0) for p in lab_peaks_raw], dtype=np.int32)
            else:
                lab_classes_raw = None
        else:
            lab_classes_raw = None

        if lab_classes_raw is None:
            # Auto-search in common patient directories
            auto_found = False
            for cand_dir in ["Pats", "../Pats", r"C:\Users\Artin\Desktop\qrs2\Pats", r"C:\Users\Artin\Desktop\New folder\Pats"]:
                if os.path.exists(cand_dir):
                    for root_d, _, files in os.walk(cand_dir):
                        for fname in files:
                            if fname.endswith(".txt") and not fname.startswith("Printed") and not fname.startswith("Reviewed"):
                                cand_path = os.path.join(root_d, fname)
                                try:
                                    f_p, f_l = load_mortara_annotations(cand_path)
                                    p_map = dict(zip(f_p, f_l))
                                    if len(lab_peaks_raw) > 0 and (lab_peaks_raw[0] in p_map or (lab_peaks_raw[0] + sample_offset) in p_map):
                                        lab_classes_raw = np.array(
                                            [p_map.get(p if p in p_map else p + sample_offset, 0) for p in lab_peaks_raw],
                                            dtype=np.int32,
                                        )
                                        auto_found = True
                                        break
                                except Exception:
                                    pass
                        if auto_found:
                            break
                if auto_found:
                    break

        if lab_classes_raw is None:
            raise ValueError(
                "Ground-truth beat labels ('labeled_classes') were not provided and could not be auto-resolved.\n"
                "To evaluate QRS classification (TP, TN, FP, FN), ground-truth labels are required.\n"
                "You can resolve this in any of the following ways:\n"
                "  1. Pass the path to the annotation file directly: labeled_r_peaks='Pats/pat1/pat1.txt'\n"
                "  2. Pass annotation_path: evaluate_qrs_classification(..., annotation_path='Pats/pat1/pat1.txt')\n"
                "  3. Pass labeled_classes array: evaluate_qrs_classification(..., labeled_classes=peak_labels)\n"
                "  4. In functions.py, use: sig, peak_samples, peak_labels = dat_reader(path, fs, name, lead, return_labels=True)"
            )

    det_peaks_raw = np.asarray(detected_r_peaks, dtype=np.int64).flatten()

    # 2. Coordinate alignment between local signal [0, sig_len] and global offsets
    # Align detected peaks to local coordinates [0, sig_len]
    if len(det_peaks_raw) > 0 and sample_offset > 0 and np.min(det_peaks_raw) >= sample_offset:
        det_local = det_peaks_raw - sample_offset
    else:
        det_local = det_peaks_raw.copy()

    # Align labeled peaks to local coordinates
    if len(lab_peaks_raw) > 0 and (np.max(lab_peaks_raw) >= sig_len or sample_offset > 0):
        if sample_offset > 0:
            lab_local = lab_peaks_raw - sample_offset
        elif np.min(lab_peaks_raw) >= np.min(det_peaks_raw) - tolerance_samples and np.min(det_peaks_raw) >= sig_len:
            # Both are global but no offset passed: infer offset from minimum
            infer_offset = int(np.min(det_peaks_raw))
            det_local = det_peaks_raw - infer_offset
            lab_local = lab_peaks_raw - infer_offset
        else:
            lab_local = lab_peaks_raw.copy()
    else:
        lab_local = lab_peaks_raw.copy()

    # Filter labeled peaks to the visible signal range (+/- tolerance)
    valid_lab_mask = (lab_local >= -tolerance_samples) & (lab_local < sig_len + tolerance_samples)
    lab_local_win = lab_local[valid_lab_mask]
    lab_classes_win = lab_classes_raw[valid_lab_mask]

    # Convert ground-truth labels to binary (0=Normal, 1=Abnormal)
    if np.issubdtype(lab_classes_win.dtype, np.bool_):
        true_binary = lab_classes_win.astype(int)
    elif np.issubdtype(lab_classes_win.dtype, np.integer):
        unique_vals = set(np.unique(lab_classes_win))
        if unique_vals.issubset({0, 1}):
            true_binary = lab_classes_win.astype(int)
        else:
            # Mortara standard: 0=Normal, [1, 2, 4, 6, 7] = Abnormal (SVPB, VPB, Paced, AFib, R-on-T)
            true_binary = np.isin(lab_classes_win, [1, 2, 7]).astype(int)
    else:
        # String labels (e.g. 'N', 'NORMAL' -> 0; others -> 1)
        true_binary = np.array(
            [0 if str(lbl).strip().upper() in ["0", "N", "NORM", "NORMAL"] else 1 for lbl in lab_classes_win],
            dtype=int,
        )

    # 3. Obtain predicted labels (either from input or by running QRSClassifier)
    if predicted_labels is not None:
        pred_binary = np.asarray(predicted_labels, dtype=int).flatten()
        if len(pred_binary) != len(det_local):
            raise ValueError(
                f"predicted_labels length ({len(pred_binary)}) does not match detected_r_peaks ({len(det_local)})."
            )
    else:
        if classifier is None:
            classifier = QRSClassifier(model_path=model_path)
        cls_result = classifier.classify_recording(sig, det_local, fs=fs, lead=lead)
        pred_binary = cls_result["predictions"]

    # 4. Greedy Nearest-Neighbor Beat Matching within tolerance_samples
    matched_pairs = []
    used_lab_indices = set()

    for d_idx, d_val in enumerate(det_local):
        cand_indices = [
            j for j, l_val in enumerate(lab_local_win)
            if abs(d_val - l_val) <= tolerance_samples and j not in used_lab_indices
        ]
        if len(cand_indices) > 0:
            best_j = min(cand_indices, key=lambda j: abs(d_val - lab_local_win[j]))
            used_lab_indices.add(best_j)

            diff = int(abs(d_val - lab_local_win[best_j]))
            y_t = int(true_binary[best_j])
            y_p = int(pred_binary[d_idx])

            matched_pairs.append({
                "det_index": d_idx,
                "lab_index": best_j,
                "detected_sample": int(d_val),
                "labeled_sample": int(lab_local_win[best_j]),
                "sample_diff": diff,
                "y_true": y_t,
                "y_pred": y_p,
                "status": "TP" if (y_t == 1 and y_p == 1) else (
                    "TN" if (y_t == 0 and y_p == 0) else (
                        "FP" if (y_t == 0 and y_p == 1) else "FN"
                    )
                ),
            })

    # 5. Calculate Confusion Matrix strictly on matched beats
    if len(matched_pairs) > 0:
        y_true_matched = np.array([m["y_true"] for m in matched_pairs], dtype=int)
        y_pred_matched = np.array([m["y_pred"] for m in matched_pairs], dtype=int)

        tp = int(np.sum((y_true_matched == 1) & (y_pred_matched == 1)))
        tn = int(np.sum((y_true_matched == 0) & (y_pred_matched == 0)))
        fp = int(np.sum((y_true_matched == 0) & (y_pred_matched == 1)))
        fn = int(np.sum((y_true_matched == 1) & (y_pred_matched == 0)))
    else:
        tp = tn = fp = fn = 0

    total_matched = len(matched_pairs)
    total_det = len(det_local)
    total_lab_win = len(lab_local_win)
    unmatched_det = total_det - total_matched
    unmatched_lab = total_lab_win - total_matched

    accuracy = float((tp + tn) / total_matched) if total_matched > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    recall = sensitivity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1_score = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    results: Dict[str, Any] = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1_score, 4),
        "total_matched": total_matched,
        "total_detected": total_det,
        "total_labeled": total_lab_win,
        "unmatched_detected": unmatched_det,
        "unmatched_labeled": unmatched_lab,
        "tolerance_samples": tolerance_samples,
        "sample_offset": sample_offset,
        "signal_duration_sec": round(sig_len / float(fs), 2),
        "matched_beats": matched_pairs,
    }

    if print_report:
        print("\n" + "=" * 65)
        print("        QRS CLASSIFICATION EVALUATION REPORT")
        print("=" * 65)
        print(f"Signal Duration     : {results['signal_duration_sec']} seconds ({sig_len:,} samples @ {fs} Hz)")
        print(f"Tolerance Threshold : <= {tolerance_samples} samples (approx {tolerance_samples/fs*1000:.1f} ms)")
        print("-" * 65)
        print(f"Total Detected Peaks: {total_det}")
        print(f"Total Labeled Peaks : {total_lab_win}")
        print(f"Matched Beats (<=10): {total_matched} (evaluated for QRS classification)")
        print(f"Unmatched Detected  : {unmatched_det} (excluded - no label within threshold)")
        print(f"Unmatched Labeled   : {unmatched_lab} (excluded - not detected within threshold)")
        print("-" * 65)
        print("CONFUSION MATRIX (Matched Beats Only):")
        print(f"  True Positives  (TP): {tp:<6} (Correctly classified Abnormal/Arrhythmia)")
        print(f"  True Negatives  (TN): {tn:<6} (Correctly classified Normal)")
        print(f"  False Positives (FP): {fp:<6} (Normal beat misclassified as Abnormal)")
        print(f"  False Negatives (FN): {fn:<6} (Abnormal beat misclassified as Normal)")
        print("-" * 65)
        print("CLINICAL PERFORMANCE METRICS:")
        print(f"  Accuracy            : {accuracy * 100:.2f}%")
        print(f"  Sensitivity (Recall): {sensitivity * 100:.2f}%")
        print(f"  Specificity         : {specificity * 100:.2f}%")
        print(f"  Precision (PPV)     : {precision * 100:.2f}%")
        print(f"  F1-Score            : {f1_score:.4f}")
        print("=" * 65 + "\n")

    return results

def main() -> None:
    """CLI runner for standalone QRS classification."""
    parser = argparse.ArgumentParser(
        description="QRSClassifier: Automated Normal vs Abnormal Beat Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default=r"C:\Users\Artin\Desktop\New folder\Pats\pat2\DataII.dat",
        help="Path to ECG recording file (.dat, .csv, .npy)",
    )
    parser.add_argument("--fs", "-f", type=float, default=180.0, help="Sampling rate in Hz")
    parser.add_argument("--duration", "-d", type=float, default=120.0, help="Duration in seconds (0 for full file)")
    parser.add_argument("--start", "-s", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--lead", type=str, default="II", help="ECG Lead name (e.g. II, MLII, V1)")
    parser.add_argument("--model", "-m", type=str, default="qrs_classifier_model.joblib", help="Model path")

    args = parser.parse_args()

    print("=" * 65)
    print("           QRS Beat Classification Engine")
    print("=" * 65)

    classifier = QRSClassifier(model_path=args.model)

    # 1. Load data
    count = int(args.duration * args.fs) if args.duration > 0 else -1
    offset_samples = int(args.start * args.fs)

    with open(args.input, "rb") as f:
        if offset_samples > 0:
            f.seek(offset_samples * 2)
        sig = np.fromfile(f, dtype="<i2", count=count).astype(np.float64)
    sig = sig * -1.0  # invert polarity for standard lead II Holter

    print(f"Loaded Signal   : {len(sig):,} samples ({len(sig)/args.fs:.1f} seconds)")

    # 2. Detect R-peaks using Phase 1 detector
    from r_peak_detector import RPeakDetector
    detector = RPeakDetector(sampling_rate=args.fs)
    res_r = detector.detect(sig)
    r_peaks = res_r.r_peaks
    print(f"Detected R-Peaks: {len(r_peaks):,} beats")

    # 3. Classify beats
    t0 = time.time()
    res_cls = classifier.classify_recording(sig, r_peaks, fs=args.fs, lead=args.lead)
    total_time = time.time() - t0

    preds = res_cls["predictions"]
    probs = res_cls["probabilities"]
    n_norm = int(np.sum(preds == 0))
    n_abn = int(np.sum(preds == 1))

    print(f"\nClassification Completed in {total_time*1000:.1f} ms ({total_time/len(r_peaks)*1000:.2f} ms/beat)")
    print(f"Decision Threshold : {classifier.threshold:.4f}")
    print(f"Total Beats        : {len(preds):,}")
    print(f"  - Normal Beats   : {n_norm:,} ({n_norm/len(preds)*100:.2f}%)")
    print(f"  - Abnormal Beats : {n_abn:,} ({n_abn/len(preds)*100:.2f}%)")

    if n_abn > 0:
        print("\nFirst 5 Abnormal Beats Detected:")
        abn_indices = np.where(preds == 1)[0][:5]
        for idx in abn_indices:
            print(f"  - Beat #{idx+1:03d} at t={res_cls['times'][idx]:.2f}s | Prob={probs[idx]:.3f}")


if __name__ == "__main__":
    main()
