r"""
Patient QRS Assessment & Dual-Mode Visualization Module
======================================================
Provides an arbitrary-length QRS assessment and visualization suite:
  1. Computes clinical confusion metrics: TP, TN, FP, FN, Sensitivity, Specificity, F1
  2. Plots the signal in the requested interval with continuous SQI-coloring and 4 clinical markers:
     - OPTION A: Static ECG strip plot (interactive zoom/pan, metrics banner)
     - OPTION B: Real-time animated telemetry streaming (speed control, pause, scrubbing)

The 4 Clinical Markers:
  1. Our Detected R-Peaks        -> Electric Purple Circle   (o, #C084FC)
  2. Cardiologist Labeled Peaks  -> Vivid Lime Green Triangle (v, #22C55E)
  3. Labeled Abnormal QRS        -> Golden Star               (*, #FACC15)
  4. Our Detected Abnormal QRS   -> Bright Crimson Diamond    (D, #EF4444)

Continuous SQI Waveform Colors:
  - Vibrant Blue : Clean ECG (S < 0.35)
  - Coral / Pink : Mild Noise (0.35 <= S < 0.70)
  - Dark Red     : Severe Artifact (S >= 0.70)

Usage Examples:
---------------
  from assess_qrs import assess_patient_qrs, plot_patient_qrs

  # 1. Direct confusion matrix unpacking:
  tp, tn, fp, fn = assess_patient_qrs("pat2", start_sec=11750, duration_sec=120)

  # 2. Static plot with SQI and 4 annotations:
  assess_patient_qrs("pat2", start_sec=11750, duration_sec=60, plot="static")

  # 3. Animated telemetry streaming with SQI:
  assess_patient_qrs("pat2", start_sec=11750, duration_sec=60, plot="animated")

  # 4. Standalone plotting function:
  plot_patient_qrs("pat1", start_sec=160, duration_sec=45, mode="static")
  plot_patient_qrs("pat1", start_sec=160, duration_sec=45, mode="animated")

  # 5. Supply YOUR OWN R-peaks (skip built-in detector):
  my_peaks = my_r_peak_function(signal)
  res = assess_patient_qrs("pat2", start_sample=0, num_samples=21600,
                            r_peaks=my_peaks, plot="static")

  # 6. Disable SQI for faster execution (plain cyan waveform):
  res = assess_patient_qrs("pat1", start_sec=160, duration_sec=60,
                            compute_sqi=False, plot="static")

  # 7. CLI usage:
  #   python assess_qrs.py --pat pat2 --start 11750 --duration 60 --plot static
  #   python assess_qrs.py --pat pat2 --start 11750 --duration 60 --plot animated
  #   python assess_qrs.py --pat pat1 --start 160 --duration 60 --plot static --no_sqi
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure stdout/stderr handles Unicode safely on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

from functions import read_holter_labels
from r_peak_detector import RPeakDetector
from sqi_detector import SQIDetector, get_noise_colormap
from hierarchical_qrs_classifier import HierarchicalQRSClassifier

PATS_DEFAULT_DIR = r"C:\Users\Artin\Desktop\New folder\Pats"
PATS_ALT_DIR = r"C:\Users\Artin\Desktop\qrs2\Pats"
MODEL_DEFAULT_PATH = r"C:\Users\Artin\Desktop\QRS\qrs_resnet1d_all_datasets.keras"

# Patient clinical profiles and default configurations
PATIENT_REGISTRY = {
    "pat1": {
        "name": "Patient 1",
        "condition": "Persistent Atrial Fibrillation / Tachycardia (99.7% Arrhythmia)",
        "invert": False,
        "fs": 180.0,
        "default_start_sec": 160.0,
        "default_duration_sec": 60.0,
    },
    "pat2": {
        "name": "Patient 2",
        "condition": "Paroxysmal Supraventricular Tachycardia (SVT) & VE",
        "invert": True,  # Hardware-inverted Lead II
        "fs": 180.0,
        "default_start_sec": 11750.0,
        "default_duration_sec": 60.0,
    },
    "pat3": {
        "name": "Patient 3",
        "condition": "Frequent Paroxysmal Ectopy (SVPB, VE, Bigeminy)",
        "invert": False,
        "fs": 180.0,
        "default_start_sec": 20.0,
        "default_duration_sec": 60.0,
    },
    "pat4": {
        "name": "Patient 4",
        "condition": "Normal Sinus Rhythm Specificity Control (99.99% Normal)",
        "invert": False,
        "fs": 180.0,
        "default_start_sec": 100.0,
        "default_duration_sec": 60.0,
    },
    "pat5": {
        "name": "Patient 5",
        "condition": "Normal Sinus Rhythm Specificity Control (99.93% Normal)",
        "invert": False,
        "fs": 180.0,
        "default_start_sec": 100.0,
        "default_duration_sec": 60.0,
    }
}


class QRSAssessmentResult(dict):
    """
    Structured container for QRS assessment results.
    Behaves as a dictionary, allows attribute access (res.TP, res.f1_score),
    supports direct 4-tuple unpacking (`tp, tn, fp, fn = res`),
    and includes .plot_static() and .plot_animated() visualization methods.
    """

    def __iter__(self):
        """Allows direct unpacking: `tp, tn, fp, fn = assess_patient_qrs(...)`."""
        return iter((self["TP"], self["TN"], self["FP"], self["FN"]))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'QRSAssessmentResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def summary(self) -> str:
        """Returns a formatted clinical assessment report."""
        lines = [
            "=" * 70,
            f"          QRS DIAGNOSTIC ASSESSMENT REPORT: {self.get('patient', 'PATIENT').upper()}",
            "=" * 70,
            f"  Window Start        : {self.get('start_sec', 0.0):.2f} s (sample {self.get('start_sample', 0):,})",
            f"  Window Duration     : {self.get('duration_sec', 0.0):.2f} s ({self.get('num_samples', 0):,} samples)",
            f"  Polarity Inverted   : {self.get('inverted', False)}",
            f"  Coincidence Window  : <= {self.get('tolerance', 10)} samples (~{(self.get('tolerance', 10)/self.get('fs', 180.0)*1000):.1f} ms)",
            "-" * 70,
            "  BEAT DETECTION & COINCIDENCE SUMMARY:",
            f"    Ground-Truth Beats: {self.get('total_ref_beats', 0):,}  (Abnormal: {self.get('ref_abnormal_beats', 0):,}, Normal: {self.get('ref_normal_beats', 0):,})",
            f"    Our Detected Peaks: {self.get('total_detected_peaks', 0):,}",
            f"    Matched Beats (<=T): {self.get('matched_beats', 0):,} ({self.get('matched_ratio', 0.0):.1f}% of reference beats)",
            "-" * 70,
            "  CONFUSION MATRIX (Abnormal Arrhythmia Classification):",
            f"    True Positives  (TP) : {self['TP']:,}   [Cardiologist Abnormal -> Model Abnormal]",
            f"    True Negatives  (TN) : {self['TN']:,}   [Cardiologist Normal   -> Model Normal]",
            f"    False Positives (FP) : {self['FP']:,}   [Cardiologist Normal   -> Model Abnormal]",
            f"    False Negatives (FN) : {self['FN']:,}   [Cardiologist Abnormal -> Model Normal]",
            "-" * 70,
            "  CLINICAL PERFORMANCE METRICS:",
            f"    Sensitivity (Recall): {self.get('sensitivity', 0.0):6.2f}%  (TP / [TP + FN])",
            f"    Specificity         : {self.get('specificity', 0.0):6.2f}%  (TN / [TN + FP])",
            f"    Precision (PPV)     : {self.get('precision', 0.0):6.2f}%  (TP / [TP + FP])",
            f"    Negative Pred (NPV) : {self.get('npv', 0.0):6.2f}%  (TN / [TN + FN])",
            f"    Accuracy            : {self.get('accuracy', 0.0):6.2f}%  ([TP + TN] / Total)",
            f"    F1-Score            : {self.get('f1_score', 0.0):6.2f}%  (Harmonic Mean of Sens & Prec)",
            "-" * 70,
            "  R-PEAK DETECTION SENSITIVITY (AAMI EC57):",
            f"    Detection TP / FP / FN : {self.get('detection_TP', 0):,} / {self.get('detection_FP', 0):,} / {self.get('detection_FN', 0):,}",
            f"    Detection Sensitivity  : {self.get('detection_sensitivity', 0.0):6.2f}%",
            f"    Detection Positive Pred: {self.get('detection_ppv', 0.0):6.2f}%",
            "=" * 70,
        ]
        return "\n".join(lines)

    def plot_static(self, save_path: Optional[str] = None, show: bool = True):
        """Displays static ECG strip plot for this interval with SQI and 4 annotations."""
        if "_plot_data" not in self:
            raise ValueError("Plot data not available in this result object.")
        pd = self["_plot_data"]
        return plot_static_qrs(
            sig=pd["sig"],
            time_axis=pd["time_axis"],
            det_peaks=pd["det_peaks"],
            ref_peaks=pd["ref_peaks"],
            ref_is_abnormal=pd["ref_is_abnormal"],
            our_abnormal_peaks=pd["our_abnormal_peaks"],
            sqi_scores=pd["sqi_scores"],
            patient_id=self.get("patient", "PAT"),
            condition=self.get("condition", ""),
            metrics=self,
            save_path=save_path,
            show=show
        )

    def plot_animated(self, speed: float = 2.0, window: float = 8.0):
        """Launches animated real-time telemetry streaming for this interval."""
        if "_plot_data" not in self:
            raise ValueError("Plot data not available in this result object.")
        pd = self["_plot_data"]
        return plot_animated_qrs(
            sig=pd["sig"],
            time_axis=pd["time_axis"],
            det_peaks=pd["det_peaks"],
            ref_peaks=pd["ref_peaks"],
            ref_is_abnormal=pd["ref_is_abnormal"],
            our_abnormal_peaks=pd["our_abnormal_peaks"],
            sqi_scores=pd["sqi_scores"],
            fs=self.get("fs", 180.0),
            speed=speed,
            window_sec=window,
            patient_id=self.get("patient", "PAT"),
            condition=self.get("condition", ""),
            metrics=self
        )

    def __repr__(self) -> str:
        return f"<QRSAssessmentResult TP={self['TP']} TN={self['TN']} FP={self['FP']} FN={self['FN']} F1={self.get('f1_score', 0.0):.1f}%>"


def resolve_patient_paths(patient: Union[str, int]) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Resolves patient ID, signal file path, annotation path, and configuration.
    Accepts 'pat1', 'pat2', 1, 2, 'PAT2', or a path.
    """
    if isinstance(patient, int):
        clean_id = f"pat{patient}"
    else:
        clean_id = str(patient).strip().lower()
        if clean_id.isdigit():
            clean_id = f"pat{clean_id}"

    cfg = PATIENT_REGISTRY.get(clean_id, {
        "name": clean_id.upper(),
        "condition": "Custom Holter Subject",
        "invert": False,
        "fs": 180.0,
        "default_start_sec": 0.0,
        "default_duration_sec": 60.0
    })

    candidates_dat = [
        os.path.join(PATS_DEFAULT_DIR, clean_id, "DataII.dat"),
        os.path.join(PATS_ALT_DIR, clean_id, "DataII.dat"),
        os.path.join(str(patient), "DataII.dat") if os.path.isdir(str(patient)) else str(patient)
    ]
    dat_path = next((p for p in candidates_dat if os.path.exists(p)), None)
    if not dat_path or not os.path.exists(dat_path):
        raise FileNotFoundError(f"Signal file DataII.dat not found for patient '{patient}' in {candidates_dat[:2]}")

    candidates_ann = [
        os.path.join(PATS_DEFAULT_DIR, clean_id, f"{clean_id}.txt"),
        os.path.join(PATS_ALT_DIR, clean_id, f"{clean_id}.txt"),
        os.path.join(str(patient), f"{clean_id}.txt") if os.path.isdir(str(patient)) else ""
    ]
    ann_path = next((p for p in candidates_ann if os.path.exists(p)), None)
    if not ann_path or not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file {clean_id}.txt not found for patient '{patient}' in {candidates_ann[:2]}")

    return clean_id, dat_path, ann_path, cfg


# Global singleton classifier cache for fast repeated inferences
_CACHED_CLASSIFIER: Optional[HierarchicalQRSClassifier] = None


def get_classifier(model_path: str = MODEL_DEFAULT_PATH) -> HierarchicalQRSClassifier:
    """Returns a cached HierarchicalQRSClassifier instance."""
    global _CACHED_CLASSIFIER
    if _CACHED_CLASSIFIER is None or _CACHED_CLASSIFIER.model_path != model_path:
        _CACHED_CLASSIFIER = HierarchicalQRSClassifier(model_path=model_path)
    return _CACHED_CLASSIFIER


# -----------------------------------------------------------------------------
# PLOTTING ENGINE: OPTION 1 (STATIC) & OPTION 2 (ANIMATED)
# -----------------------------------------------------------------------------

def plot_static_qrs(
    sig: np.ndarray,
    time_axis: np.ndarray,
    det_peaks: np.ndarray,
    ref_peaks: np.ndarray,
    ref_is_abnormal: np.ndarray,
    our_abnormal_peaks: np.ndarray,
    sqi_scores: np.ndarray,
    patient_id: str = "PAT",
    condition: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plots the ECG signal in the given interval statically with:
      - Continuous SQI-colored waveform
      - The 4 distinct clinical markers
      - TP, TN, FP, FN and performance metrics HUD banner
    """
    fig, ax = plt.subplots(figsize=(15, 7.2), facecolor="#0B0F19")
    ax.set_facecolor("#111827")

    # Grid & Axis Styling
    ax.grid(True, color="#1F2937", linestyle="--", alpha=0.8)
    ax.set_ylabel("ECG Amplitude (ADC Units)", color="#F3F4F6", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time along Recording (Seconds)", color="#9CA3AF", fontsize=10)
    ax.tick_params(colors="#9CA3AF", labelsize=9.5)

    y_min, y_max = np.percentile(sig, 0.5), np.percentile(sig, 99.5)
    y_range = max(50.0, y_max - y_min)
    ax.set_ylim(y_min - 0.18 * y_range, y_max + 0.50 * y_range)
    ax.set_xlim(time_axis[0], time_axis[-1])

    # 1. Continuous SQI-Colored Waveform
    cmap = get_noise_colormap()
    norm = Normalize(vmin=0.0, vmax=1.0)
    points = np.column_stack([time_axis, sig]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.2, zorder=3)
    lc.set_array(sqi_scores[:-1])
    ax.add_collection(lc)

    # 2. The 4 Distinct Clinical Markers:
    # 1. Our Detected R-Peaks (Purple Circle on peak)
    if len(det_peaks) > 0:
        ax.scatter(
            time_axis[det_peaks], sig[det_peaks],
            color="#C084FC", marker="o", s=80, edgecolors="#7E22CE", linewidths=1.2, zorder=5,
            label="1. Our R-Peaks (Purple)"
        )
    else:
        ax.scatter([], [], color="#C084FC", marker="o", s=80, label="1. Our R-Peaks (Purple)")

    # 2. Cardiologist Labeled Peaks (Green Triangle pointing down)
    if len(ref_peaks) > 0:
        ax.scatter(
            time_axis[ref_peaks], sig[ref_peaks] + 0.04 * y_range,
            color="#22C55E", marker="v", s=90, edgecolors="#14532D", linewidths=1.2, zorder=6,
            label="2. Labeled Peaks (Green)"
        )
    else:
        ax.scatter([], [], color="#22C55E", marker="v", s=90, label="2. Labeled Peaks (Green)")

    # 3. Labeled Abnormal QRS (Gold Star hovering above peak)
    lab_abn_peaks = ref_peaks[ref_is_abnormal == 1]
    if len(lab_abn_peaks) > 0:
        ax.scatter(
            time_axis[lab_abn_peaks], sig[lab_abn_peaks] + 0.11 * y_range,
            color="#FACC15", marker="*", s=150, edgecolors="#78350F", linewidths=1.2, zorder=7,
            label="3. Labeled Abnormal (Gold)"
        )
    else:
        ax.scatter([], [], color="#FACC15", marker="*", s=150, label="3. Labeled Abnormal (Gold)")

    # 4. Our Detected Abnormal QRS (Bright Crimson Diamond on peak)
    if len(our_abnormal_peaks) > 0:
        ax.scatter(
            time_axis[our_abnormal_peaks], sig[our_abnormal_peaks],
            color="#EF4444", marker="D", s=100, edgecolors="#FFFFFF", linewidths=1.4, zorder=8,
            label="4. Detected Abnormal (Red)"
        )
    else:
        ax.scatter([], [], color="#EF4444", marker="D", s=100, label="4. Detected Abnormal (Red)")

    # Legend in Upper Right
    ax.legend(
        loc="upper right", ncol=2, framealpha=0.92, facecolor="#0F172A",
        edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8.8
    )

    # Diagnostic HUD Banners
    dur_sec = time_axis[-1] - time_axis[0]
    ax.text(
        0.012, 0.94,
        f"PATIENT: {patient_id.upper()}  |  Interval: [{time_axis[0]:.1f}s - {time_axis[-1]:.1f}s] ({dur_sec:.1f}s)",
        transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#38BDF8",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#03254C", edgecolor="#0284C7", alpha=0.9),
        zorder=10
    )

    if metrics:
        tp, tn, fp, fn = metrics.get("TP", 0), metrics.get("TN", 0), metrics.get("FP", 0), metrics.get("FN", 0)
        rec, spec, f1 = metrics.get("sensitivity", 0.0), metrics.get("specificity", 0.0), metrics.get("f1_score", 0.0)
        stats_str = f"TP={tp:,} | TN={tn:,} | FP={fp:,} | FN={fn:,}  |  Sens: {rec:.1f}% | Spec: {spec:.1f}% | F1: {f1:.1f}%"
    else:
        stats_str = f"Detected: {len(det_peaks)} beats | Labeled: {len(ref_peaks)} beats (Abnormal: {len(lab_abn_peaks)})"

    ax.text(
        0.012, 0.86,
        stats_str,
        transform=ax.transAxes, fontsize=8.8, fontweight="bold", color="#E2E8F0",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", edgecolor="#475569", alpha=0.85),
        zorder=10
    )

    ax.text(
        0.52, 0.94,
        "SQI Waveform [Blue: Clean | Coral: Mild Noise | Dark-Red: Severe Artifact]",
        transform=ax.transAxes, ha="center", fontsize=8.5, fontweight="bold", color="#94A3B8",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F172A", edgecolor="#334155", alpha=0.85),
        zorder=10
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_animated_qrs(
    sig: np.ndarray,
    time_axis: np.ndarray,
    det_peaks: np.ndarray,
    ref_peaks: np.ndarray,
    ref_is_abnormal: np.ndarray,
    our_abnormal_peaks: np.ndarray,
    sqi_scores: np.ndarray,
    fs: float = 180.0,
    speed: float = 2.0,
    window_sec: float = 8.0,
    patient_id: str = "PAT",
    condition: str = "",
    metrics: Optional[Dict[str, Any]] = None,
):
    """
    Plots the ECG signal in the given interval as real-time animated telemetry with:
      - Continuous scrolling SQI waveform
      - The 4 clinical markers updated dynamically
      - Interactive keyboard controls (SPACE: Pause, Arrow keys, R: Rewind)
    """
    fig, ax = plt.subplots(figsize=(15, 7.5), facecolor="#0B0F19")
    ax.set_facecolor("#111827")

    ax.grid(True, color="#1F2937", linestyle="--", alpha=0.8)
    ax.set_ylabel("ECG Amplitude (ADC Units)", color="#F3F4F6", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time along Recording (Seconds)", color="#9CA3AF", fontsize=10)
    ax.tick_params(colors="#9CA3AF", labelsize=9.5)

    y_min, y_max = np.percentile(sig, 0.5), np.percentile(sig, 99.5)
    y_range = max(50.0, y_max - y_min)
    ax.set_ylim(y_min - 0.18 * y_range, y_max + 0.50 * y_range)

    cmap = get_noise_colormap()
    norm = Normalize(vmin=0.0, vmax=1.0)
    lc = LineCollection([], cmap=cmap, norm=norm, linewidth=2.2, zorder=3)
    ax.add_collection(lc)

    # 4 Clinical Markers
    sc_our_peaks = ax.scatter(
        [], [], color="#C084FC", marker="o", s=80, edgecolors="#7E22CE", linewidths=1.2, zorder=5,
        label="1. Our R-Peaks (Purple)"
    )
    sc_lab_peaks = ax.scatter(
        [], [], color="#22C55E", marker="v", s=90, edgecolors="#14532D", linewidths=1.2, zorder=6,
        label="2. Labeled Peaks (Green)"
    )
    sc_lab_abnormal = ax.scatter(
        [], [], color="#FACC15", marker="*", s=150, edgecolors="#78350F", linewidths=1.2, zorder=7,
        label="3. Labeled Abnormal (Gold)"
    )
    sc_our_abnormal = ax.scatter(
        [], [], color="#EF4444", marker="D", s=100, edgecolors="#FFFFFF", linewidths=1.4, zorder=8,
        label="4. Detected Abnormal (Red)"
    )

    ax.legend(
        loc="upper right", ncol=2, framealpha=0.92, facecolor="#0F172A",
        edgecolor="#334155", labelcolor="#F1F5F9", fontsize=8.8
    )

    # Top Banners
    dur_sec = time_axis[-1] - time_axis[0]
    short_cond = condition.split("(")[0].strip() if condition else ""
    title_banner = ax.text(
        0.012, 0.94,
        f"PATIENT: {patient_id.upper()} | {short_cond} | Interval: [{time_axis[0]:.1f}s - {time_axis[-1]:.1f}s] ({dur_sec:.1f}s)",
        transform=ax.transAxes, fontsize=9.2, fontweight="bold", color="#38BDF8",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#03254C", edgecolor="#0284C7", alpha=0.9),
        zorder=10
    )

    hud_stats = ax.text(
        0.012, 0.86,
        "Visible Beats: 0 | Live HR: -- BPM | SQI: 0.00 (Clean Blue)",
        transform=ax.transAxes, fontsize=8.8, fontweight="bold", color="#E2E8F0",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", edgecolor="#475569", alpha=0.85),
        zorder=10
    )

    controls_banner = ax.text(
        0.52, 0.94,
        f"SPEED: {speed:.1f}x [SPACE: Pause | <-/->: Step 1s | Up/Down: Speed | R: Rewind]",
        transform=ax.transAxes, ha="center", fontsize=8.5, fontweight="bold", color="#94A3B8",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F172A", edgecolor="#334155", alpha=0.85),
        zorder=10
    )

    # Animation State
    sig_len = len(sig)
    window_samples = min(sig_len, int(window_sec * fs))
    step_per_frame = int(max(1, (fs / 30.0) * speed))
    anim_state = {
        "cur_idx": 0,
        "is_paused": False,
        "speed": speed,
        "step": step_per_frame
    }

    def on_key(event):
        if event.key == " ":
            anim_state["is_paused"] = not anim_state["is_paused"]
            status_txt = "PAUSED" if anim_state["is_paused"] else "STREAMING"
            controls_banner.set_text(f"SPEED: {anim_state['speed']:.1f}x [{status_txt} | SPACE: Pause | <-/->: Step | R: Rewind]")
        elif event.key == "right":
            anim_state["cur_idx"] = min(sig_len - window_samples, anim_state["cur_idx"] + int(fs))
            controls_banner.set_text(f"SPEED: {anim_state['speed']:.1f}x [STEPPED +1s | SPACE: Resume]")
        elif event.key == "left":
            anim_state["cur_idx"] = max(0, anim_state["cur_idx"] - int(fs))
            controls_banner.set_text(f"SPEED: {anim_state['speed']:.1f}x [STEPPED -1s | SPACE: Resume]")
        elif event.key == "up":
            anim_state["speed"] = min(10.0, anim_state["speed"] + 0.5)
            anim_state["step"] = int(max(1, (fs / 30.0) * anim_state["speed"]))
            controls_banner.set_text(f"SPEED: {anim_state['speed']:.1f}x [STREAMING | SPACE: Pause]")
        elif event.key == "down":
            anim_state["speed"] = max(0.5, anim_state["speed"] - 0.5)
            anim_state["step"] = int(max(1, (fs / 30.0) * anim_state["speed"]))
            controls_banner.set_text(f"SPEED: {anim_state['speed']:.1f}x [STREAMING | SPACE: Pause]")
        elif event.key in ("r", "R"):
            anim_state["cur_idx"] = 0
            controls_banner.set_text(f"SPEED: {anim_state['speed']:.1f}x [REWOUND | SPACE: Pause]")
        elif event.key in ("q", "Q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    lab_abnormal_peaks = ref_peaks[ref_is_abnormal == 1]


    def update_frame(frame_num):
        if not anim_state["is_paused"]:
            anim_state["cur_idx"] += anim_state["step"]
            if anim_state["cur_idx"] + window_samples >= sig_len:
                anim_state["cur_idx"] = 0  # loop back

        idx_start = anim_state["cur_idx"]
        idx_end = idx_start + window_samples

        x_win = time_axis[idx_start:idx_end]
        y_win = sig[idx_start:idx_end]
        s_win = sqi_scores[idx_start:idx_end]

        points = np.column_stack([x_win, y_win]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc.set_segments(segments)
        lc.set_array(s_win[:-1])

        ax.set_xlim(x_win[0], x_win[-1])

        # 1. Our R-Peaks in Window
        m_our = (det_peaks >= idx_start) & (det_peaks < idx_end)
        our_pks_win = det_peaks[m_our]
        if len(our_pks_win) > 0:
            sc_our_peaks.set_offsets(np.column_stack([time_axis[our_pks_win], sig[our_pks_win]]))
        else:
            sc_our_peaks.set_offsets(np.empty((0, 2)))

        # 2. Labeled Reference Peaks in Window
        m_ref = (ref_peaks >= idx_start) & (ref_peaks < idx_end)
        ref_pks_win = ref_peaks[m_ref]
        if len(ref_pks_win) > 0:
            sc_lab_peaks.set_offsets(np.column_stack([time_axis[ref_pks_win], sig[ref_pks_win] + 0.04 * y_range]))
        else:
            sc_lab_peaks.set_offsets(np.empty((0, 2)))

        # 3. Labeled Abnormal Beats in Window
        m_lab_abn = (lab_abnormal_peaks >= idx_start) & (lab_abnormal_peaks < idx_end)
        lab_abn_win = lab_abnormal_peaks[m_lab_abn]
        if len(lab_abn_win) > 0:
            sc_lab_abnormal.set_offsets(np.column_stack([time_axis[lab_abn_win], sig[lab_abn_win] + 0.11 * y_range]))
        else:
            sc_lab_abnormal.set_offsets(np.empty((0, 2)))

        # 4. Our Detected Abnormal Beats in Window
        m_our_abn = (our_abnormal_peaks >= idx_start) & (our_abnormal_peaks < idx_end)
        our_abn_win = our_abnormal_peaks[m_our_abn]
        if len(our_abn_win) > 0:
            sc_our_abnormal.set_offsets(np.column_stack([time_axis[our_abn_win], sig[our_abn_win]]))
        else:
            sc_our_abnormal.set_offsets(np.empty((0, 2)))

        live_hr = 0.0
        if len(our_pks_win) >= 2:
            live_rr = np.diff(our_pks_win) / fs
            live_hr = 60.0 / np.median(live_rr)

        cur_sqi = s_win[-1] if len(s_win) > 0 else 0.0
        if cur_sqi < 0.35:
            sqi_str = f"{cur_sqi:.2f} (Clean Blue)"
        elif cur_sqi < 0.70:
            sqi_str = f"{cur_sqi:.2f} (Mild Noise - Coral)"
        else:
            sqi_str = f"{cur_sqi:.2f} (Severe Artifact - Dark Red)"

        hr_str = f"{live_hr:.0f} BPM" if live_hr > 0 else "-- BPM"
        hud_stats.set_text(
            f"Visible Beats: {len(our_pks_win)} (Abnormal Detected: {len(our_abn_win)} | Labeled Abnormal: {len(lab_abn_win)}) | "
            f"Live HR: {hr_str} | SQI: {sqi_str}"
        )

        return (lc, sc_our_peaks, sc_lab_peaks, sc_lab_abnormal, sc_our_abnormal, hud_stats)

    anim = FuncAnimation(fig, update_frame, interval=33, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return anim


# -----------------------------------------------------------------------------
# CORE ASSESSMENT FUNCTION WITH INTEGRATED PLOTTING
# -----------------------------------------------------------------------------

def assess_patient_qrs(
    patient: Union[str, int],
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    start_sample: Optional[int] = None,
    num_samples: Optional[int] = None,
    r_peaks: Optional[np.ndarray] = None,
    tolerance: int = 10,
    invert: Optional[bool] = None,
    fs: Optional[float] = None,
    model_path: str = MODEL_DEFAULT_PATH,
    classifier: Optional[HierarchicalQRSClassifier] = None,
    compute_sqi: bool = True,
    give_detail: bool = False,
    print_report: bool = False,
    plot: Optional[str] = None,
    speed: float = 2.0,
    window: float = 8.0,
    save_plot: Optional[str] = None,
) -> QRSAssessmentResult:
    """
    Evaluates QRS classification performance (TP, TN, FP, FN) on a patient over
    an arbitrary duration, with optional Static or Animated SQI-colored visualization.

    Parameters:
    -----------
    patient : str or int
        Patient identifier (e.g. 'pat1', 'pat2', 'pat3', 'pat4', 'pat5', or 1, 2, 3).
    start_sec : float, optional
        Starting time in seconds (defaults to 0.0 or patient arrhythmia onset).
    duration_sec : float, optional
        Duration in seconds. If None or <= 0, evaluates the entire recording.
    start_sample : int, optional
        Starting sample index. Overrides start_sec if provided.
    num_samples : int, optional
        Number of samples to evaluate. Overrides duration_sec if provided.
    r_peaks : array-like, optional
        User-supplied R-peak indices (relative to the loaded signal slice, i.e.
        0-based within the window).  When provided, the built-in RPeakDetector
        is skipped entirely — only peaks within a `tolerance`-sample threshold
        of labeled peaks are counted.  Accepts any 1-D array or list of ints.
    tolerance : int, default=10
        Coincidence tolerance window in samples (default: 10 samples ~ 55ms @ 180Hz).
    invert : bool, optional
        Whether to invert ECG signal polarity. If None, automatically determined from
        patient configuration (pat2 is inverted; others are normal).
    fs : float, optional
        Sampling frequency in Hz (default: 180.0 Hz).
    model_path : str, default=MODEL_DEFAULT_PATH
        Path to the trained CNN/Hierarchical model file.
    classifier : HierarchicalQRSClassifier, optional
        Pre-instantiated classifier instance. If None, uses cached singleton.
    compute_sqi : bool, default=True
        If True, computes SQI scores for waveform coloring (takes ~3-4 s for 120 s
        of signal).  Set to False to skip SQI entirely — the waveform is drawn in
        a uniform cyan color, which is much faster.
    give_detail : bool, default=False
        If True, the result dict will also contain:
          - 'detected_abnormal_qrs' : int array of sample positions of our detected abnormal beats
          - 'labeled_abnormal_qrs'  : int array of sample positions of cardiologist-labeled abnormal beats
          - 'detected_r_peaks'      : int array of all our detected R-peak positions
          - 'labeled_r_peaks'       : int array of all labeled R-peak positions
        All positions are 0-based within the signal window.
    print_report : bool, default=False
        If True, prints a formatted clinical report to the console.
    plot : {'static', 'animated', None}, optional
        Visualization mode:
          - 'static'   : Displays a static ECG strip with SQI coloring and 4 markers.
          - 'animated' : Displays real-time streaming telemetry with interactive controls.
          - None       : No plot, pure calculation (fastest).
    speed : float, default=2.0
        Animation speed multiplier (used if plot='animated').
    window : float, default=8.0
        Viewport scrolling window width in seconds (used if plot='animated').
    save_plot : str, optional
        Path to save the static plot image (e.g. 'interval.png').

    Returns:
    --------
    QRSAssessmentResult:
        Dict-like object with TP, TN, FP, FN and clinical metrics.
        Supports direct 4-tuple unpacking: `tp, tn, fp, fn = assess_patient_qrs(...)`
        Supports methods: `res.plot_static()` and `res.plot_animated()`
    """
    pat_id, dat_path, ann_path, cfg = resolve_patient_paths(patient)
    sampling_rate = fs if fs is not None else cfg.get("fs", 180.0)

    # Resolve start and duration in samples
    if start_sample is not None:
        start_samp = int(start_sample)
    elif start_sec is not None:
        start_samp = int(round(start_sec * sampling_rate))
    else:
        start_samp = 0

    if num_samples is not None:
        count = int(num_samples) if num_samples > 0 else -1
    elif duration_sec is not None and duration_sec > 0:
        count = int(round(duration_sec * sampling_rate))
    else:
        count = -1  # Load entire file

    should_invert = cfg["invert"] if invert is None else bool(invert)

    # 1. Read ECG Signal Slice
    with open(dat_path, "rb") as f:
        if start_samp > 0:
            f.seek(start_samp * 2)
        sig = np.fromfile(f, dtype="<i2", count=count).astype(np.float64)

    if should_invert:
        sig = sig * -1.0

    sig_len = len(sig)
    end_samp = start_samp + sig_len
    actual_duration_sec = sig_len / sampling_rate
    time_axis = (start_samp + np.arange(sig_len)) / sampling_rate
    from functions import r_detection
    det_peaks=r_detection(sig=sig,fs=fs)
    print("finished processing")
    # print(det_peaks)
    # 2. R-Peak Detection: user-supplied or built-in detector
    # if r_peaks is not None:
    #     print("im here")
    #     det_peaks = np.asarray(r_peaks, dtype=np.int64).ravel()
    #     # Filter to valid range within signal slice
    #     det_peaks = det_peaks[(det_peaks >= 0) & (det_peaks < sig_len)]
    # else:
    #     detector = RPeakDetector(sampling_rate=sampling_rate, method="adaptive_prominence")
    #     det_res = detector.detect(sig)
    #     det_peaks = np.array(det_res.r_peaks, dtype=np.int64)
    
    # print(det_peaks)
    # 3. Read Cardiologist Ground-Truth Annotations
    ref_samples_all, ref_labels_all = read_holter_labels(
        ann_path,
        start_sample=start_samp,
        upto_sample=end_samp,
        return_peaks=True
    )
    ref_peaks = ref_samples_all - start_samp
    ref_is_abnormal = (ref_labels_all != 0).astype(int)

    # Filter strictly within the current slice
    valid_mask = (ref_peaks >= 0) & (ref_peaks < sig_len)
    ref_peaks = ref_peaks[valid_mask]
    ref_is_abnormal = ref_is_abnormal[valid_mask]

    n_ref_total = len(ref_peaks)
    n_ref_abn = int(np.sum(ref_is_abnormal == 1))
    n_ref_norm = n_ref_total - n_ref_abn

    # 4. Run Hierarchical QRS Arrhythmia Classifier Engine
    engine = classifier if classifier is not None else get_classifier(model_path)
    cls_res = engine.classify_recording(sig, det_peaks, fs=sampling_rate)
    det_preds = cls_res["predictions"]  # 0 = Normal, 1 = Abnormal
    our_abnormal_peaks = det_peaks[det_preds == 1]

    # -------------------------------------------------------------------------
    # 5. Beat Coincidence Matching & Confusion Matrix (TP, TN, FP, FN)
    # -------------------------------------------------------------------------
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    matched_det_indices = set()
    matched_ref_count = 0

    for r_pk, r_abn in zip(ref_peaks, ref_is_abnormal):
        if len(det_peaks) == 0:
            if r_abn == 1:
                fn += 1
            continue

        dists = np.abs(det_peaks - r_pk)
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_dist <= tolerance:
            matched_ref_count += 1
            matched_det_indices.add(min_idx)
            pred = det_preds[min_idx]

            if r_abn == 1 and pred == 1:
                tp += 1
            elif r_abn == 0 and pred == 0:
                tn += 1
            elif r_abn == 0 and pred == 1:
                fp += 1
            elif r_abn == 1 and pred == 0:
                fn += 1
        else:
            if r_abn == 1:
                fn += 1  # Missed abnormal beat

    # -------------------------------------------------------------------------
    # 6. Detection & Classification Metrics
    # -------------------------------------------------------------------------
    det_tp = matched_ref_count
    det_fn = n_ref_total - matched_ref_count
    det_fp = len(det_peaks) - len(matched_det_indices)

    det_sens = (det_tp / (det_tp + det_fn) * 100.0) if (det_tp + det_fn) > 0 else 100.0
    det_ppv = (det_tp / (det_tp + det_fp) * 100.0) if (det_tp + det_fp) > 0 else 100.0

    sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else (100.0 if n_ref_abn == 0 else 0.0)
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else (100.0 if n_ref_norm == 0 else 0.0)
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else (100.0 if (tp + fn) == 0 else 0.0)
    npv = (tn / (tn + fn) * 100.0) if (tn + fn) > 0 else 100.0
    acc = ((tp + tn) / (tp + tn + fp + fn) * 100.0) if (tp + tn + fp + fn) > 0 else 100.0
    f1 = (2 * tp / (2 * tp + fp + fn) * 100.0) if (2 * tp + fp + fn) > 0 else 0.0
    matched_ratio = (matched_ref_count / n_ref_total * 100.0) if n_ref_total > 0 else 100.0

    # -------------------------------------------------------------------------
    # 7. Continuous SQI Scoring (skippable via compute_sqi=False for speed)
    # -------------------------------------------------------------------------
    sqi_scores = None
    if compute_sqi and (plot is not None or actual_duration_sec <= 300.0):
        sqi_engine = SQIDetector(window_sec=10.0, step_sec=2.0)
        _, sqi_scores = sqi_engine.get_sample_colors(sig, fs=sampling_rate)
    else:
        sqi_scores = np.zeros(sig_len, dtype=np.float32)

    # Package Result
    result = QRSAssessmentResult({
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "signal" : sig , 
        "sensitivity": round(float(sens), 2),
        "recall": round(float(sens), 2),
        "specificity": round(float(spec), 2),
        "precision": round(float(prec), 2),
        "ppv": round(float(prec), 2),
        "npv": round(float(npv), 2),
        "accuracy": round(float(acc), 2),
        "f1_score": round(float(f1), 2),

        "patient": pat_id,
        "condition": cfg.get("condition", ""),
        "start_sec": float(start_samp / sampling_rate),
        "duration_sec": round(float(actual_duration_sec), 2),
        "start_sample": int(start_samp),
        "num_samples": int(sig_len),
        "fs": float(sampling_rate),
        "inverted": bool(should_invert),
        "tolerance": int(tolerance),
        "total_ref_beats": int(n_ref_total),
        "ref_abnormal_beats": int(n_ref_abn),
        "ref_normal_beats": int(n_ref_norm),
        "total_detected_peaks": int(len(det_peaks)),
        "matched_beats": int(matched_ref_count),
        "matched_ratio": round(float(matched_ratio), 2),
        "ref_labels":ref_labels_all,
        "detection_TP": int(det_tp),
        "detection_FP": int(det_fp),
        "detection_FN": int(det_fn),
        "detection_sensitivity": round(float(det_sens), 2),
        "detection_ppv": round(float(det_ppv), 2),
        "predicted_beats" : det_preds,

        # Internal raw data for instant re-plotting without recomputation
        "_plot_data": {
            "sig": sig,
            "time_axis": time_axis,
            "det_peaks": det_peaks,
            "ref_peaks": ref_peaks,
            "ref_is_abnormal": ref_is_abnormal,
            "our_abnormal_peaks": our_abnormal_peaks,
            "sqi_scores": sqi_scores
        }
    })

    # Inject detailed beat-position arrays when requested
    if give_detail:
        lab_abn_positions = ref_peaks[ref_is_abnormal == 1].astype(np.int64)
        result["detected_abnormal_qrs"] = our_abnormal_peaks.astype(np.int64)
        result["labeled_abnormal_qrs"] = lab_abn_positions
        result["detected_r_peaks"] = det_peaks.astype(np.int64)
        result["labeled_r_peaks"] = ref_peaks.astype(np.int64)
        result["predicted_beats"] = det_preds.astype(np.int64)
    if print_report:
        print(result.summary())

    # Execute visualization if requested
    if plot:
        clean_plot = str(plot).strip().lower()
        if clean_plot in ("static", "stat"):
            result.plot_static(save_path=save_plot, show=True)
        elif clean_plot in ("animated", "animation", "anim"):
            result.plot_animated(speed=speed, window=window)

    return result


def get_patient_sqi(
    patient: Union[str, int],
    start_sample: int = 0,
    end_sample: Optional[int] = None,
    invert: Optional[bool] = None,
    fs: Optional[float] = None,
    return_hex: bool = False,
    window_sec: float = 10.0,
    step_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Returns per-sample SQI colors and noise-severity scores for a patient segment.

    Parameters:
    -----------
    patient : str or int
        Patient identifier ('pat1', 'pat2', ... or 1, 2, ...).
    start_sample : int
        First sample index (0-based absolute position in the recording).
    end_sample : int, optional
        Last sample index (exclusive). If None, reads to end of file.
    invert : bool, optional
        Force polarity inversion. If None, uses patient registry default.
    fs : float, optional
        Sampling rate (default: 180.0 Hz from registry).
    return_hex : bool, default=False
        If True  -> colors is a list of hex strings ('#0055D4', ...).
        If False -> colors is an RGBA float array of shape (N, 4).
    window_sec : float, default=10.0
        SQI analysis window length in seconds.
    step_sec : float, default=2.0
        SQI analysis step size in seconds.

    Returns:
    --------
    dict with keys:
        'colors'       : RGBA array (N,4) or list of hex strings
        'scores'       : np.ndarray of shape (N,) with noise severity in [0, 1]
                         0.0 = perfectly clean, 1.0 = severe artifact
        'signal'       : the loaded ECG segment (np.float64)
        'start_sample' : int
        'end_sample'   : int
        'fs'           : float
        'patient'      : str

    Interpretation of scores:
        score < 0.35  -> Clean (Blue)
        0.35 <= score < 0.70 -> Mild Noise (Coral/Pink)
        score >= 0.70 -> Severe Artifact (Dark Red)

    Example:
    --------
        from assess_qrs import get_patient_sqi
        info = get_patient_sqi("pat2", start_sample=0, end_sample=21600)
        colors = info['colors']   # (21600, 4) RGBA
        scores = info['scores']   # (21600,)  floats in [0,1]
    """
    pat_id, dat_path, _, cfg = resolve_patient_paths(patient)
    sampling_rate = fs if fs is not None else cfg.get("fs", 180.0)
    should_invert = cfg["invert"] if invert is None else bool(invert)

    # Load signal slice
    with open(dat_path, "rb") as f:
        if start_sample > 0:
            f.seek(start_sample * 2)
        count = (end_sample - start_sample) if end_sample is not None else -1
        sig = np.fromfile(f, dtype="<i2", count=count).astype(np.float64)

    if should_invert:
        sig = sig * -1.0

    actual_end = start_sample + len(sig)

    # Compute SQI
    sqi_engine = SQIDetector(window_sec=window_sec, step_sec=step_sec)
    colors, scores = sqi_engine.get_sample_colors(sig, fs=sampling_rate, return_hex=return_hex)

    return {
        "colors": colors,
        "scores": scores,
        "signal": sig,
        "start_sample": int(start_sample),
        "end_sample": int(actual_end),
        "fs": float(sampling_rate),
        "patient": pat_id,
    }


def plot_patient_qrs(
    patient: Union[str, int],
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    mode: str = "static",
    r_peaks: Optional[np.ndarray] = None,
    compute_sqi: bool = True,
    speed: float = 2.0,
    window: float = 8.0,
    save_path: Optional[str] = None,
    tolerance: int = 10,
    invert: Optional[bool] = None,
) -> QRSAssessmentResult:
    """
    Direct function to assess and plot a patient ECG interval with SQI and 4 annotations.

    Parameters:
    -----------
    patient : str or int
        'pat1', 'pat2', 1, 2, etc.
    start_sec : float, optional
        Start time in seconds.
    duration_sec : float, optional
        Duration in seconds (default: 60s).
    mode : {'static', 'animated'}
        Plotting mode.
    r_peaks : array-like, optional
        User-supplied R-peak indices (0-based within the signal window).
        Skips the built-in detector when provided.
    compute_sqi : bool, default=True
        If False, skips the SQI computation for faster operation.
    speed : float, default=2.0
        Playback speed multiplier for animated mode.
    window : float, default=8.0
        Scrolling window in seconds for animated mode.
    save_path : str, optional
        File path to save the static plot image.
    """
    clean_mode = "animated" if mode.lower() in ("animated", "animation", "anim") else "static"
    return assess_patient_qrs(
        patient=patient,
        start_sec=start_sec,
        duration_sec=duration_sec if duration_sec is not None else 60.0,
        r_peaks=r_peaks,
        plot=clean_mode,
        speed=speed,
        window=window,
        save_plot=save_path,
        tolerance=tolerance,
        invert=invert,
        compute_sqi=compute_sqi,
        print_report=True
    )


# -----------------------------------------------------------------------------
# COMMAND LINE INTERFACE (CLI)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess QRS Performance (TP, TN, FP, FN) & Plot Interval (Static or Animated) with SQI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pat", "-p", type=str, default="pat2", help="Patient ID: pat1, pat2, pat3, pat4, pat5")
    parser.add_argument("--start", "-s", type=float, default=None, help="Start time in seconds (default: clinical onset)")
    parser.add_argument("--duration", "-d", type=float, default=60.0, help="Duration in seconds (0 for full recording)")
    parser.add_argument("--plot", choices=["none", "static", "animated"], default="static", help="Plot mode: static, animated, or none")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed for animation (e.g. 1.0, 2.0)")
    parser.add_argument("--window", "-w", type=float, default=8.0, help="Scrolling window in seconds for animation")
    parser.add_argument("--save", type=str, default=None, help="Optional image path to save static plot (e.g. strip.png)")
    parser.add_argument("--tolerance", "-t", type=int, default=10, help="Tolerance window in samples")
    parser.add_argument("--invert", action="store_true", help="Force signal polarity inversion")
    parser.add_argument("--no_invert", action="store_true", help="Force normal polarity")
    parser.add_argument("--no_sqi", action="store_true", help="Skip SQI computation (much faster, plain cyan waveform)")

    args = parser.parse_args()

    inv = None
    if args.invert:
        inv = True
    elif args.no_invert:
        inv = False

    plot_opt = None if args.plot == "none" else args.plot

    res = assess_patient_qrs(
        patient=args.pat,
        start_sec=args.start,
        duration_sec=args.duration if args.duration > 0 else None,
        plot=plot_opt,
        speed=args.speed,
        window=args.window,
        save_plot=args.save,
        tolerance=args.tolerance,
        invert=inv,
        compute_sqi=not args.no_sqi,
        print_report=True
    )
