"""
SQI Detector Engine - Continuous Noise Classification & Live Waveform Visualizer
================================================================================
A self-contained Signal Quality Index (SQI) module for electrocardiogram (ECG) signals
driven by a trained 19-feature Support Vector Machine (SVM).

Features:
- Rolling sliding-window analysis (e.g. 10s window with fine interval steps).
- Continuous (non-discrete) noise severity score S[n] in [0.0, 1.0] for every sample.
- Sample-level Color Output:
  * Clean (Good)           -> Vibrant Blue (#0055D4)
  * Little Noise (Medium)  -> Light Red / Coral (#FFA0A0 to #FF5555)
  * Very Much Noise (Poor) -> Dark Red to Black-Red (#2B0000)
- Live Real-Time Waveform Player:
  * Streams the ECG signal across the screen like a bedside cardiac monitor.
  * Dynamically colors every sample of the waveform according to its noise severity.
  * Live status indicator badge ("CLEAN", "LITTLE NOISE", "SEVERE ARTIFACT").
  * Spacebar controls Pause / Resume; arrow keys adjust playback speed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.special import softmax

# Scipy compatibility shim for trapz if needed
if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = getattr(np, "trapz", np.trapezoid)

# Import SQI feature calculation functions
try:
    from SQI_module import (
        calculate_s_sqi,
        calculate_k_sqi,
        calculate_sdn_sqi,
        calculate_pur_sqi,
        calculate_bas_sqi,
        calculate_p_sqi,
        calculate_lp_sqi,
        calculate_mp_sqi,
        calculate_hp_sqi,
        calculate_t_sqi,
        calculate_bs_sqi,
        calculate_hf_sqi,
        calculate_e_sqi,
        calculate_i_sqi,
        calculate_rsd_sqi,
        calculate_pca_sqi,
        calculate_pica_sqi,
        calculate_specen_sqi,
        calculate_phasen_sqi,
    )
    _SQI_FUNCTIONS = {
        "s_sqi": (calculate_s_sqi, "signal"),
        "k_sqi": (calculate_k_sqi, "signal"),
        "sdn_sqi": (calculate_sdn_sqi, "signal_fs"),
        "pur_sqi": (calculate_pur_sqi, "signal_fs"),
        "bas_sqi": (calculate_bas_sqi, "signal_fs"),
        "p_sqi": (calculate_p_sqi, "signal_fs"),
        "lp_sqi": (calculate_lp_sqi, "signal_fs"),
        "mp_sqi": (calculate_mp_sqi, "signal_fs"),
        "hp_sqi": (calculate_hp_sqi, "signal_fs"),
        "t_sqi": (calculate_t_sqi, "signal_fs"),
        "bs_sqi": (calculate_bs_sqi, "signal_fs"),
        "hf_sqi": (calculate_hf_sqi, "signal_fs"),
        "e_sqi": (calculate_e_sqi, "signal_fs"),
        "i_sqi": (calculate_i_sqi, "signal_fs"),
        "rsd_sqi": (calculate_rsd_sqi, "signal_fs"),
        "pca_sqi": (calculate_pca_sqi, "signal_fs"),
        "pica_sqi": (calculate_pica_sqi, "signal_fs"),
        "specen_sqi": (calculate_specen_sqi, "signal_fs_freqs"),
        "phasen_sqi": (calculate_phasen_sqi, "signal"),
    }
except ImportError as err:
    raise ImportError(f"Failed to import SQI_module. Make sure SQI_module.py is in the directory: {err}")


def get_noise_colormap() -> LinearSegmentedColormap:
    """
    Continuous color transition:
    - 0.00: Deep Blue (Clean)
    - 0.25: Muted Blue / Transition
    - 0.50: Light Red / Coral (Little Noise)
    - 0.75: Deep Crimson
    - 1.00: Dark Black-Red (Very Much Noise)
    """
    color_stops = [
        (0.00, "#0055D4"),  # Deep vibrant blue
        (0.25, "#3388EE"),  # Blue-cyan transition
        (0.45, "#FFA0A0"),  # Soft pink/light red
        (0.60, "#FF5555"),  # Light-medium coral red
        (0.80, "#A00000"),  # Deep crimson
        (1.00, "#2B0000"),  # Dark black-red
    ]
    return LinearSegmentedColormap.from_list("ecg_noise_continuous", color_stops)


class SQIDetector:
    """
    Signal Quality Index (SQI) Detector for continuous ECG noise evaluation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        window_sec: float = 10.0,
        step_sec: float = 2.0,
    ) -> None:
        """
        Initialize the SQI Detector.

        Args:
            model_path: Path to the trained SVM model. If None, looks for
                        'svm.joblib' or 'sqi_models/svm.joblib' in current directory.
            window_sec: Analysis window duration in seconds (default: 10.0s).
            step_sec: Step / interval size in seconds for sliding analysis (default: 2.0s).
        """
        if model_path is None:
            # Candidate default paths
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(cur_dir, "svm.joblib"),
                os.path.join(cur_dir, "sqi_models", "svm.joblib"),
                r"C:\Users\Artin\Desktop\New folder\sqi_models\svm.joblib",
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not locate svm.joblib model artifact. Searched: {model_path}")

        self.model_path = model_path
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.colormap = get_noise_colormap()

        # Load model artifact
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.label_encoder = artifact["label_encoder"]
        self.feature_names = artifact["feature_names"]
        self.classes = self.label_encoder.classes_

        self.idx_good = np.where(self.classes == "good")[0][0] if "good" in self.classes else 0
        self.idx_med = np.where(self.classes == "medium")[0][0] if "medium" in self.classes else 1
        self.idx_poor = np.where(self.classes == "poor")[0][0] if "poor" in self.classes else 2

    def _compute_window_features(self, window: np.ndarray, fs_int: int) -> List[float]:
        """Calculates the 19 SQI features for a single window."""
        row = []
        for name in self.feature_names:
            fn, mode = _SQI_FUNCTIONS[name]
            try:
                if mode == "signal":
                    val = fn(window)
                elif mode == "signal_fs":
                    val = fn(window, fs_int)
                else:
                    val = fn(window, fs_int, [3, 30])
                val = np.asarray(val, dtype=np.float64).reshape(-1)
                val_f = float(np.nanmean(val)) if val.size else np.nan
            except Exception:
                val_f = np.nan
            row.append(val_f)
        return row

    def get_sample_colors(
        self,
        ecg_signal: np.ndarray,
        fs: float = 180.0,
        return_hex: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Union[np.ndarray, List[str]], np.ndarray]:
        """
        Computes the continuous noise severity and color for EVERY sample of the signal.

        Args:
            ecg_signal: 1D array of ECG signal samples.
            fs: Sampling frequency in Hz.
            return_hex: If True, returns color strings ('#0055D4').
                        If False (default), returns RGBA float array of shape (N, 4) in [0.0, 1.0].
            progress_callback: Optional callback(current, total) for progress reporting.

        Returns:
            Tuple of:
            - sample_colors: Array of RGBA values (N, 4) or list of hex strings.
            - sample_scores: Continuous noise severity scores S[n] in [0.0, 1.0] for each sample.
        """
        sig = np.asarray(ecg_signal, dtype=np.float64).flatten()
        if len(sig) == 0:
            raise ValueError("Input ECG signal is empty.")

        fs_int = int(round(fs))
        win_samples = int(round(self.window_sec * fs_int))
        step_samples = int(round(self.step_sec * fs_int))

        total_samples = len(sig)
        start_indices = list(range(0, total_samples - win_samples + 1, step_samples))
        if not start_indices:
            start_indices = [0]
            win_samples = min(win_samples, total_samples)

        feature_rows = []
        valid_starts = []
        tot = len(start_indices)

        for i, start in enumerate(start_indices):
            window = sig[start : start + win_samples]
            row = self._compute_window_features(window, fs_int)
            if all(np.isfinite(v) for v in row):
                feature_rows.append(row)
                valid_starts.append(start)
            else:
                feature_rows.append([0.0] * len(self.feature_names))
                valid_starts.append(start)

            if progress_callback and (i % 250 == 0 or i == tot - 1):
                progress_callback(i + 1, tot)

        features = np.asarray(feature_rows, dtype=np.float64)
        scaled = self.scaler.transform(features)

        # Calibrated decision values via softmax to obtain probabilities
        if hasattr(self.model, "decision_function"):
            df = self.model.decision_function(scaled)
            probs = softmax(df, axis=1)
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled)
        else:
            preds = self.model.predict(scaled)
            probs = np.zeros((len(preds), len(self.classes)))
            for j, p in enumerate(preds):
                probs[j, int(p)] = 1.0

        # Continuous Noise Score: 0.0 * P(good) + 0.5 * P(medium) + 1.0 * P(poor)
        window_scores = (
            0.0 * probs[:, self.idx_good]
            + 0.5 * probs[:, self.idx_med]
            + 1.0 * probs[:, self.idx_poor]
        )

        window_centers = np.array([(s + win_samples / 2.0) for s in valid_starts])
        sample_indices = np.arange(total_samples, dtype=float)

        if len(window_centers) > 1:
            interp_fn = interp1d(
                window_centers,
                window_scores,
                kind="linear",
                bounds_error=False,
                fill_value=(window_scores[0], window_scores[-1]),
            )
            sample_scores = interp_fn(sample_indices)
        else:
            sample_scores = np.full(total_samples, window_scores[0] if len(window_scores) else 0.0)

        # Clip scores to [0.0, 1.0]
        sample_scores = np.clip(sample_scores, 0.0, 1.0)

        # Map scores to continuous RGBA colors
        rgba_colors = self.colormap(sample_scores)  # Shape: (N, 4)

        if return_hex:
            hex_colors = [to_hex(c) for c in rgba_colors]
            return hex_colors, sample_scores

        return rgba_colors, sample_scores

    # =========================================================================
    # Live Waveform Visualizer ("Plot it Alive")
    # =========================================================================

    def plot_alive(
        self,
        ecg_signal: np.ndarray,
        fs: float = 180.0,
        window_width_sec: float = 8.0,
        speed: float = 2.0,
        start_sec: float = 0.0,
        detected_r_peaks: Optional[np.ndarray] = None,
        labeled_r_peaks: Optional[np.ndarray] = None,
        abnormal_flags: Optional[np.ndarray] = None,
        abnormal_probs: Optional[np.ndarray] = None,
        time_offset_sec: float = 0.0,
        max_duration_sec: Optional[float] = None,
        save_animation: Optional[str] = None,
    ) -> None:
        """
        Interactive live telemetry monitor that streams the ECG waveform across
        the screen in real-time or accelerated time, dynamically painting every
        sample with its exact continuous noise severity color (Blue -> Light Red -> Black-Red),
        with overlays for detected R-peaks (Purple), labeled reference peaks (Green),
        and real-time detection of Abnormal QRS complexes (Vivid Red).

        Controls:
        - Spacebar    : Pause / Resume playback (replays from start if at end)
        - Up / Down   : Increase / Decrease speed (+0.5x / -0.5x)
        - Left / Right: Step backward / Step forward by 1 second (scrub while paused)
        - R / Home    : Rewind to start time and resume
        - Esc / Q     : Close window

        Args:
            ecg_signal: 1D ECG array.
            fs: Sampling rate in Hz.
            window_width_sec: Visible duration of the scrolling viewport (default: 8.0s).
            speed: Playback speed factor (e.g. 1.0 = real-time, 2.0 = 2x, etc.).
            start_sec: Starting timestamp in seconds where playback begins.
            detected_r_peaks: Array of detected R-peak sample indices (plotted in Purple).
            labeled_r_peaks: Array of ground-truth / labeled peak sample indices (plotted in Green).
            abnormal_flags: Optional binary array (0=Normal, 1=Abnormal) corresponding to detected_r_peaks.
            abnormal_probs: Optional array of abnormal class probabilities.
            time_offset_sec: Global time offset in seconds for absolute time labeling.
            max_duration_sec: Optional maximum duration to inspect.
            save_animation: Optional filename (.gif) to save playback recording.
        """
        sig = np.asarray(ecg_signal, dtype=np.float64).flatten()
        det_peaks = (
            np.asarray(detected_r_peaks, dtype=np.int64).flatten()
            if detected_r_peaks is not None
            else np.array([], dtype=np.int64)
        )
        lab_peaks = (
            np.asarray(labeled_r_peaks, dtype=np.int64).flatten()
            if labeled_r_peaks is not None
            else np.array([], dtype=np.int64)
        )
        abn_flags = (
            np.asarray(abnormal_flags, dtype=np.int32).flatten()
            if abnormal_flags is not None
            else np.zeros(len(det_peaks), dtype=np.int32)
        )
        abn_peaks = det_peaks[abn_flags == 1] if len(abn_flags) == len(det_peaks) else np.array([], dtype=np.int64)

        if max_duration_sec is not None:
            max_samples = int(max_duration_sec * fs)
            sig = sig[:max_samples]
            valid_mask = det_peaks < max_samples
            det_peaks = det_peaks[valid_mask]
            if len(abn_flags) > len(det_peaks):
                abn_flags = abn_flags[valid_mask]
            abn_peaks = det_peaks[abn_flags == 1] if len(abn_flags) == len(det_peaks) else np.array([], dtype=np.int64)
            lab_peaks = lab_peaks[lab_peaks < max_samples]

        print(f"Computing sample-level SQI colors for {len(sig):,} samples...")
        t0 = time.time()
        rgba_colors, sample_scores = self.get_sample_colors(sig, fs=fs)
        print(f"Computed in {time.time()-t0:.2f}s. Initializing live playback...")

        time_axis = time_offset_sec + (np.arange(len(sig)) / fs)
        window_samples = int(window_width_sec * fs)

        # Initial playback cursor
        start_sec_clamped = max(0.0, float(start_sec))
        initial_idx = int(start_sec_clamped * fs)
        initial_idx = max(0, min(initial_idx, len(sig) - 1))

        # Configure figure
        fig, (ax_main, ax_gauge) = plt.subplots(
            2, 1,
            figsize=(14, 7.5),
            gridspec_kw={"height_ratios": [4.2, 0.8], "hspace": 0.28},
            facecolor="#12151C"
        )
        for ax in (ax_main, ax_gauge):
            ax.set_facecolor("#181C24")

        # Styling
        y_min, y_max = np.percentile(sig, 0.5), np.percentile(sig, 99.5)
        margin = max(50.0, 0.25 * (y_max - y_min))
        ax_main.set_ylim(y_min - margin, y_max + margin)

        init_time = time_axis[initial_idx]
        init_x_min = max(time_axis[0] + start_sec_clamped, init_time - window_width_sec)
        init_x_max = max(time_axis[0] + start_sec_clamped + window_width_sec, init_time)
        ax_main.set_xlim(init_x_min, init_x_max)

        ax_main.grid(True, color="#2A303C", linestyle="--", alpha=0.7)
        ax_main.set_ylabel("ECG Amplitude (ADC)", color="#E0E6ED", fontsize=11, fontweight="bold")
        ax_main.set_xlabel("Time (seconds)", color="#E0E6ED", fontsize=10)
        ax_main.tick_params(colors="#A0AEC0", labelsize=10)

        # Header Badges
        status_badge = ax_main.text(
            0.02, 0.92, "LIVE STREAM: READY",
            transform=ax_main.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color="#FFFFFF",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0055D4", alpha=0.9, edgecolor="#3388EE"),
            zorder=10,
        )

        speed_text = ax_main.text(
            0.98, 0.92, f"SPEED: {speed:.1f}x [SPACE: Pause | ←/→: Step | R: Rewind]",
            transform=ax_main.transAxes,
            fontsize=9.5,
            ha="right",
            fontweight="bold",
            color="#A0AEC0",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#181C24", alpha=0.85, edgecolor="#2A303C"),
            zorder=10,
        )

        # Gauge Subplot (instant noise meter)
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(0, 1)
        ax_gauge.set_yticks([])
        ax_gauge.set_xticks([0.0, 0.35, 0.70, 1.0])
        ax_gauge.set_xticklabels(
            ["0.0\n(Clean Blue)", "0.35\n(Mild)", "0.70\n(Moderate)", "1.0\n(Black-Red)"],
            color="#A0AEC0",
            fontsize=9
        )
        ax_gauge.set_title("Instantaneous Noise Severity Meter S(t)", color="#E0E6ED", fontsize=10, pad=4)
        ax_gauge.tick_params(colors="#A0AEC0")

        # Colorbar reference bar under gauge
        grad = np.linspace(0, 1, 300).reshape(1, -1)
        ax_gauge.imshow(grad, extent=[0, 1, 0, 1], cmap=self.colormap, aspect="auto")
        gauge_needle = ax_gauge.axvline(0.0, color="#FFFFFF", linewidth=3.0)

        # Initialize LineCollection for ECG waveform
        norm = Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection([], cmap=self.colormap, norm=norm, linewidth=1.8, zorder=4)
        ax_main.add_collection(lc)

        # Initialize Detected R-Peaks Scatter (PURPLE)
        scatter_det = ax_main.scatter(
            [], [],
            color="#9B51E0",
            s=55,
            zorder=6,
            label=f"Detected R-Peaks (Purple, N={len(det_peaks)})" if len(det_peaks) else "Detected R-Peaks (Purple)",
            edgecolors="#FFFFFF",
            linewidths=0.9,
        )

        # Initialize Labeled Reference Peaks Scatter (GREEN DIAMOND)
        scatter_ref = ax_main.scatter(
            [], [],
            color="#00E676",
            marker="D",
            s=60,
            zorder=7,
            label=f"Labeled Reference Peaks (Green, N={len(lab_peaks)})" if len(lab_peaks) else "Labeled Reference Peaks (Green)",
            edgecolors="#003311",
            linewidths=0.9,
        )

        # Initialize Abnormal QRS Beats Scatter (VIVID RED DIAMOND)
        scatter_abn = ax_main.scatter(
            [], [],
            color="#FF0055",
            marker="D",
            s=85,
            zorder=8,
            label=f"Abnormal QRS Beats (Red, N={len(abn_peaks)})" if len(abn_peaks) else "Abnormal QRS Beats (Red)",
            edgecolors="#FFFFFF",
            linewidths=1.2,
        )

        # Show legend if any peak annotations are available
        if len(det_peaks) > 0 or len(lab_peaks) > 0 or len(abn_peaks) > 0:
            ax_main.legend(
                loc="upper left",
                bbox_to_anchor=(0.02, 0.84),
                framealpha=0.88,
                facecolor="#181C24",
                edgecolor="#2A303C",
                fontsize=8.5,
                labelcolor="#E0E6ED",
            )

        # State tracking
        state = {
            "is_paused": False,
            "speed": speed,
            "cur_idx": initial_idx,
            "initial_idx": initial_idx,
            "step_per_frame": int(max(1, (fs / 30.0) * speed)),
            "reached_end": False,
        }

        def on_key(event):
            if event.key == " ":
                if state["reached_end"]:
                    # Replay from requested start time
                    state["cur_idx"] = state["initial_idx"]
                    state["reached_end"] = False
                    state["is_paused"] = False
                    status_badge.set_text("STATUS: RESUMING PLAYBACK...")
                else:
                    state["is_paused"] = not state["is_paused"]
                p_str = "PAUSED" if state["is_paused"] else "PLAYING"
                speed_text.set_text(f"SPEED: {state['speed']:.1f}x [{p_str} | ←/→: Step | R: Rewind]")
            elif event.key == "up":
                state["speed"] = min(10.0, state["speed"] + 0.5)
                state["step_per_frame"] = int(max(1, (fs / 30.0) * state["speed"]))
                p_str = "PAUSED" if state["is_paused"] else "PLAYING"
                speed_text.set_text(f"SPEED: {state['speed']:.1f}x [{p_str} | ←/→: Step | R: Rewind]")
            elif event.key == "down":
                state["speed"] = max(0.5, state["speed"] - 0.5)
                state["step_per_frame"] = int(max(1, (fs / 30.0) * state["speed"]))
                p_str = "PAUSED" if state["is_paused"] else "PLAYING"
                speed_text.set_text(f"SPEED: {state['speed']:.1f}x [{p_str} | ←/→: Step | R: Rewind]")
            elif event.key == "left":
                # Step backward by 1.0 second (allows scrubbing while paused)
                step_samples = int(fs * 1.0)
                state["cur_idx"] = max(0, state["cur_idx"] - step_samples)
                state["reached_end"] = False
                speed_text.set_text(f"SPEED: {state['speed']:.1f}x [STEPPED BACK 1s]")
            elif event.key == "right":
                # Step forward by 1.0 second
                step_samples = int(fs * 1.0)
                state["cur_idx"] = min(len(sig), state["cur_idx"] + step_samples)
                speed_text.set_text(f"SPEED: {state['speed']:.1f}x [STEPPED FWD 1s]")
            elif event.key in ("r", "R", "home"):
                # Rewind to start_sec
                state["cur_idx"] = state["initial_idx"]
                state["reached_end"] = False
                state["is_paused"] = False
                speed_text.set_text(f"SPEED: {state['speed']:.1f}x [REWOUND TO START]")
            elif event.key in ("q", "escape"):
                plt.close(fig)
                return
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)

        def update(frame):
            if not state["is_paused"]:
                state["cur_idx"] += state["step_per_frame"]
                if state["cur_idx"] >= len(sig):
                    # Continue rendering right up to the final sample and hold view
                    state["cur_idx"] = len(sig)
                    state["is_paused"] = True
                    state["reached_end"] = True

            end_idx = min(len(sig), state["cur_idx"])
            start_idx = max(0, end_idx - window_samples)

            if end_idx <= 1:
                end_idx = min(len(sig), 2)
                start_idx = 0

            # Sub-segment data
            t_sub = time_axis[start_idx:end_idx]
            sig_sub = sig[start_idx:end_idx]
            scores_sub = sample_scores[start_idx:end_idx]

            if len(t_sub) > 1:
                # Build LineCollection segments
                points = np.array([t_sub, sig_sub]).T.reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                lc.set_segments(segs)
                lc.set_array(scores_sub[:-1])

                # Update X limits (scrolling smoothly from start_sec to the end)
                cur_time = t_sub[-1]
                t_origin = time_axis[0] + start_sec_clamped
                x_min = max(t_origin, cur_time - window_width_sec)
                x_max = max(t_origin + window_width_sec, cur_time)
                if x_max - x_min < window_width_sec:
                    x_max = x_min + window_width_sec
                ax_main.set_xlim(x_min, x_max)

                # Update Detected R-peaks (Purple) in visible viewport
                if len(det_peaks) > 0:
                    det_mask = (det_peaks >= start_idx) & (det_peaks < end_idx)
                    v_det = det_peaks[det_mask]
                    if len(v_det) > 0:
                        scatter_det.set_offsets(np.c_[time_axis[v_det], sig[v_det]])
                    else:
                        scatter_det.set_offsets(np.empty((0, 2)))

                # Update Labeled Reference Peaks (Green Diamond) in visible viewport
                if len(lab_peaks) > 0:
                    lab_mask = (lab_peaks >= start_idx) & (lab_peaks < end_idx)
                    v_lab = lab_peaks[lab_mask]
                    if len(v_lab) > 0:
                        scatter_ref.set_offsets(np.c_[time_axis[v_lab], sig[v_lab]])
                    else:
                        scatter_ref.set_offsets(np.empty((0, 2)))

                # Update Abnormal QRS Beats (Vivid Red Diamond) in visible viewport
                has_abnormal_in_view = False
                latest_abn_time = 0.0
                if len(abn_peaks) > 0:
                    abn_mask = (abn_peaks >= start_idx) & (abn_peaks < end_idx)
                    v_abn = abn_peaks[abn_mask]
                    if len(v_abn) > 0:
                        scatter_abn.set_offsets(np.c_[time_axis[v_abn], sig[v_abn]])
                        # Check if abnormal beat is within recent 1.5 seconds
                        recent_mask = (time_axis[v_abn] >= cur_time - 1.5) & (time_axis[v_abn] <= cur_time + 0.5)
                        if np.any(recent_mask):
                            has_abnormal_in_view = True
                            latest_abn_time = float(time_axis[v_abn][recent_mask][-1])
                    else:
                        scatter_abn.set_offsets(np.empty((0, 2)))


                # Update live badge & meter
                cur_score = float(scores_sub[-1])
                gauge_needle.set_xdata([cur_score, cur_score])

                if state["reached_end"]:
                    status_badge.set_text("STATUS: END OF SIGNAL REACHED (100%) | [SPACE / R: Replay | ←: Step Back]")
                    status_badge.set_bbox(dict(boxstyle="round,pad=0.35", facecolor="#1F2937", alpha=0.95, edgecolor="#10B981"))
                elif state["is_paused"]:
                    status_badge.set_text(f"PAUSED AT {cur_time:.1f}s | S(t): {cur_score:.2f} [SPACE: Resume | ←/→: Step]")
                    status_badge.set_bbox(dict(boxstyle="round,pad=0.35", facecolor="#374151", alpha=0.9, edgecolor="#9CA3AF"))
                elif has_abnormal_in_view:
                    status_badge.set_text(f"ALERT: ABNORMAL BEAT DETECTED at t={latest_abn_time:.1f}s! [RED]")
                    status_badge.set_bbox(dict(boxstyle="round,pad=0.35", facecolor="#B91C1C", alpha=0.95, edgecolor="#EF4444"))
                elif cur_score < 0.35:
                    status_badge.set_text(f"QUALITY: CLEAN [BLUE]  |  S(t): {cur_score:.2f}  |  t={cur_time:.1f}s")
                    status_badge.set_bbox(dict(boxstyle="round,pad=0.35", facecolor="#0055D4", alpha=0.9, edgecolor="#3388EE"))
                elif cur_score < 0.70:
                    status_badge.set_text(f"QUALITY: LITTLE NOISE [LIGHT RED]  |  S(t): {cur_score:.2f}  |  t={cur_time:.1f}s")
                    status_badge.set_bbox(dict(boxstyle="round,pad=0.35", facecolor="#CC4444", alpha=0.9, edgecolor="#FFA0A0"))
                else:
                    status_badge.set_text(f"QUALITY: SEVERE NOISE [BLACK-RED]  |  S(t): {cur_score:.2f}  |  t={cur_time:.1f}s")
                    status_badge.set_bbox(dict(boxstyle="round,pad=0.35", facecolor="#2B0000", alpha=0.95, edgecolor="#8B0000"))

            return lc, scatter_det, scatter_ref, scatter_abn, status_badge, speed_text, gauge_needle

        ani = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)

        if save_animation:
            print(f"Saving animation to: {save_animation}...")
            ani.save(save_animation, writer="pillow", fps=25)
            print("Animation saved successfully!")
        else:
            plt.show()
        plt.close(fig)


# =============================================================================
    # Helper to load annotations from file
# =============================================================================

def load_peaks_file(path: str) -> np.ndarray:
    """Parses peaks from .txt (Mortara), .csv, or .npy file."""
    if not os.path.exists(path):
        return np.array([], dtype=np.int64)

    if path.endswith(".npy"):
        return np.load(path).astype(np.int64)

    # Check for Mortara text format
    samples = []
    in_block = False
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "Beat    Beat     Beat" in line or "Beat    Beat    Beat" in line:
                in_block = True
                continue
            if in_block:
                if "Beat type" in line or "******" in line:
                    continue
                import re
                m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s*$", line)
                if m:
                    samples.append(int(m.group(2)))
            else:
                # Try simple number per line
                parts = line.strip().split(",")
                for p in parts:
                    try:
                        samples.append(int(float(p)))
                    except ValueError:
                        pass
    return np.array(samples, dtype=np.int64)


# =============================================================================
# Command-Line Interface (CLI)
# =============================================================================

def main() -> None:
    """CLI Runner for SQIDetector."""
    parser = argparse.ArgumentParser(
        description="SQIDetector: Continuous Noise Classification & Live Waveform Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default=r"C:\Users\Artin\Desktop\New folder\Pats\pat2\DataII.dat",
        help="Path to ECG recording file (.dat, .csv, .npy)"
    )
    parser.add_argument("--ann", "-a", type=str, default=r"C:\Users\Artin\Desktop\New folder\Pats\pat2\pat2.txt", help="Path to labeled peaks file (.txt, .npy, .csv)")
    parser.add_argument("--fs", "-f", type=float, default=180.0, help="Sampling rate in Hz")
    parser.add_argument("--model", "-m", type=str, default="svm.joblib", help="Path to SVM model")
    parser.add_argument("--window", "-w", type=float, default=10.0, help="Window size in seconds")
    parser.add_argument("--step", "-s", type=float, default=2.0, help="Sliding step in seconds")
    parser.add_argument("--duration", "-d", type=float, default=120.0, help="Duration in seconds to load (default: 120s)")
    parser.add_argument("--start", "--start-time", type=float, default=0.0, help="Start time in seconds for live playback")
    parser.add_argument("--detect-r", action="store_true", help="Detect R-peaks with RPeakDetector and plot in purple")
    parser.add_argument("--live", action="store_true", help="Plot it alive (streaming interactive player)")
    parser.add_argument("--speed", type=float, default=2.0, help="Live playback speed factor (e.g. 1.0, 2.0, 5.0)")
    parser.add_argument("--save-colors", type=str, default=None, help="Save sample colors to a CSV / NPY file")
    parser.add_argument("--demo", action="store_true", help="Run self-contained live demo")

    args = parser.parse_args()

    print("=" * 65)
    print("           SQIDetector Continuous Noise Engine")
    print("=" * 65)

    detector = SQIDetector(model_path=args.model, window_sec=args.window, step_sec=args.step)

    # Load data
    if args.demo or not os.path.exists(args.input):
        print("Running Demo on synthesized Holter segment...")
        t = np.arange(int(args.fs * 60)) / args.fs
        # Simulated clean ECG + noise burst
        clean = 400.0 + 80.0 * np.sin(2 * np.pi * 1.2 * t)
        noise = np.zeros_like(clean)
        # Add noise between 20s and 35s
        mask_med = (t >= 20.0) & (t < 30.0)
        noise[mask_med] += np.random.randn(np.sum(mask_med)) * 40.0
        mask_poor = (t >= 30.0) & (t < 40.0)
        noise[mask_poor] += np.random.randn(np.sum(mask_poor)) * 180.0 + 300.0 * np.sin(2 * np.pi * 0.3 * t[mask_poor])
        sig = clean + noise
        labeled_peaks = np.array([], dtype=np.int64)
    else:
        print(f"Loading ECG from: {args.input}...")
        count = int(args.duration * args.fs) if args.duration else -1
        sig = np.fromfile(args.input, dtype="<i2", count=count).astype(np.float64)
        sig = sig * -1.0  # invert polarity for standard lead II Holter
        labeled_peaks = np.array([], dtype=np.int64)
        if args.ann and os.path.exists(args.ann):
            all_ann = load_peaks_file(args.ann)
            labeled_peaks = all_ann[all_ann < len(sig)]
            print(f"Loaded {len(labeled_peaks):,} labeled reference peaks.")

    print(f"Loaded {len(sig):,} samples ({len(sig) / args.fs:.1f} seconds).")

    # Detect R-peaks if requested or live
    detected_peaks = np.array([], dtype=np.int64)
    if args.detect_r or args.live:
        try:
            from r_peak_detector import RPeakDetector
            r_det = RPeakDetector(sampling_rate=args.fs)
            res = r_det.detect(sig)
            detected_peaks = res.r_peaks
            print(f"Detected {len(detected_peaks):,} R-peaks (Purple).")
        except Exception as e:
            print(f"R-peak detection skipped: {e}")

    # Compute sample-level colors
    colors, scores = detector.get_sample_colors(sig, fs=args.fs)
    print(f"\nSample Color Output Shape: {colors.shape} (N_samples, RGBA)")
    print(f"Sample Noise Scores Shape: {scores.shape} (N_samples,)")
    print(f"Average Noise Severity   : {np.mean(scores):.3f}")

    if args.save_colors:
        if args.save_colors.endswith(".npy"):
            np.save(args.save_colors, colors)
        else:
            np.savetxt(args.save_colors, colors, delimiter=",", header="R,G,B,A", comments="")
        print(f"Saved sample colors to: {args.save_colors}")

    # Plot it alive
    if args.live or args.demo or len(sys.argv) == 1:
        detector.plot_alive(
            sig,
            fs=args.fs,
            speed=args.speed,
            start_sec=args.start,
            detected_r_peaks=detected_peaks,
            labeled_r_peaks=labeled_peaks,
            window_width_sec=8.0,
        )


if __name__ == "__main__":
    main()
