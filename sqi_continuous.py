"""
Continuous Signal Quality Index (SQI) Engine - Phase 2
======================================================
Applies a trained machine learning SQI classifier to an ECG signal using a rolling,
sliding-window approach to achieve continuous (non-discrete) noise assessment.

Color Scheme:
- Clean (Good)           -> Blue (#0055D4)
- Little Noise (Medium)  -> Light Red / Coral (#FF7F7F)
- Very Much Noise (Poor) -> Dark Red to Black-Red (#3A0000)

Features:
- Sub-window sliding analysis (e.g., 10s window with fine interval steps).
- Continuous noise severity index: S(t) in [0.0, 1.0] derived from model decision values / softmax probabilities.
- Sample-accurate interpolation so every sample in the waveform has an exact color.
- Publication-quality multi-panel visualization with continuous LineCollection rendering.
- Macro (Holter timeline overview) and Micro (zoomed multi-strip waveform inspection) plots.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.special import softmax

# Add parent directory of SQI_module to sys.path if needed
NEW_FOLDER_PATH = r"C:\Users\Artin\Desktop\New folder"
if NEW_FOLDER_PATH not in sys.path:
    sys.path.append(NEW_FOLDER_PATH)

try:
    from detect_noise import FEATURE_FUNCTIONS, _compute_feature_row
    _HAS_FEATURE_FUNCS = True
except ImportError:
    _HAS_FEATURE_FUNCS = False


def create_ecg_noise_colormap() -> LinearSegmentedColormap:
    """
    Creates the custom continuous colormap:
    - 0.00: Deep Blue (clean)
    - 0.25: Muted Blue / Transition
    - 0.50: Light Red / Coral (little noise)
    - 0.75: Vivid Crimson
    - 1.00: Dark Black-Red (very much noise)
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


class ContinuousSQIEngine:
    """
    Continuous SQI Engine for evaluating and visualizing noise in ECG recordings.
    """

    def __init__(
        self,
        model_path: str = r"C:\Users\Artin\Desktop\New folder\sqi_models\svm.joblib",
        window_sec: float = 10.0,
        step_sec: float = 2.0,
    ) -> None:
        """
        Initialize the Continuous SQI Engine.

        Args:
            model_path: Path to the trained SQI model (.joblib).
            window_sec: Window length in seconds (default: 10.0s).
            step_sec: Step / interval size in seconds for sliding analysis (default: 2.0s).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")

        self.model_path = model_path
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)

        # Load model artifact
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.label_encoder = artifact["label_encoder"]
        self.feature_names = artifact["feature_names"]
        self.classes = self.label_encoder.classes_

        # Verify class indices
        self.idx_good = np.where(self.classes == "good")[0][0] if "good" in self.classes else 0
        self.idx_med = np.where(self.classes == "medium")[0][0] if "medium" in self.classes else 1
        self.idx_poor = np.where(self.classes == "poor")[0][0] if "poor" in self.classes else 2

    def analyze_signal(
        self,
        ecg_signal: np.ndarray,
        fs: float = 180.0,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Runs sliding-window continuous noise estimation across the ECG signal.

        Args:
            ecg_signal: 1D array of ECG signal samples.
            fs: Sampling rate in Hz.
            progress_callback: Optional callback(current, total) for reporting progress.

        Returns:
            Dictionary containing:
            - 'window_times': Center times of each window (seconds)
            - 'noise_scores': Continuous noise score in [0.0, 1.0] for each window
            - 'probabilities': (N_windows, 3) matrix of [P(good), P(medium), P(poor)]
            - 'sample_noise_scores': Interpolated noise score for every sample in ecg_signal
            - 'quality_stats': Percentage breakdown of clean vs medium vs poor
        """
        sig = np.asarray(ecg_signal, dtype=np.float64).flatten()
        fs_int = int(round(fs))
        window_samples = int(round(self.window_sec * fs_int))
        step_samples = int(round(self.step_sec * fs_int))

        if window_samples <= 0 or step_samples <= 0:
            raise ValueError("window_sec and step_sec must be positive.")

        total_samples = len(sig)
        start_indices = list(range(0, total_samples - window_samples + 1, step_samples))

        if not start_indices:
            # Signal shorter than one window
            start_indices = [0]
            window_samples = min(window_samples, total_samples)

        feature_rows = []
        valid_starts = []
        total_windows = len(start_indices)

        for i, start in enumerate(start_indices):
            window = sig[start : start + window_samples]
            row = _compute_feature_row(window, fs_int, self.feature_names)
            # Check for non-finite values
            if all(np.isfinite(v) for v in row):
                feature_rows.append(row)
                valid_starts.append(start)
            else:
                # Replace non-finite with high noise default (poor)
                feature_rows.append([0.0] * len(self.feature_names))
                valid_starts.append(start)

            if progress_callback and (i % 250 == 0 or i == total_windows - 1):
                progress_callback(i + 1, total_windows)

        features = np.asarray(feature_rows, dtype=np.float64)
        scaled_features = self.scaler.transform(features)

        # Compute continuous probabilities
        if hasattr(self.model, "decision_function"):
            decision_values = self.model.decision_function(scaled_features)
            probs = softmax(decision_values, axis=1)
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled_features)
        else:
            preds = self.model.predict(scaled_features)
            probs = np.zeros((len(preds), len(self.classes)))
            for j, p in enumerate(preds):
                probs[j, int(p)] = 1.0

        # Continuous Noise Severity Score: 0.0 * P(good) + 0.5 * P(medium) + 1.0 * P(poor)
        noise_scores = (
            0.0 * probs[:, self.idx_good]
            + 0.5 * probs[:, self.idx_med]
            + 1.0 * probs[:, self.idx_poor]
        )

        window_centers_s = np.array([
            (s + window_samples / 2.0) / fs for s in valid_starts
        ])

        # Interpolate continuous noise score to every sample
        time_axis = np.arange(total_samples) / fs
        if len(window_centers_s) > 1:
            interp_fn = interp1d(
                window_centers_s,
                noise_scores,
                kind="linear",
                bounds_error=False,
                fill_value=(noise_scores[0], noise_scores[-1]),
            )
            sample_noise_scores = interp_fn(time_axis)
        else:
            sample_noise_scores = np.full(total_samples, noise_scores[0] if len(noise_scores) else 0.0)

        # Quality Statistics
        pct_clean = float(np.mean(noise_scores < 0.35) * 100.0)
        pct_medium = float(np.mean((noise_scores >= 0.35) & (noise_scores < 0.70)) * 100.0)
        pct_poor = float(np.mean(noise_scores >= 0.70) * 100.0)

        return {
            "window_times": window_centers_s,
            "noise_scores": noise_scores,
            "probabilities": probs,
            "sample_noise_scores": sample_noise_scores,
            "time_axis": time_axis,
            "quality_stats": {
                "pct_clean": round(pct_clean, 2),
                "pct_medium": round(pct_medium, 2),
                "pct_poor": round(pct_poor, 2),
            },
        }

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_continuous_strip(
        self,
        ax: plt.Axes,
        time_segment: np.ndarray,
        signal_segment: np.ndarray,
        noise_segment: np.ndarray,
        cmap: LinearSegmentedColormap,
        title: str = "",
        ylabel: str = "ECG (ADC)",
    ) -> None:
        """
        Renders an ECG signal strip where each segment of the line is continuously
        colored according to the local noise severity value.
        """
        points = np.array([time_segment, signal_segment]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        # Color array aligns with segment starts
        lc.set_array(noise_segment[:-1])
        lc.set_linewidth(1.3)

        ax.add_collection(lc)
        ax.set_xlim(time_segment[0], time_segment[-1])

        # Dynamic y limits with margin
        p1 = np.percentile(signal_segment, 1)
        p99 = np.percentile(signal_segment, 99)
        margin = max(50.0, 0.2 * (p99 - p1))
        ax.set_ylim(p1 - margin, p99 + margin)

        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)

    def generate_holter_report_plot(
        self,
        ecg_signal: np.ndarray,
        analysis_result: Dict[str, Any],
        fs: float = 180.0,
        patient_name: str = "pat1",
        lead_name: str = "Lead II",
        save_path: str = "holter_continuous_sqi_plot.png",
    ) -> str:
        """
        Generates a comprehensive multi-panel figure displaying:
        - Panel 1: Full-Holter Continuous Noise Severity Timeline with color bands.
        - Panel 2: Clean baseline strip (rendered in deep Blue).
        - Panel 3: Mild/Moderate noise strip (rendered in Light Red / Coral).
        - Panel 4: Severe noise artifact strip (rendered in Dark Black-Red).
        - Panel 5: Continuous transition strip illustrating the live color morphing.
        """
        time_axis = analysis_result["time_axis"]
        sample_noise = analysis_result["sample_noise_scores"]
        window_times = analysis_result["window_times"]
        noise_scores = analysis_result["noise_scores"]
        stats = analysis_result["quality_stats"]

        cmap = create_ecg_noise_colormap()
        total_duration_hours = len(ecg_signal) / (fs * 3600.0)

        fig = plt.figure(figsize=(18, 14), dpi=150)
        gs = fig.add_gridspec(5, 1, height_ratios=[1.8, 1.0, 1.0, 1.0, 1.0], hspace=0.35)

        # ---------------------------------------------------------------------
        # Panel 1: Full-Holter Macro Noise Timeline
        # ---------------------------------------------------------------------
        ax0 = fig.add_subplot(gs[0])
        times_hrs = window_times / 3600.0

        # Continuous colored fill under the noise timeline
        points = np.array([times_hrs, noise_scores]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.0)
        lc.set_array(noise_scores[:-1])
        ax0.add_collection(lc)

        ax0.fill_between(times_hrs, 0, noise_scores, color="#FF9999", alpha=0.25)
        ax0.axhline(0.35, color="#3388EE", linestyle=":", linewidth=1.2, label="Clean/Medium Boundary (0.35)")
        ax0.axhline(0.70, color="#8B0000", linestyle=":", linewidth=1.2, label="Medium/Poor Boundary (0.70)")

        ax0.set_xlim(0, max(0.1, times_hrs[-1]))
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_ylabel("Noise Severity Index S(t)", fontsize=10, fontweight="bold")
        ax0.set_xlabel("Recording Time (Hours)", fontsize=10)
        ax0.set_title(
            f"Continuous SQI Holter Noise Examination: {patient_name} ({lead_name}) | "
            f"Duration: {total_duration_hours:.2f} Hours | Clean: {stats['pct_clean']}% (Blue) | "
            f"Little Noise: {stats['pct_medium']}% (Light Red) | Very Much Noise: {stats['pct_poor']}% (Black-Red)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax0.legend(loc="upper right", framealpha=0.85, fontsize=8)
        ax0.grid(True, linestyle="--", alpha=0.35)

        # Add Colorbar for Panel 1
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.915, 0.76, 0.015, 0.15])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([0.05, 0.50, 0.95])
        cbar.set_ticklabels(["Clean\n(Blue)", "Little Noise\n(Light Red)", "Extreme Noise\n(Black-Red)"])
        cbar.ax.tick_params(labelsize=8)

        # ---------------------------------------------------------------------
        # Helper to find representative strips
        # ---------------------------------------------------------------------
        strip_len_s = 20.0
        strip_samples = int(strip_len_s * fs)

        # 1. Representative Clean Segment (minimum noise score)
        clean_idx = int(np.argmin(noise_scores))
        clean_t0 = max(0.0, window_times[clean_idx] - strip_len_s / 2.0)
        s0 = int(clean_t0 * fs)
        s1 = min(len(ecg_signal), s0 + strip_samples)

        # 2. Representative Moderate Noise Segment (near 0.50)
        med_idx = int(np.argmin(np.abs(noise_scores - 0.50)))
        med_t0 = max(0.0, window_times[med_idx] - strip_len_s / 2.0)
        m0 = int(med_t0 * fs)
        m1 = min(len(ecg_signal), m0 + strip_samples)

        # 3. Representative Severe Noise Segment (near maximum)
        poor_idx = int(np.argmax(noise_scores))
        poor_t0 = max(0.0, window_times[poor_idx] - strip_len_s / 2.0)
        p0 = int(poor_t0 * fs)
        p1 = min(len(ecg_signal), p0 + strip_samples)

        # 4. Continuous Transition Segment (highest gradient of noise change)
        grad = np.abs(np.diff(noise_scores))
        trans_idx = int(np.argmax(grad)) if len(grad) > 0 else clean_idx
        trans_t0 = max(0.0, window_times[trans_idx] - strip_len_s / 2.0)
        t0_s = int(trans_t0 * fs)
        t1_s = min(len(ecg_signal), t0_s + strip_samples)

        # ---------------------------------------------------------------------
        # Panel 2: Clean Waveform Strip (Blue)
        # ---------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[1])
        self.plot_continuous_strip(
            ax1,
            time_axis[s0:s1],
            ecg_signal[s0:s1],
            sample_noise[s0:s1],
            cmap,
            title=f"Sample Strip 1: Clean ECG Waveform (t = {clean_t0:.1f}s - {clean_t0 + strip_len_s:.1f}s) -> Rendered in Clean Blue",
        )

        # ---------------------------------------------------------------------
        # Panel 3: Mild/Moderate Noise Strip (Light Red)
        # ---------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[2])
        self.plot_continuous_strip(
            ax2,
            time_axis[m0:m1],
            ecg_signal[m0:m1],
            sample_noise[m0:m1],
            cmap,
            title=f"Sample Strip 2: Little Noise / Moderate Artifact (t = {med_t0:.1f}s - {med_t0 + strip_len_s:.1f}s) -> Rendered in Light Red",
        )

        # ---------------------------------------------------------------------
        # Panel 4: Severe Noise Strip (Dark Black-Red)
        # ---------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[3])
        self.plot_continuous_strip(
            ax3,
            time_axis[p0:p1],
            ecg_signal[p0:p1],
            sample_noise[p0:p1],
            cmap,
            title=f"Sample Strip 3: Very Much Noise / Motion & Detachment Artifact (t = {poor_t0:.1f}s - {poor_t0 + strip_len_s:.1f}s) -> Rendered in Black-Red",
        )

        # ---------------------------------------------------------------------
        # Panel 5: Continuous Transition Strip
        # ---------------------------------------------------------------------
        ax4 = fig.add_subplot(gs[4])
        self.plot_continuous_strip(
            ax4,
            time_axis[t0_s:t1_s],
            ecg_signal[t0_s:t1_s],
            sample_noise[t0_s:t1_s],
            cmap,
            title=f"Sample Strip 4: Continuous Noise Transition (t = {trans_t0:.1f}s - {trans_t0 + strip_len_s:.1f}s) -> Smooth Morphing: Blue -> Light Red -> Black-Red",
        )
        ax4.set_xlabel("Time (seconds)", fontsize=10)

        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Holter continuous SQI plot saved to: {save_path}")
        return save_path


def run_holter_sqi_cli() -> None:
    """CLI runner for continuous SQI analysis."""
    parser = argparse.ArgumentParser(description="Phase 2 Continuous SQI Engine for ECG Recordings")
    parser.add_argument(
        "--input", "-i", type=str,
        default=r"C:\Users\Artin\Desktop\New folder\Pats\pat1\DataII.dat",
        help="Path to binary raw ECG (.dat) or file"
    )
    parser.add_argument("--fs", "-f", type=float, default=180.0, help="Sampling frequency in Hz")
    parser.add_argument(
        "--model", "-m", type=str,
        default=r"C:\Users\Artin\Desktop\New folder\sqi_models\svm.joblib",
        help="Path to trained SQI model (.joblib)"
    )
    parser.add_argument("--window", "-w", type=float, default=10.0, help="Window size in seconds")
    parser.add_argument("--step", "-s", type=float, default=2.0, help="Step / hop size in seconds for sliding analysis")
    parser.add_argument("--duration-hours", type=float, default=None, help="Duration in hours to process (default: all)")
    parser.add_argument("--output", "-o", type=str, default="holter_continuous_sqi.png", help="Output plot image filename")

    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 2: CONTINUOUS SIGNAL QUALITY INDEX (SQI) ENGINE")
    print("=" * 70)
    print(f"Input Signal : {args.input}")
    print(f"Model Path   : {args.model}")
    print(f"Window Size  : {args.window}s | Step Size: {args.step}s | Sampling Rate: {args.fs} Hz")

    # Read binary ECG file
    sig = np.fromfile(args.input, dtype="<i2").astype(np.float64)
    total_hours = len(sig) / (args.fs * 3600.0)
    print(f"Total Available Signal: {len(sig):,} samples ({total_hours:.2f} hours)")

    if args.duration_hours is not None:
        max_samples = int(args.duration_hours * 3600.0 * args.fs)
        sig = sig[:max_samples]
        print(f"Subsetting to first {args.duration_hours:.2f} hours ({len(sig):,} samples).")

    engine = ContinuousSQIEngine(model_path=args.model, window_sec=args.window, step_sec=args.step)

    t0 = time.time()
    def progress(cur, tot):
        pct = (cur / tot) * 100.0
        sys.stdout.write(f"\rProcessing Windows: {cur}/{tot} ({pct:.1f}%)")
        sys.stdout.flush()

    results = engine.analyze_signal(sig, fs=args.fs, progress_callback=progress)
    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed:.2f} seconds!")

    stats = results["quality_stats"]
    print("\nQuality Breakdown:")
    print(f"  - Clean Parts (Blue)        : {stats['pct_clean']}%")
    print(f"  - Little Noise (Light Red)  : {stats['pct_medium']}%")
    print(f"  - Very Much Noise (Black-Red): {stats['pct_poor']}%")

    out_file = engine.generate_holter_report_plot(
        sig,
        results,
        fs=args.fs,
        patient_name="Patient 1 (pat1)",
        lead_name="Lead II",
        save_path=args.output,
    )
    print(f"\nSUCCESS: Visualization generated at '{out_file}'.")


if __name__ == "__main__":
    run_holter_sqi_cli()
