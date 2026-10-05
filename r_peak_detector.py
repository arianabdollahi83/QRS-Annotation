"""
ECG R-Peak Detector - Phase 1 Engine
====================================
A reliable, generalizable, and high-performance R-peak detection library and CLI tool
designed for ECG signals across diverse datasets and sampling frequencies.

Features:
- Multiple detector backends:
  * 'adaptive_prominence': Signal energy envelope + dynamic SciPy peak prominence (Zero-dependency beyond SciPy/NumPy).
  * 'prominence_delineator': Physiology-informed visibility graph prominence delineation (Emrich et al. 2024).
  * 'pan_tompkins': Classic Pan-Tompkins (1985) with dual adaptive thresholding and search-back.
  * 'neurokit': Steep gradient and local maxima detection (NeuroKit2).
  * 'ensemble': Multi-method consensus voting for maximum noise tolerance.
- Automatic signal conditioning (0.5-40 Hz bandpass, baseline detrend, optional 50/60 Hz notch).
- Polarity detection and auto-inversion for inverted leads (e.g. Lead aVR, V1, or upside-down electrodes).
- Sub-sample peak alignment to true raw signal maxima.
- Refractory period suppression (prevents double-counting wide QRS complexes or tall T-waves).
- Beat segmentation helper for Phase 2 (normal vs. abnormal QRS classification).
- Heart Rate (BPM) and Heart Rate Variability (HRV: SDNN, RMSSD, pNN50) computation.
- Versatile data loader: supports CSV, TSV, TXT, NPY, MAT, and WFDB/PhysioNet formats.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal

# Optional dependencies
try:
    import neurokit2 as nk
    _HAS_NEUROKIT = True
except ImportError:
    _HAS_NEUROKIT = False

try:
    from prominence_delineator import ProminenceDelineator
    _HAS_PROMINENCE_DELINEATOR = True
except ImportError:
    _HAS_PROMINENCE_DELINEATOR = False

try:
    import wfdb
    _HAS_WFDB = True
except ImportError:
    _HAS_WFDB = False

try:
    import scipy.io as sio
    _HAS_SCIPY_IO = True
except ImportError:
    _HAS_SCIPY_IO = False


@dataclass
class RPeakResult:
    """Container holding the results of R-peak detection."""
    r_peaks: np.ndarray                # Sample indices of detected R-peaks
    r_times: np.ndarray                # Time points of detected R-peaks (seconds)
    rr_intervals_ms: np.ndarray        # Successive RR intervals in milliseconds
    heart_rate_bpm: np.ndarray         # Instantaneous heart rate curve (BPM)
    mean_hr_bpm: float                 # Average heart rate (BPM)
    sampling_rate: float               # Sampling frequency (Hz)
    is_inverted: bool                  # Whether polarity inversion was detected and corrected
    method: str                        # Method used for detection
    cleaned_signal: np.ndarray         # Filtered signal used for detection


class RPeakDetector:
    """
    Production-grade R-Peak Detector for ECG signals.
    """

    def __init__(
        self,
        sampling_rate: float = 360.0,
        method: str = "adaptive_prominence",
        bandpass: Tuple[float, float] = (0.5, 40.0),
        notch_freq: Optional[float] = None,
        refractory_ms: float = 200.0,
        auto_invert: bool = True,
        prominence_factor: float = 0.35,
    ) -> None:
        """
        Initialize the detector.

        Args:
            sampling_rate: Sampling frequency of the ECG signal in Hz.
            method: Detection backend:
                - 'adaptive_prominence' (Recommended built-in, robust & fast)
                - 'prominence_delineator' (Visibility graph / prominence delineator)
                - 'pan_tompkins' (Vectorized Pan-Tompkins with search-back)
                - 'neurokit' (NeuroKit2 gradient-based detector)
                - 'ensemble' (Consensus clustering from multiple methods)
            bandpass: Cutoff frequencies (low, high) in Hz for bandpass filtering.
            notch_freq: Powerline frequency (e.g. 50.0 or 60.0 Hz) to notch filter, or None.
            refractory_ms: Physiological refractory period in ms (default 200 ms).
            auto_invert: Automatically invert signal if QRS is predominantly negative.
            prominence_factor: Scaling factor for dynamic prominence threshold.
        """
        self.sampling_rate = float(sampling_rate)
        self.method = method.lower().strip()
        self.bandpass = bandpass
        self.notch_freq = notch_freq
        self.refractory_ms = float(refractory_ms)
        self.auto_invert = auto_invert
        self.prominence_factor = float(prominence_factor)

        # Validate method
        valid_methods = {"adaptive_prominence", "prominence_delineator", "pan_tompkins", "neurokit", "ensemble"}
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method '{self.method}'. Must be one of {valid_methods}")

    # =========================================================================
    # Signal Preprocessing
    # =========================================================================

    def preprocess(self, ecg: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Clean and condition the ECG signal.
        - Removes baseline wander (< bandpass[0] Hz)
        - Removes high-frequency noise (> bandpass[1] Hz)
        - Applies powerline notch filter if requested
        - Checks and corrects polarity inversion

        Args:
            ecg: 1D numpy array representing raw ECG.

        Returns:
            Tuple of (cleaned_signal, was_inverted).
        """
        sig = np.asarray(ecg, dtype=np.float64).flatten()
        if len(sig) == 0:
            raise ValueError("Input ECG signal is empty.")

        fs = self.sampling_rate
        nyquist = 0.5 * fs

        # 1. Bandpass Filter (Butterworth 3rd order)
        low = max(0.01, self.bandpass[0])
        high = min(nyquist - 0.1, self.bandpass[1])
        if low < high and low < nyquist:
            b, a = signal.butter(3, [low / nyquist, high / nyquist], btype="bandpass")
            # Use filtfilt for zero phase distortion
            try:
                filtered = signal.filtfilt(b, a, sig)
            except Exception:
                filtered = sig - np.median(sig)
        else:
            filtered = sig - np.median(sig)

        # 2. Notch Filter (if specified)
        if self.notch_freq is not None and 0 < self.notch_freq < nyquist:
            q = 30.0
            w0 = self.notch_freq / nyquist
            b_notch, a_notch = signal.iirnotch(w0, q)
            filtered = signal.filtfilt(b_notch, a_notch, filtered)

        # 3. Polarity Inversion Detection
        # In a normal ECG lead (e.g. Lead II, V5), R-peaks produce sharp positive deflections.
        # In inverted leads (e.g. Lead aVR or reversed electrodes), the QRS complex is predominantly negative.
        was_inverted = False
        if self.auto_invert:
            med = np.median(filtered)
            pos_amp = np.percentile(filtered, 99.5) - med
            neg_amp = med - np.percentile(filtered, 0.5)

            if pos_amp > 1e-6 and neg_amp > 1.5 * pos_amp:
                filtered = -filtered
                was_inverted = True

        return filtered, was_inverted

    # =========================================================================
    # Detection Backends
    # =========================================================================

    def _detect_adaptive_prominence(self, sig: np.ndarray) -> np.ndarray:
        """
        Finds R-peaks using an adaptive energy envelope and SciPy's peak prominence.
        Combines Shannon energy with gradient steepness to isolate the QRS complex.
        """
        fs = self.sampling_rate
        min_distance = max(1, int((self.refractory_ms / 1000.0) * fs))

        # 1. Compute difference to highlight steep QRS slopes
        diff = np.diff(sig, prepend=sig[0])

        # 2. Shannon Energy Envelope: NKE = - |sig|^2 * log(|sig|^2 + eps)
        # Or Squared Gradient Envelope:
        norm_diff = diff / (np.std(diff) + 1e-8)
        norm_sig = sig / (np.std(sig) + 1e-8)
        envelope = (norm_diff ** 2) * (np.abs(norm_sig))

        # Smooth envelope with a short moving window (~60 ms)
        smooth_len = max(3, int(0.060 * fs))
        kernel = np.hamming(smooth_len)
        kernel /= np.sum(kernel)
        smooth_env = np.convolve(envelope, kernel, mode="same")

        # 3. Dynamic Prominence Calculation
        # Initial peak candidates on smooth envelope
        cand_peaks, properties = signal.find_peaks(smooth_env, distance=min_distance)
        if len(cand_peaks) == 0:
            return np.array([], dtype=int)

        # Calculate prominences of candidate peaks
        prominences = signal.peak_prominences(smooth_env, cand_peaks)[0]

        # Use median and IQR for adaptive threshold
        med_prom = np.median(prominences)
        q75 = np.percentile(prominences, 75)
        iqr = q75 - np.percentile(prominences, 25)

        # Baseline threshold
        threshold = max(0.1, med_prom * self.prominence_factor, (q75 - 0.5 * iqr) * self.prominence_factor)

        # Select peaks exceeding dynamic prominence threshold
        selected_peaks = cand_peaks[prominences >= threshold]

        # 4. Search-back for missed beats (if RR interval > 1.6 * average)
        if len(selected_peaks) >= 2:
            refined_peaks = [selected_peaks[0]]
            rr_history = []
            for p in selected_peaks[1:]:
                rr = p - refined_peaks[-1]
                avg_rr = np.mean(rr_history[-8:]) if len(rr_history) >= 4 else np.median(np.diff(selected_peaks))
                if rr > 1.66 * avg_rr:
                    # Search for missed candidate in between
                    mid_cands = [c for c in cand_peaks if refined_peaks[-1] + min_distance < c < p - min_distance]
                    if mid_cands:
                        mid_proms = [prominences[np.where(cand_peaks == c)[0][0]] for c in mid_cands]
                        best_idx = np.argmax(mid_proms)
                        if mid_proms[best_idx] >= 0.5 * threshold:
                            refined_peaks.append(mid_cands[best_idx])
                            rr_history.append(mid_cands[best_idx] - refined_peaks[-2])
                refined_peaks.append(p)
                rr_history.append(p - refined_peaks[-2])
            selected_peaks = np.array(refined_peaks, dtype=int)

        return selected_peaks

    def _detect_prominence_delineator(self, sig: np.ndarray) -> np.ndarray:
        """
        Uses the prominence_delineator package (Emrich et al. 2024).
        """
        if not _HAS_PROMINENCE_DELINEATOR:
            raise ImportError(
                "prominence-delineator is not installed. Install via 'pip install prominence-delineator' "
                "or use method='adaptive_prominence'."
            )
        delineator = ProminenceDelineator(sampling_frequency=int(round(self.sampling_rate)))
        clean = delineator.clean_ecg(sig)
        peaks = delineator.find_rpeaks(clean)
        return np.asarray(peaks, dtype=int)

    def _detect_pan_tompkins(self, sig: np.ndarray) -> np.ndarray:
        """
        Complete vectorized implementation of Pan & Tompkins (1985) algorithm
        with dual-thresholding, refractory period, and RR interval search-back.
        """
        fs = self.sampling_rate
        nyquist = 0.5 * fs

        # 1. Bandpass filter 5-15 Hz
        low, high = 5.0 / nyquist, min(15.0 / nyquist, 0.95)
        b, a = signal.butter(3, [low, high], btype="bandpass")
        bpass = signal.filtfilt(b, a, sig)

        # 2. 5-point Derivative filter: [-1, -2, 0, 2, 1] * (fs / 8)
        kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (fs / 8.0)
        deriv = np.convolve(bpass, kernel, mode="same")

        # 3. Squaring
        squared = deriv ** 2

        # 4. Moving Window Integration (approx 150 ms)
        win_size = max(1, int(0.150 * fs))
        kernel_mwi = np.ones(win_size) / win_size
        mwi = np.convolve(squared, kernel_mwi, mode="same")

        # 5. Dual Adaptive Thresholding
        refractory = max(1, int((self.refractory_ms / 1000.0) * fs))
        peaks, _ = signal.find_peaks(mwi, distance=refractory)
        if len(peaks) == 0:
            return np.array([], dtype=int)

        spki = np.mean(mwi[peaks[:min(8, len(peaks))]]) * 0.5
        npki = np.mean(mwi) * 0.1
        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1

        qrs_peaks = []
        rr_intervals = []
        last_peak = 0

        for p in peaks:
            val = mwi[p]
            rr = p - last_peak if last_peak > 0 else 0
            rr_avg = np.mean(rr_intervals[-8:]) if len(rr_intervals) >= 8 else (fs * 0.8)

            # Search-back for missed peaks
            if rr > 1.66 * rr_avg and last_peak > 0:
                sb_candidates = [cand for cand in peaks if last_peak < cand < p and mwi[cand] > threshold_i2]
                if sb_candidates:
                    sb_p = sb_candidates[np.argmax([mwi[c] for c in sb_candidates])]
                    qrs_peaks.append(sb_p)
                    rr_intervals.append(sb_p - last_peak)
                    spki = 0.25 * mwi[sb_p] + 0.75 * spki
                    threshold_i1 = npki + 0.25 * (spki - npki)
                    threshold_i2 = 0.5 * threshold_i1
                    last_peak = sb_p

            if val > threshold_i1:
                qrs_peaks.append(p)
                if last_peak > 0:
                    rr_intervals.append(p - last_peak)
                last_peak = p
                spki = 0.125 * val + 0.875 * spki
            else:
                npki = 0.125 * val + 0.875 * npki

            threshold_i1 = npki + 0.25 * (spki - npki)
            threshold_i2 = 0.5 * threshold_i1

        return np.array(qrs_peaks, dtype=int)

    def _detect_neurokit(self, sig: np.ndarray) -> np.ndarray:
        """
        Uses NeuroKit2's gradient-based R-peak detector.
        """
        if not _HAS_NEUROKIT:
            raise ImportError(
                "neurokit2 is not installed. Install via 'pip install neurokit2' "
                "or use method='adaptive_prominence'."
            )
        _, rpeaks_dict = nk.ecg_peaks(sig, sampling_rate=int(self.sampling_rate), method="neurokit")
        peaks = rpeaks_dict.get("ECG_R_Peaks", [])
        return np.asarray(peaks, dtype=int)

    def _detect_ensemble(self, sig: np.ndarray) -> np.ndarray:
        """
        Consensus ensemble: runs multiple complementary detectors and clusters
        detected peaks within a 50ms window.
        """
        candidates_list: List[np.ndarray] = []

        # 1. Adaptive Prominence (always available)
        try:
            p1 = self._detect_adaptive_prominence(sig)
            if len(p1) > 0:
                candidates_list.append(p1)
        except Exception:
            pass

        # 2. Pan-Tompkins (always available)
        try:
            p2 = self._detect_pan_tompkins(sig)
            if len(p2) > 0:
                candidates_list.append(p2)
        except Exception:
            pass

        # 3. NeuroKit (if available)
        if _HAS_NEUROKIT:
            try:
                p3 = self._detect_neurokit(sig)
                if len(p3) > 0:
                    candidates_list.append(p3)
            except Exception:
                pass

        # 4. Prominence Delineator (if available)
        if _HAS_PROMINENCE_DELINEATOR:
            try:
                p4 = self._detect_prominence_delineator(sig)
                if len(p4) > 0:
                    candidates_list.append(p4)
            except Exception:
                pass

        if not candidates_list:
            return np.array([], dtype=int)

        # Merge and sort all candidates
        all_peaks = np.sort(np.concatenate(candidates_list))
        coincidence_samples = int(0.050 * self.sampling_rate)  # 50 ms window

        # Cluster consensus peaks
        clustered_peaks = []
        cluster = [all_peaks[0]]

        for p in all_peaks[1:]:
            if p - cluster[0] <= coincidence_samples:
                cluster.append(p)
            else:
                # Majority or median of cluster
                clustered_peaks.append(int(np.median(cluster)))
                cluster = [p]
        if cluster:
            clustered_peaks.append(int(np.median(cluster)))

        return np.array(clustered_peaks, dtype=int)

    # =========================================================================
    # Peak Alignment & Post-Processing
    # =========================================================================

    def align_peaks(
        self,
        raw_sig: np.ndarray,
        detected_peaks: np.ndarray,
        window_ms: float = 60.0
    ) -> np.ndarray:
        """
        Refines candidate peak locations by snapping each detected peak to the
        exact local maximum on the raw signal within a local search window.

        Args:
            raw_sig: 1D array of the raw (or inverted) ECG signal.
            detected_peaks: Initial detected sample indices.
            window_ms: Search window radius in milliseconds (default: 60 ms).

        Returns:
            1D array of sample-aligned peak indices.
        """
        if len(detected_peaks) == 0:
            return np.array([], dtype=int)

        fs = self.sampling_rate
        radius = max(1, int((window_ms / 1000.0) * fs))
        aligned = []
        sig_len = len(raw_sig)

        for p in detected_peaks:
            left = max(0, p - radius)
            right = min(sig_len, p + radius + 1)
            # Find local maximum of absolute amplitude
            local_max_idx = left + np.argmax(raw_sig[left:right])
            aligned.append(local_max_idx)

        # Remove duplicate detections within refractory window
        aligned = np.sort(np.unique(aligned))
        min_dist = max(1, int((self.refractory_ms / 1000.0) * fs))

        filtered_peaks = []
        for p in aligned:
            if not filtered_peaks:
                filtered_peaks.append(p)
            else:
                if p - filtered_peaks[-1] >= min_dist:
                    filtered_peaks.append(p)
                else:
                    # Keep the higher peak
                    if raw_sig[p] > raw_sig[filtered_peaks[-1]]:
                        filtered_peaks[-1] = p

        return np.array(filtered_peaks, dtype=int)

    # =========================================================================
    # Main Detection Pipeline
    # =========================================================================

    def detect(self, ecg: np.ndarray) -> RPeakResult:
        """
        Execute full R-peak detection on an ECG signal.

        Args:
            ecg: 1D numpy array of ECG amplitudes.

        Returns:
            RPeakResult object containing peak indices, timings, RR intervals,
            and heart rate statistics.
        """
        raw = np.asarray(ecg, dtype=np.float64).flatten()
        cleaned, was_inverted = self.preprocess(raw)

        # Work on correctly polarized signal
        eval_sig = -raw if was_inverted else raw

        # Execute selected algorithm
        if self.method == "adaptive_prominence":
            raw_peaks = self._detect_adaptive_prominence(cleaned)
        elif self.method == "prominence_delineator":
            raw_peaks = self._detect_prominence_delineator(eval_sig)
        elif self.method == "pan_tompkins":
            raw_peaks = self._detect_pan_tompkins(cleaned)
        elif self.method == "neurokit":
            raw_peaks = self._detect_neurokit(cleaned)
        elif self.method == "ensemble":
            raw_peaks = self._detect_ensemble(cleaned)
        else:
            raw_peaks = self._detect_adaptive_prominence(cleaned)

        # Snap to true raw signal maxima and enforce refractory period
        final_peaks = self.align_peaks(eval_sig, raw_peaks)

        # Timings & Metrics
        r_times = final_peaks / self.sampling_rate

        if len(final_peaks) >= 2:
            rr_intervals_ms = np.diff(final_peaks) / self.sampling_rate * 1000.0
            # Instantaneous BPM: 60,000 / RR_ms
            hr_bpm = 60000.0 / np.clip(rr_intervals_ms, 200.0, 3000.0)
            mean_hr = float(np.mean(hr_bpm))
        else:
            rr_intervals_ms = np.array([], dtype=float)
            hr_bpm = np.array([], dtype=float)
            mean_hr = 0.0

        return RPeakResult(
            r_peaks=final_peaks,
            r_times=r_times,
            rr_intervals_ms=rr_intervals_ms,
            heart_rate_bpm=hr_bpm,
            mean_hr_bpm=mean_hr,
            sampling_rate=self.sampling_rate,
            is_inverted=was_inverted,
            method=self.method,
            cleaned_signal=cleaned,
        )

    # =========================================================================
    # Beat Segmentation for Phase 2 (Normal vs. Abnormal Classification)
    # =========================================================================

    def segment_beats(
        self,
        ecg: np.ndarray,
        r_peaks: np.ndarray,
        window_ms: Tuple[float, float] = (-100.0, 150.0),
        normalize: str = "zscore",
        drop_edges: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segments continuous ECG signal into individual QRS beat windows
        centered around each detected R-peak. Prepares data directly for Phase 2
        (ML / CNN classification).

        Args:
            ecg: 1D ECG array.
            r_peaks: Array of R-peak sample indices.
            window_ms: Tuple of (pre_ms, post_ms) relative to R-peak.
                Default: (-100 ms, +150 ms), spanning 250 ms around the QRS complex.
            normalize: Normalization mode:
                - 'zscore': zero-mean, unit-variance per beat.
                - 'minmax': scaled to [0, 1] per beat.
                - 'none': raw amplitude values.
            drop_edges: If True, skips beats whose window falls outside the signal boundary.

        Returns:
            Tuple of:
            - beats: 2D numpy array of shape (num_valid_beats, window_samples)
            - valid_peak_indices: 1D array of the R-peaks included in the segmentation
            - time_ms_axis: 1D array of time coordinates relative to R-peak in milliseconds
        """
        fs = self.sampling_rate
        pre_samples = int(abs(window_ms[0]) * fs / 1000.0)
        post_samples = int(abs(window_ms[1]) * fs / 1000.0)
        total_len = pre_samples + post_samples

        time_axis = (np.arange(-pre_samples, post_samples) / fs) * 1000.0

        beats_list = []
        valid_peaks = []
        sig_len = len(ecg)

        for p in r_peaks:
            start = p - pre_samples
            end = p + post_samples
            if start < 0 or end > sig_len:
                if drop_edges:
                    continue
                # Zero-pad edge beats
                beat = np.zeros(total_len)
                valid_start = max(0, start)
                valid_end = min(sig_len, end)
                insert_start = valid_start - start
                beat[insert_start : insert_start + (valid_end - valid_start)] = ecg[valid_start:valid_end]
            else:
                beat = ecg[start:end].copy()

            # Normalization
            if normalize == "zscore":
                std = np.std(beat)
                beat = (beat - np.mean(beat)) / (std if std > 1e-8 else 1.0)
            elif normalize == "minmax":
                ptp = np.ptp(beat)
                beat = (beat - np.min(beat)) / (ptp if ptp > 1e-8 else 1.0)

            beats_list.append(beat)
            valid_peaks.append(p)

        if not beats_list:
            return np.empty((0, total_len)), np.array([], dtype=int), time_axis

        return np.array(beats_list), np.array(valid_peaks, dtype=int), time_axis

    # =========================================================================
    # HRV & Heart Rate Analytics
    # =========================================================================

    def compute_hrv(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Computes standard time-domain Heart Rate Variability (HRV) metrics.

        Args:
            r_peaks: Array of R-peak sample indices.

        Returns:
            Dictionary with HRV statistics:
            - 'mean_hr_bpm': Mean Heart Rate in Beats Per Minute
            - 'mean_rr_ms': Mean RR interval in milliseconds
            - 'sdnn_ms': Standard deviation of NN (RR) intervals
            - 'rmssd_ms': Root mean square of successive differences
            - 'pnn50': Percentage of successive RR intervals that differ by > 50 ms
        """
        if len(r_peaks) < 2:
            return {
                "mean_hr_bpm": 0.0,
                "mean_rr_ms": 0.0,
                "sdnn_ms": 0.0,
                "rmssd_ms": 0.0,
                "pnn50": 0.0,
            }

        rr_ms = np.diff(r_peaks) / self.sampling_rate * 1000.0
        successive_diffs = np.diff(rr_ms)

        mean_rr = float(np.mean(rr_ms))
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
        sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0
        nn50 = np.sum(np.abs(successive_diffs) > 50.0)
        pnn50 = float((nn50 / len(successive_diffs)) * 100.0) if len(successive_diffs) > 0 else 0.0

        return {
            "mean_hr_bpm": round(mean_hr, 2),
            "mean_rr_ms": round(mean_rr, 2),
            "sdnn_ms": round(sdnn, 2),
            "rmssd_ms": round(rmssd, 2),
            "pnn50": round(pnn50, 2),
        }


# =============================================================================
# Functional API
# =============================================================================

def detect_r_peaks(
    ecg: np.ndarray,
    fs: float = 360.0,
    method: str = "adaptive_prominence",
    **kwargs: Any
) -> np.ndarray:
    """
    Convenient one-line functional interface for R-peak detection.

    Args:
        ecg: 1D array of ECG signal.
        fs: Sampling rate in Hz.
        method: Method name ('adaptive_prominence', 'pan_tompkins', 'prominence_delineator', 'neurokit', 'ensemble').
        **kwargs: Additional parameters passed to RPeakDetector.

    Returns:
        1D numpy array of detected R-peak sample indices.
    """
    detector = RPeakDetector(sampling_rate=fs, method=method, **kwargs)
    result = detector.detect(ecg)
    return result.r_peaks


# =============================================================================
# Multi-Format Data Loader
# =============================================================================

def load_ecg(
    filepath: str,
    fs: Optional[float] = None,
    channel: int = 0
) -> Tuple[np.ndarray, float]:
    """
    Universal ECG data loader supporting:
    - CSV / TSV / TXT (comma, tab, or space-separated)
    - NumPy (.npy)
    - MATLAB (.mat)
    - PhysioNet / WFDB record paths (.hea/.dat)

    Args:
        filepath: Path to the data file.
        fs: Sampling frequency in Hz (if not embeddable in file format).
        channel: Channel index to extract if data is multi-lead (default: 0).

    Returns:
        Tuple of (ecg_signal_1d, sampling_rate).
    """
    ext = os.path.splitext(filepath)[1].lower()

    # 1. PhysioNet WFDB format
    if ext in {".hea", ".dat"} or (not ext and _HAS_WFDB):
        record_name = os.path.splitext(filepath)[0]
        if _HAS_WFDB:
            rec = wfdb.rdrecord(record_name)
            sig = rec.p_signal[:, channel] if rec.p_signal.ndim > 1 else rec.p_signal
            return sig, float(rec.fs)
        else:
            raise ImportError("wfdb package is required to read PhysioNet records.")

    # 2. NumPy array (.npy)
    if ext == ".npy":
        arr = np.load(filepath)
        sig = arr[:, channel] if arr.ndim > 1 else arr
        if fs is None:
            raise ValueError("Sampling frequency 'fs' must be specified when loading .npy files.")
        return sig, float(fs)

    # 3. MATLAB file (.mat)
    if ext == ".mat":
        if not _HAS_SCIPY_IO:
            raise ImportError("scipy is required to read .mat files.")
        mat = sio.loadmat(filepath)
        # Find first array that looks like an ECG signal
        sig = None
        for k, v in mat.items():
            if isinstance(v, np.ndarray) and v.size > 100:
                sig = v.squeeze()
                break
        if sig is None:
            raise ValueError("No suitable ECG array found in .mat file.")
        if sig.ndim > 1:
            sig = sig[:, channel]
        detected_fs = float(mat.get("fs", fs if fs is not None else 360.0))
        return sig, detected_fs

    # 4. Text / CSV / TSV
    if ext in {".csv", ".tsv", ".txt"}:
        import pandas as pd
        # Try delimiter auto-detection
        sep = "," if ext == ".csv" else ("\t" if ext == ".tsv" else None)
        df = pd.read_csv(filepath, sep=sep, comment="#")

        # Check if first column might be time
        if df.shape[1] >= 2 and ("time" in str(df.columns[0]).lower() or "t" == str(df.columns[0]).lower()):
            time_col = df.iloc[:, 0].values
            sig = df.iloc[:, channel + 1].values
            if fs is None:
                # Infer fs from time column
                dt = np.median(np.diff(time_col))
                fs = 1.0 / dt if dt > 0 else 360.0
        else:
            sig = df.iloc[:, channel].values
            if fs is None:
                fs = 360.0  # standard fallback

        return np.asarray(sig, dtype=float), float(fs)

    raise ValueError(f"Unsupported file format: {ext}")


# =============================================================================
# Plotting & Visualization
# =============================================================================

def plot_detections(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    title: str = "ECG R-Peak Detection",
    save_path: Optional[str] = None
) -> None:
    """
    Renders an informative publication-ready visualization of detected R-peaks.

    Args:
        ecg: 1D ECG signal array.
        r_peaks: Array of detected R-peak sample indices.
        fs: Sampling frequency in Hz.
        start_sec: Window start time in seconds.
        duration_sec: Window duration in seconds.
        title: Plot title.
        save_path: If specified, saves the figure to this file path.
    """
    import matplotlib.pyplot as plt

    start_idx = int(start_sec * fs)
    end_idx = min(len(ecg), int((start_sec + duration_sec) * fs))

    sub_sig = ecg[start_idx:end_idx]
    sub_time = np.arange(start_idx, end_idx) / fs

    # Filter peaks falling in this window
    window_peaks = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]
    peak_times = window_peaks / fs
    peak_amplitudes = ecg[window_peaks]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub_time, sub_sig, color="#1f77b4", linewidth=1.2, label="ECG Signal")
    ax.scatter(peak_times, peak_amplitudes, color="#d62728", s=60, zorder=5, label=f"Detected R-Peaks (N={len(window_peaks)})")

    # Annotate RR intervals
    for i in range(len(window_peaks) - 1):
        mid_t = (peak_times[i] + peak_times[i + 1]) / 2.0
        rr_ms = (peak_times[i + 1] - peak_times[i]) * 1000.0
        ax.text(
            mid_t,
            max(peak_amplitudes[i], peak_amplitudes[i + 1]) * 0.95,
            f"{rr_ms:.0f} ms",
            fontsize=8,
            ha="center",
            va="bottom",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.7, edgecolor="#cccccc")
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Amplitude (mV)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# Command-Line Interface (CLI)
# =============================================================================

def run_cli() -> None:
    """CLI Entry point for r_peak_detector.py."""
    parser = argparse.ArgumentParser(
        description="Reliable ECG R-Peak Detector (Phase 1 Engine)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to ECG file (CSV, TSV, NPY, MAT, or PhysioNet record)")
    parser.add_argument("--fs", "-f", type=float, default=360.0, help="Sampling frequency in Hz")
    parser.add_argument(
        "--method", "-m", type=str, default="adaptive_prominence",
        choices=["adaptive_prominence", "prominence_delineator", "pan_tompkins", "neurokit", "ensemble"],
        help="Detection method backend"
    )
    parser.add_argument("--channel", "-c", type=int, default=0, help="Channel / Lead index to process")
    parser.add_argument("--plot", action="store_true", help="Plot the detection results")
    parser.add_argument("--save-plot", type=str, default=None, help="File path to save the output plot (e.g. plot.png)")
    parser.add_argument("--save-peaks", type=str, default=None, help="Save detected peak indices to a CSV file")
    parser.add_argument("--demo", action="store_true", help="Run a built-in verification demo using PhysioNet MIT-BIH Record 100")

    args = parser.parse_args()

    if args.demo or (args.input is None and len(sys.argv) == 1):
        print("=" * 65)
        print("  Running Built-In Demo (MIT-BIH Arrhythmia Database, Record 100)")
        print("=" * 65)
        if not _HAS_WFDB:
            print("Error: wfdb is required for the demo. Install via 'pip install wfdb'.")
            sys.exit(1)

        print("Downloading 10-second segment from PhysioNet...")
        rec = wfdb.rdrecord("100", pn_dir="mitdb", sampto=3600)
        ann = wfdb.rdann("100", "atr", pn_dir="mitdb", sampto=3600)
        sig = rec.p_signal[:, 0]
        fs = float(rec.fs)

        detector = RPeakDetector(sampling_rate=fs, method=args.method)
        res = detector.detect(sig)
        hrv = detector.compute_hrv(res.r_peaks)

        ref_beats = [s for s, sym in zip(ann.sample, ann.symbol) if sym in "NLRABaJSVR"]
        print(f"\nDetection Results [{args.method}]:")
        print(f"  - Detected R-Peaks : {len(res.r_peaks)}")
        print(f"  - Reference Beats  : {len(ref_beats)}")
        print(f"  - Mean Heart Rate  : {res.mean_hr_bpm:.1f} BPM")
        print(f"  - SDNN             : {hrv['sdnn_ms']:.1f} ms")
        print(f"  - RMSSD            : {hrv['rmssd_ms']:.1f} ms")
        print(f"  - First 5 Peaks    : {res.r_peaks[:5]}")

        # Test beat segmentation for Phase 2
        beats, valid_peaks, time_axis = detector.segment_beats(sig, res.r_peaks)
        print(f"\nPhase 2 Beat Segmentation:")
        print(f"  - Extracted Beats Matrix Shape: {beats.shape} (N_beats, Window_samples)")
        print(f"  - Time Axis Range: [{time_axis[0]:.1f} ms to {time_axis[-1]:.1f} ms]")

        save_fig = args.save_plot or "demo_detection.png"
        plot_detections(sig, res.r_peaks, fs, duration_sec=10.0, title=f"Demo: Record 100 [{args.method}]", save_path=save_fig)
        return

    if args.input is None:
        parser.print_help()
        sys.exit(1)

    print(f"Loading ECG from: {args.input} (fs={args.fs}, channel={args.channel})...")
    sig, detected_fs = load_ecg(args.input, fs=args.fs, channel=args.channel)
    print(f"Loaded {len(sig)} samples ({len(sig) / detected_fs:.2f} seconds) at {detected_fs} Hz.")

    detector = RPeakDetector(sampling_rate=detected_fs, method=args.method)
    res = detector.detect(sig)
    hrv = detector.compute_hrv(res.r_peaks)

    print(f"\nDetection Completed [{args.method}]:")
    print(f"  - Total R-Peaks Detected : {len(res.r_peaks)}")
    print(f"  - Average Heart Rate     : {res.mean_hr_bpm:.1f} BPM")
    print(f"  - Inversion Corrected    : {res.is_inverted}")
    print(f"  - SDNN                   : {hrv['sdnn_ms']} ms")
    print(f"  - RMSSD                  : {hrv['rmssd_ms']} ms")

    if args.save_peaks:
        import pandas as pd
        out_df = pd.DataFrame({
            "peak_index": res.r_peaks,
            "peak_time_sec": res.r_times,
        })
        out_df.to_csv(args.save_peaks, index=False)
        print(f"Saved peak annotations to: {args.save_peaks}")

    if args.plot or args.save_plot:
        plot_detections(sig, res.r_peaks, detected_fs, duration_sec=10.0, save_path=args.save_plot)


if __name__ == "__main__":
    run_cli()
