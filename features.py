import numpy as np
from scipy import signal
from typing import Dict, Any, Tuple

TARGET_FS = 360.0
PRE_MS = 200.0   # -200 ms
POST_MS = 250.0  # +250 ms
BEAT_SAMPLES = int(round((PRE_MS + POST_MS) / 1000.0 * TARGET_FS))  # 162 samples
P_WAVE_SAMPLES = 10
N_NEIGHBORS = 20  # 20 before, 20 after -> 40 RR intervals, 41-beat morphology pool
CLASS_NAMES = ["N", "V", "S", "F", "Q"]


def bessel_bandpass_filter(sig: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Zero-phase forward-backward Bessel filter. Bessel filters exhibit maximally flat
    group delay across the passband, preserving ECG wave morphology without phase distortion.
    Filtering the continuous signal as a whole prevents boundary artifacts.
    """
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = signal.bessel(order, [low, high], btype='bandpass', output='ba')
    return signal.filtfilt(b, a, sig)


def resample_beat_window(beat_raw: np.ndarray, orig_fs: float, target_len: int = BEAT_SAMPLES) -> np.ndarray:
    """Resamples a sliced raw window to target_len (360 Hz equivalent)."""
    if len(beat_raw) == target_len and np.isclose(orig_fs, TARGET_FS):
        return beat_raw.astype(np.float32)
    resampled = signal.resample(beat_raw, target_len)
    return resampled.astype(np.float32)


def z_score_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalization with epsilon guard."""
    mu = np.mean(x)
    std = np.std(x)
    return (x - mu) / (std + 1e-8)


def extract_p_wave_features(z_beat: np.ndarray) -> np.ndarray:
    """
    Extracts P-wave representation from the first 10 samples of the normalized beat:
    1) Normalized energy: mean(x^2)
    2) Peak amplitude: max(x)
    3) Area under the curve: trapezoidal integral
    """
    p_seg = z_beat[:P_WAVE_SAMPLES]
    energy = float(np.mean(p_seg ** 2))
    peak = float(np.max(p_seg))
    # area = float(np.trapz(p_seg))
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    area = float(trapz_fn(p_seg))
    return np.array([energy, peak, area], dtype=np.float32)


def extract_all_features(sig: np.ndarray, r_peaks: np.ndarray, fs: float, 
                         mode: str = "inference") -> Dict[str, np.ndarray]:
    """
    Extracts features for all detected R-peaks.
    
    Inputs:
      sig: 1D ECG array (filtered)
      r_peaks: 1D array of sample indices
      fs: Sampling rate of sig
      mode: 'train' (excludes edge beats lacking 20 neighbors) or 'inference' (pads boundaries)
    
    Returns a dictionary of structured numpy arrays.
    """
    n_peaks = len(r_peaks)
    pre_samp = int(round(PRE_MS / 1000.0 * fs))
    post_samp = int(round(POST_MS / 1000.0 * fs))

    # 1. Slice raw beats
    beats_resampled = np.zeros((n_peaks, BEAT_SAMPLES), dtype=np.float32)
    beats_zscored = np.zeros((n_peaks, BEAT_SAMPLES), dtype=np.float32)
    p_features = np.zeros((n_peaks, 3), dtype=np.float32)

    for i, pk in enumerate(r_peaks):
        s0 = pk - pre_samp
        s1 = pk + post_samp
        if s0 < 0:
            left_pad = -s0
            chunk = np.pad(sig[0:max(0, s1)], (left_pad, 0), mode='edge')
        elif s1 > len(sig):
            right_pad = s1 - len(sig)
            chunk = np.pad(sig[s0:], (0, right_pad), mode='edge')
        else:
            chunk = sig[s0:s1]
            
        rs = resample_beat_window(chunk, fs, BEAT_SAMPLES)
        z_rs = z_score_normalize(rs)
        beats_resampled[i] = rs
        beats_zscored[i] = z_rs
        p_features[i] = extract_p_wave_features(z_rs)

    # 2. RR Intervals (in seconds)
    rr_intervals = np.diff(r_peaks) / float(fs)
    if len(rr_intervals) == 0:
        rr_intervals = np.array([0.8], dtype=np.float32)

    rec_med_rr = float(np.median(rr_intervals))
    rec_med_beat = np.median(beats_resampled, axis=0)

    # Prepare output containers
    if mode == "train":
        start_idx = N_NEIGHBORS
        end_idx = n_peaks - N_NEIGHBORS
        eval_indices = np.arange(start_idx, max(start_idx, end_idx))
    else:
        eval_indices = np.arange(n_peaks)

    num_eval = len(eval_indices)
    out_beat = np.zeros((num_eval, BEAT_SAMPLES), dtype=np.float32)
    out_diff = np.zeros((num_eval, BEAT_SAMPLES), dtype=np.float32)
    out_p_feat = np.zeros((num_eval, 3), dtype=np.float32)
    out_rr_raw = np.zeros((num_eval, 40), dtype=np.float32)
    out_rr_norm = np.zeros((num_eval, 40), dtype=np.float32)
    out_rr_z = np.zeros((num_eval, 40), dtype=np.float32)

    for out_i, idx in enumerate(eval_indices):
        out_beat[out_i] = beats_zscored[idx]
        out_p_feat[out_i] = p_features[idx]

        # 41-beat local window for median morphology (20 left + center + 20 right)
        w_left = max(0, idx - N_NEIGHBORS)
        w_right = min(n_peaks, idx + N_NEIGHBORS + 1)
        local_beats = beats_resampled[w_left:w_right]

        if mode == "inference" and (idx < N_NEIGHBORS or idx >= n_peaks - N_NEIGHBORS):
            pad_needed = 41 - len(local_beats)
            if pad_needed > 0:
                pads = np.tile(rec_med_beat, (pad_needed, 1))
                local_beats = np.vstack([local_beats, pads])
        
        loc_med_beat = np.median(local_beats, axis=0)
        diff_wave = beats_resampled[idx] - loc_med_beat
        out_diff[out_i] = z_score_normalize(diff_wave)

        # 40 RR intervals (20 preceding, 20 succeeding)
        left_rr_start = idx - N_NEIGHBORS
        left_rr_end = idx
        right_rr_start = idx
        right_rr_end = idx + N_NEIGHBORS

        left_rr = rr_intervals[max(0, left_rr_start):max(0, left_rr_end)]
        right_rr = rr_intervals[min(len(rr_intervals), right_rr_start):min(len(rr_intervals), right_rr_end)]

        if mode == "inference":
            if len(left_rr) < N_NEIGHBORS:
                pad_l = np.full(N_NEIGHBORS - len(left_rr), rec_med_rr, dtype=np.float32)
                left_rr = np.concatenate([pad_l, left_rr])
            if len(right_rr) < N_NEIGHBORS:
                pad_r = np.full(N_NEIGHBORS - len(right_rr), rec_med_rr, dtype=np.float32)
                right_rr = np.concatenate([right_rr, pad_r])

        full_rr = np.concatenate([left_rr, right_rr]).astype(np.float32)
        if len(full_rr) == 40:
            local_med_rr = float(np.median(full_rr))
            out_rr_raw[out_i] = full_rr
            out_rr_norm[out_i] = full_rr / (local_med_rr + 1e-6)
            out_rr_z[out_i] = z_score_normalize(full_rr)

    return {
        "beat": out_beat,
        "diff": out_diff,
        "p_wave": out_p_feat,
        "rr_raw": out_rr_raw,
        "rr_norm": out_rr_norm,
        "rr_z": out_rr_z,
        "eval_indices": eval_indices
    }
