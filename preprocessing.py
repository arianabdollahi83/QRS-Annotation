import numpy as np
from scipy.stats import zscore
from typing import Tuple, Optional
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


class ECGPreprocessor:
    def __init__(
        self,
        sampling_rate: float = 360.0,
        adaptive_frequency: float = 50.0,
        baseline_method: str | Sequence[str] = "dual_median_sg",
        baseline_highpass_cutoff: float = 0.5,
        baseline_highpass_order: int = 4,
        baseline_median_window1_sec: float = 0.2,
        baseline_median_window2_sec: float = 0.6,
        baseline_sg_window_sec: float = 10.0,
        baseline_sg_poly_order: int = 3,
        ac_noise_method: str = "adaptive_oscillator",
        high_frequency_method: str = "savgol",
        muscle_method: Optional[str] = None,
        motion_method: Optional[str] = None,
    ):
        self.sampling_rate = sampling_rate

        self.adaptive_mu = 0.01
        self.ac_freq = adaptive_frequency
        self.ac_range = 4.0
        self.ac_harmonics = 3

        self.sg_window_size = 5  # must be odd
        self.sg_poly_order = 3
        self.baseline_highpass_cutoff = baseline_highpass_cutoff
        self.baseline_highpass_order = baseline_highpass_order
        self.baseline_median_window1_sec = baseline_median_window1_sec
        self.baseline_median_window2_sec = baseline_median_window2_sec
        self.baseline_sg_window_sec = baseline_sg_window_sec
        self.baseline_sg_poly_order = baseline_sg_poly_order

        self.filter_methods = {
            "baseline": baseline_method,
            "ac_noise": ac_noise_method,
            "high_frequency": high_frequency_method,
            "muscle": muscle_method,
            "motion": motion_method,
        }

        self._filter_algorithms = {
            "baseline": {
                "cubic_spline": self._baseline_cubic_spline,
                "dual_median_sg": self._baseline_dual_median_sg,
                "butter_highpass": self._baseline_butter_highpass,
            },
            "ac_noise": {
                "adaptive_oscillator": self._ac_adaptive_oscillator,
            },
            "high_frequency": {
                "savgol": self._high_frequency_savgol,
            },
            "muscle": {},
            "motion": {},
        }

        for filter_name, method in self.filter_methods.items():
            if filter_name == "baseline":
                self.set_baseline_method(method)
            elif method is not None:
                self.set_filter_method(filter_name, method)

    def set_filter_method(self, filter_name: str, method: Optional[str]) -> None:
        if filter_name not in self._filter_algorithms:
            raise ValueError(f"Unknown filter: {filter_name}")
        if method is not None and method not in self._filter_algorithms[filter_name]:
            available = ", ".join(self._filter_algorithms[filter_name]) or "none"
            raise ValueError(
                f"Unsupported {filter_name} method '{method}'. Available: {available}"
            )
        self.filter_methods[filter_name] = method

    def set_baseline_method(self, method: str | Sequence[str]) -> None:
        methods = [method] if isinstance(method, str) else list(method)
        if not methods:
            raise ValueError("At least one baseline method must be configured")
        for selected_method in methods:
            self.set_filter_method("baseline", selected_method)
        self.filter_methods["baseline"] = methods

    def set_baseline_highpass_cutoff(self, cutoff: float) -> None:
        if cutoff <= 0 or cutoff >= self.sampling_rate / 2:
            raise ValueError(
                "Baseline high-pass cutoff must be between 0 and the Nyquist frequency"
            )
        self.baseline_highpass_cutoff = cutoff

    def set_baseline_dual_median_sg_windows(
        self,
        median_window1_sec: float,
        median_window2_sec: float,
        sg_window_sec: float,
        sg_poly_order: Optional[int] = None,
    ) -> None:
        if median_window1_sec <= 0 or median_window2_sec <= 0 or sg_window_sec <= 0:
            raise ValueError("All baseline filter windows must be greater than zero")
        if sg_poly_order is not None and sg_poly_order < 1:
            raise ValueError("SG polynomial order must be at least 1")

        self.baseline_median_window1_sec = median_window1_sec
        self.baseline_median_window2_sec = median_window2_sec
        self.baseline_sg_window_sec = sg_window_sec
        if sg_poly_order is not None:
            self.baseline_sg_poly_order = sg_poly_order

    def _get_filter_algorithm(self, filter_name: str, method: Optional[str] = None):
        selected_method = method or self.filter_methods[filter_name]
        if selected_method is None:
            raise ValueError(f"No {filter_name} method is configured")
        algorithms = self._filter_algorithms[filter_name]
        if selected_method not in algorithms:
            available = ", ".join(algorithms) or "none"
            raise ValueError(
                f"Unsupported {filter_name} method '{selected_method}'. "
                f"Available: {available}"
            )
        return algorithms[selected_method]

    def __baseline_filter(
        self,
        sig: np.ndarray,
        baseline_method: Optional[str | Sequence[str]] = None,
    ) -> np.ndarray:
        if sig.size == 0:
            return sig

        configured_methods = self.filter_methods["baseline"]
        selected_methods = baseline_method or configured_methods
        if isinstance(selected_methods, str):
            selected_methods = [selected_methods]

        filtered = sig
        for method in selected_methods:
            filtered = self._get_filter_algorithm("baseline", method)(filtered)
        return filtered

    def _baseline_cubic_spline(self, sig: np.ndarray) -> np.ndarray:
        import neurokit2 as nk
        from scipy.interpolate import CubicSpline

        _, rpeaks = nk.ecg_peaks(sig, sampling_rate=self.sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

        if len(rpeaks) < 2:
            return sig.copy()

        midpoints = [(rpeaks[i] + rpeaks[i + 1]) // 2 for i in range(len(rpeaks) - 1)]

        iso = []
        N = len(sig)
        for i, mid in enumerate(midpoints):
            left = rpeaks[i]
            right = rpeaks[i + 1]
            dist = max(1, right - left)
            half = max(1, min(15, dist // 6))

            start = max(0, mid - half)
            end = min(N - 1, mid + half)

            window = sig[start : end + 1]
            iso.append(np.median(window))

        knot_x = np.array(midpoints, dtype=float)
        knot_y = np.array(iso, dtype=float)

        spline = CubicSpline(knot_x, knot_y, bc_type="natural")
        baseline = spline(np.arange(N))

        return sig - baseline

    def _baseline_dual_median_sg(self, sig: np.ndarray) -> np.ndarray:
        from scipy.ndimage import median_filter
        from scipy.signal import savgol_filter

        fs = self.sampling_rate
        win1 = max(1, int(self.baseline_median_window1_sec * fs))
        win2 = max(1, int(self.baseline_median_window2_sec * fs))

        if win1 % 2 == 0:
            win1 += 1
        if win2 % 2 == 0:
            win2 += 1

        baseline_stage1 = median_filter(sig, size=win1)
        baseline = median_filter(baseline_stage1, size=win2)

        sg_poly_order = self.baseline_sg_poly_order
        sg_window = max(
            int(np.round(self.baseline_sg_window_sec * fs)),
            sg_poly_order + 2,
        )
        if sg_window % 2 == 0:
            sg_window += 1

        if sg_window > len(baseline):
            sg_window = len(baseline) if len(baseline) % 2 == 1 else len(baseline) - 1

        if sg_window <= sg_poly_order:
            baseline_smooth = baseline
        else:
            baseline_smooth = savgol_filter(
                baseline,
                sg_window,
                sg_poly_order,
            )

        return sig - baseline_smooth

    def _baseline_butter_highpass(self, sig: np.ndarray) -> np.ndarray:
        from scipy.signal import butter, filtfilt

        cutoff = self.baseline_highpass_cutoff
        nyquist = self.sampling_rate / 2
        if cutoff <= 0 or cutoff >= nyquist:
            raise ValueError(
                "Baseline high-pass cutoff must be between 0 and the Nyquist frequency"
            )

        b, a = butter(
            self.baseline_highpass_order,
            cutoff,
            btype="highpass",
            fs=self.sampling_rate,
        )
        padlen = min(3 * max(len(a), len(b)), len(sig) - 1)
        if padlen <= 0:
            return sig.copy()
        return filtfilt(b, a, sig, padlen=padlen)

    def __ac_noise_filter(self, sig: np.ndarray) -> np.ndarray:
        if sig.size == 0:
            return sig

        return self._get_filter_algorithm("ac_noise")(sig)

    def _ac_adaptive_oscillator(self, sig: np.ndarray) -> np.ndarray:

        fs = self.sampling_rate
        mu = self.adaptive_mu
        center_freq = self.ac_freq
        search_range = self.ac_range
        num_harmonics = self.ac_harmonics

        def detect_frequency_fft(x, center: float) -> float:
            n = len(x)
            fft_vals = np.fft.fft(x)
            freqs = np.fft.fftfreq(n, 1.0 / fs)

            mask = (freqs >= center - search_range / 2) & (
                freqs <= center + search_range / 2
            )
            if not np.any(mask):
                return float(center)

            mag = np.abs(fft_vals[mask])
            freq_subset = freqs[mask]
            best_idx = int(np.argmax(mag))
            return float(np.abs(freq_subset[best_idx]))

        def remove_frequency(x, target_freq: float) -> np.ndarray:
            # Simple adaptive oscillator subtraction (LMS-like)
            r_cos = 1.0
            r_sin = 0.0
            w_cos = 0.0
            w_sin = 0.0
            omega = 2.0 * np.pi * target_freq / fs
            cosw = np.cos(omega)
            sinw = np.sin(omega)
            eps = 1e-8
            out = np.zeros_like(x, dtype=np.float32)

            for n, xn in enumerate(x):
                y = w_cos * r_cos + w_sin * r_sin
                e = xn - y
                out[n] = e
                ref_energy = r_cos * r_cos + r_sin * r_sin + eps
                norm_mu = mu / ref_energy
                w_cos += norm_mu * e * r_cos
                w_sin += norm_mu * e * r_sin
                next_cos = r_cos * cosw - r_sin * sinw
                next_sin = r_cos * sinw + r_sin * cosw
                r_cos, r_sin = next_cos, next_sin

            return out

        mains_freq = detect_frequency_fft(sig, center_freq)
        cleaned = remove_frequency(sig, mains_freq)

        for h in range(2, num_harmonics + 1):
            harmonic_center = mains_freq * h
            harmonic_freq = detect_frequency_fft(cleaned, harmonic_center)
            cleaned = remove_frequency(cleaned, harmonic_freq)

        return cleaned

    def __high_frequency_noise_filter(self, sig: np.ndarray) -> np.ndarray:
        if sig.size == 0:
            return sig

        return self._get_filter_algorithm("high_frequency")(sig)

    def _high_frequency_savgol(self, sig: np.ndarray) -> np.ndarray:

        from scipy.signal import savgol_filter

        filtered = savgol_filter(
            sig,
            window_length=self.sg_window_size,
            polyorder=self.sg_poly_order,
            mode="mirror",
        )

        return filtered.astype(np.float32)

    def resample(
        self, sig: np.ndarray, target_rate: float = 500.0
    ) -> Tuple[np.ndarray, float]:
        if sig.size == 0:
            return sig, self.sampling_rate

        from scipy.signal import resample_poly

        up = int(target_rate)
        down = int(self.sampling_rate)

        sig_resampled = resample_poly(sig, up, down)

        return sig_resampled.astype(np.float32), target_rate

    def preprocess(
        self,
        sig: np.ndarray,
        apply_baseline_filter: bool = True,
        apply_ac_noise_filter: bool = True,
        apply_high_frequency_noise_filter: bool = True,
        do_normalize: bool = False,
        do_resample: bool = False,
        target_rate: Optional[float] = None,
        baseline_method: Optional[str | Sequence[str]] = None,
    ) -> Tuple[np.ndarray, float]:

        current_sig = sig.copy()
        current_rate = self.sampling_rate

        if apply_ac_noise_filter:
            logger.info("Applying AC noise filter...")
            current_sig = self.__ac_noise_filter(current_sig)

        if apply_high_frequency_noise_filter:
            logger.info("Applying high-frequency noise filter...")
            current_sig = self.__high_frequency_noise_filter(current_sig)

        if apply_baseline_filter:
            logger.info("Applying baseline filter...")
            current_sig = self.__baseline_filter(current_sig, baseline_method)

        if do_normalize:
            logger.info("Normalizing signal...")
            current_sig = zscore(current_sig)

        if do_resample and target_rate:
            logger.info(f"Resampling to {target_rate} Hz...")
            current_sig, current_rate = self.resample(current_sig, target_rate)
            logger.info("Applying high-frequency noise filter...")
            current_sig = self.__high_frequency_noise_filter(current_sig)

        return current_sig, current_rate


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from scipy.signal import welch
#     from signal_reader import ECGSignalReader

#     # ---------------------------------------------------------
#     # Load a MIT-BIH record using the project's signal reader
#     # ---------------------------------------------------------
#     # data_dir = r"Datasets/MIT BIH Arrhythmia 1.0.0"
#     # record_name = "102"
#     data_dir = r"./Datasets/Data-Holter.csv"

#     reader = ECGSignalReader()
#     # data, meta = reader.read_mit_bih(record_name, data_dir=data_dir)
#     data, meta = reader.read_csv(data_dir)

#     # choose lead (if multi-lead) and sampling rate from metadata
#     lead_index = 0
#     if data.ndim > 1:
#         sig = data[:, lead_index]
#     else:
#         sig = data

#     fs_meta = float(meta.get("fs", 250.0))

#     # print(
#     #     f"Loaded record {record_name} from {data_dir}, signal shape: {sig.shape}, fs={fs_meta}"
#     # )

#     # ---------------------------------------------------------
#     # Create preprocessor with sampling rate from record
#     # ---------------------------------------------------------
#     pre = ECGPreprocessor(sampling_rate=fs_meta)

#     # ---------------------------------------------------------
#     # Run full preprocessing pipeline
#     # ---------------------------------------------------------
#     print("Running full preprocessing pipeline...")

#     processed, new_fs = pre.preprocess(
#         sig,
#         apply_baseline_filter=True,
#         apply_ac_noise_filter=True,
#         apply_high_frequency_noise_filter=True,
#         do_resample=True,
#         target_rate=250.0,
#     )

#     print(f"Output sampling rate: {new_fs} Hz")
#     print(f"Processed signal length: {len(processed)}")

#     # ---------------------------------------------------------
#     # Compute PSD using Welch's method
#     # ---------------------------------------------------------
#     f_raw, psd_raw = welch(sig, fs=fs_meta, nperseg=1024)
#     f_proc, psd_proc = welch(processed, fs=new_fs, nperseg=1024)

#     # ---------------------------------------------------------
#     # Plot results
#     # ---------------------------------------------------------
#     # ---------------------------------------------------------
#     plt.figure(figsize=(14, 8))

#     # ---------------------------------------------------------
#     # 1) Time‑domain comparison
#     # ---------------------------------------------------------
#     plt.subplot(2, 1, 1)
#     plt.title("ECG Signal Comparison (Raw vs Processed)")
#     plt.plot(sig, label="Raw ECG", alpha=0.7)
#     plt.plot(processed, label=f"Processed ECG ({int(new_fs)} Hz)", alpha=0.7)
#     plt.xlabel("Samples")
#     plt.ylabel("Amplitude")
#     plt.legend()

#     # ---------------------------------------------------------
#     # 2) PSD comparison
#     # ---------------------------------------------------------
#     plt.subplot(2, 1, 2)
#     plt.title("PSD Comparison (Raw vs Processed)")
#     plt.semilogy(f_raw, psd_raw, label="Raw PSD", alpha=0.7)
#     plt.semilogy(f_proc, psd_proc, label="Processed PSD", alpha=0.7)
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Power")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # plt.figure(figsize=(14, 10))

#     # # Time-domain plots
#     # plt.subplot(2, 2, 1)
#     # plt.title("Original ECG (raw)")
#     # plt.plot(sig)
#     # plt.xlabel("Samples")

#     # plt.subplot(2, 2, 2)
#     # plt.title(f"Processed ECG ({int(new_fs)} Hz, filtered)")
#     # plt.plot(processed)
#     # plt.xlabel("Samples")

#     # # PSD plots
#     # plt.subplot(2, 2, 3)
#     # plt.title("PSD of Original ECG")
#     # plt.semilogy(f_raw, psd_raw)
#     # plt.xlabel("Frequency (Hz)")
#     # plt.ylabel("Power")

#     # plt.subplot(2, 2, 4)
#     # plt.title("PSD of Processed ECG")
#     # plt.semilogy(f_proc, psd_proc)
#     # plt.xlabel("Frequency (Hz)")
#     # plt.ylabel("Power")

#     # plt.tight_layout()
#     # plt.show()

#     print("Test completed.")
