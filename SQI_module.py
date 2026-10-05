import os

import EntropyHub as eh
import biosppy
import neurokit2 as nk
import numpy as np
import scipy.io as sio
from scipy.signal import welch, butter, filtfilt
from sklearn.decomposition import PCA


def lowpass_filter(signal, fs, cutoff=1, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    filter_ = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(filter_[0], filter_[1], signal)


def calculate_s_sqi(ecg_signal):
    s_sqi = np.abs(biosppy.signals.ecg.sSQI(ecg_signal))
    return s_sqi


def calculate_k_sqi(ecg_signal):
    k_sqi = biosppy.signals.ecg.kSQI(ecg_signal)
    return k_sqi


def calculate_sdn_sqi(ecg_signal, fs):
    block_size = int(0.5 * fs)  # Number of samples in 0.5 seconds

    # Ensure ecg_signal is 2D: (num_leads, num_samples)
    if len(ecg_signal.shape) == 1:
        ecg_signal = ecg_signal[
            np.newaxis, :
        ]  # Convert 1D signal to 2D with shape (1, num_samples)

    num_leads = ecg_signal.shape[0]
    num_blocks = ecg_signal.shape[1] // block_size
    sdn_sqi_values = []

    for lead in range(num_leads):
        # Step 1: Calculate the standard deviation for each block of the current lead
        std_devs = [
            np.std(ecg_signal[lead, i * block_size : (i + 1) * block_size])
            for i in range(num_blocks)
        ]

        # Step 2: Calculate SDN-SQI for every group of 10 blocks
        lead_sdn_sqi = []
        for i in range(num_blocks - 9):  # Ensures groups of 10 blocks
            block_group = std_devs[i : i + 10]
            mean_std_dev = np.mean(block_group)
            std_std_dev = np.std(block_group)

            # Step 3: Compute SDN-SQI for the current lead
            sdn_sqi = mean_std_dev + 2 * std_std_dev
            lead_sdn_sqi.append(sdn_sqi)

        sdn_sqi_values.append(lead_sdn_sqi)
        sdn_sqi_final = sdn_sqi_values if num_leads > 1 else sdn_sqi_values[0]
    return np.mean(sdn_sqi_final)


def calculate_pur_sqi(ecg_signal, fs, nperseg=1024):
    """
    Calculate the purSQI metric for an ECG signal.

    Parameters:
    - ecg_signal (array-like): The ECG signal data.
    - fs (int): Sampling rate of the ECG signal in Hz.
    - nperseg (int, optional): Number of samples per segment for Welch's method. Default is 1024.

    Returns:
    - pur_sqi (float): The calculated purSQI value.
    """
    # Step 1: Compute the Power Spectrum using Welch's method
    f, Pxx = welch(ecg_signal, fs=fs, nperseg=nperseg)

    # Step 2: Convert frequency to angular frequency (ω = 2πf)
    omega = 2 * np.pi * f

    # Step 3: Compute the moments ω_0, ω_2, ω_4
    omega_0 = np.sum(Pxx)  # Total power
    omega_2 = np.sum((omega**2) * Pxx)  # Second moment
    omega_4 = np.sum((omega**4) * Pxx)  # Fourth moment

    # Step 4: Calculate purSQI
    if omega_0 == 0 or omega_4 == 0:
        pur_sqi = np.nan  # Avoid division by zero
    else:
        pur_sqi = (omega_2**2) / (omega_0 * omega_4)

    return pur_sqi


def calculate_bas_sqi(ecg_signal, fs):
    bas_sqi = biosppy.signals.ecg.fSQI(
        ecg_signal,
        fs=fs,
        nseg=2 * fs,
        num_spectrum=[0, 1],
        dem_spectrum=[0, 40],
        mode="bas",
    )
    return bas_sqi


def calculate_p_sqi(ecg_signal, fs):
    p_sqi = biosppy.signals.ecg.fSQI(
        ecg_signal, fs=fs, nseg=2 * fs, num_spectrum=[5, 15], dem_spectrum=[5, 40]
    )
    return p_sqi


def calculate_lp_sqi(ecg_signal, fs):
    p_sqi = biosppy.signals.ecg.fSQI(
        ecg_signal, fs=fs, nseg=2 * fs, num_spectrum=[0, 3], dem_spectrum=[0, 100]
    )
    return p_sqi


def calculate_mp_sqi(ecg_signal, fs):
    p_sqi = biosppy.signals.ecg.fSQI(
        ecg_signal, fs=fs, nseg=2 * fs, num_spectrum=[5, 35], dem_spectrum=[0, 100]
    )
    return p_sqi


def calculate_hp_sqi(ecg_signal, fs):
    p_sqi = biosppy.signals.ecg.fSQI(
        ecg_signal, fs=fs, nseg=2 * fs, num_spectrum=[40, 100], dem_spectrum=[0, 100]
    )
    return p_sqi


def calculate_t_sqi(ecg_signal, fs, window_ms=250):
    """
    Calculate the t_SQI metric for an ECG signal.

    Parameters:
    - ecg_signal (array-like): The ECG signal data.
    - fs (int): Sampling rate of the ECG signal in Hz.
    - window_ms (int, optional): Fixed time window around R-peaks in milliseconds. Default is 300 ms.

    Returns:
    - t_sqi (float): The calculated t_SQI value.
    """
    # Step 1: Detect R-Peaks
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]
    if len(rpeaks) == 0:
        return -1  # Return 0 if no R-peaks are detected

    # Step 2: Define window size in samples
    window_samples = int(window_ms * fs / 1000)  # Convert window from ms to samples

    # Step 3: Extract ECG Beats
    beats = []
    for r in rpeaks:
        start = max(0, r - window_samples)
        end = min(len(ecg_signal), r + window_samples)
        beats.append(ecg_signal[start:end])

    # Ensure all beats are the same length (pad or truncate if necessary)
    beat_length = min(len(beat) for beat in beats)
    beats = [beat[:beat_length] for beat in beats]

    # Step 4: Compute Correlation Matrix
    beats_array = np.array(beats)
    m = len(beats)
    correlation_matrix = np.corrcoef(beats_array)

    # Step 5: Compute t_SQI
    t_sqi = np.sum(correlation_matrix) / (m**2)

    return t_sqi


def calculate_bs_sqi(ecg_signal, fs):
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]
    # Check if rpeaks are detected
    if len(rpeaks) == 0:
        return -1  # Return 0 if no R-peaks are detected

    bs_sqi_values = []

    # Filter the ECG signal for baseline estimation
    baseline_signal = lowpass_filter(ecg_signal, fs, cutoff=1, order=6)

    for rpeak in rpeaks:
        # Step 1: Extract ECG segment around each R-peak (R-0.07s to R+0.08s)
        start_ecg = int(rpeak - 0.07 * fs)
        end_ecg = int(rpeak + 0.08 * fs)
        ecg_segment = ecg_signal[start_ecg:end_ecg]

        # Step 2: Calculate peak-to-peak amplitude of ECG segment (Ra)
        r_a = np.ptp(ecg_segment)

        # Step 3: Extract baseline segment around each R-peak (R-1s to R+1s)
        start_baseline = max(0, int(rpeak - 1 * fs))  # Ensure start is not negative
        end_baseline = min(
            len(baseline_signal), int(rpeak + 1 * fs)
        )  # Ensure end does not exceed signal length
        baseline_segment = baseline_signal[start_baseline:end_baseline]

        # Step 4: Calculate peak-to-peak amplitude of baseline segment (Ba)
        b_a = np.ptp(baseline_segment)

        # Step 5: Calculate Ra/Ba and store it
        if b_a != 0:
            bs_sqi_values.append(r_a / b_a)
        else:
            bs_sqi_values.append(0)

    # Step 6: Calculate the average bsSQI
    bs_sqi = np.mean(bs_sqi_values)
    return bs_sqi


def calculate_hf_sqi(ecg_signal, fs):
    """
    Calculate the hfSQI metric for an ECG signal.

    Parameters:
    - ecg_signal (array-like): The ECG signal data.
    - fs (int): Sampling rate of the ECG signal in Hz.

    Returns:
    - hf_sqi (float): The calculated hfSQI value.
    """
    # Step 1: Detect R-peaks
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]

    if len(rpeaks) == 0:
        return -1  # Return 0 if no R-peaks are detected

    # Step 2: Apply the high-pass filter using the difference equation
    y = ecg_signal[2:] - 2 * ecg_signal[1:-1] + ecg_signal[:-2]

    # Step 3: Compute the summation over every six points
    s = np.array([np.sum(np.abs(y[i : i + 6])) for i in range(len(y) - 5)])

    # Step 4: Calculate R_a_i (peak-to-peak amplitude of each QRS complex)
    R_a_i = []
    for r in rpeaks:
        start = max(0, r - int(0.05 * fs))
        end = min(len(ecg_signal), r + int(0.05 * fs))
        R_a_i.append(np.max(ecg_signal[start:end]) - np.min(ecg_signal[start:end]))

    # Step 5: Calculate H_i (mean of s(j) before each R-peak in the interval R - 0.28s to R - 0.05s)
    H_i = []
    for r in rpeaks:
        start = max(0, r - int(0.28 * fs))
        end = max(0, r - int(0.05 * fs))
        H_i.append(np.mean(s[start:end]))

    # Step 6: Compute hfSQI
    H_i = np.array(H_i)
    R_a_i = np.array(R_a_i)

    # Step 7: Handle division by zero
    nonzero_indices = H_i != 0
    if np.any(~nonzero_indices):
        print("Warning: Zero values found in H_i, excluding them from calculation.")

    R_a_i = R_a_i[nonzero_indices]
    H_i = H_i[nonzero_indices]

    hf_sqi = np.mean(R_a_i / H_i) if len(H_i) > 0 else -1

    return hf_sqi


def calculate_e_sqi(ecg_signal, fs):
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]
    # Check if rpeaks are detected
    if len(rpeaks) == 0:
        return -1  # Return 0 if no R-peaks are detected
    e_a = np.sum(ecg_signal**2)

    # Check if rpeaks are detected and Ea is non-zero
    if e_a == 0:
        return 0  # Return 0 if no R-peaks are detected or Ea is zero

    # Calculate the energy of each QRS segment
    qrs_energies = []
    for rpeak in rpeaks:
        # Extract ECG segment around each R-peak (R-0.07s to R+0.08s)
        start_ecg = int(rpeak - 0.07 * fs)
        end_ecg = int(rpeak + 0.08 * fs)
        ecg_segment = ecg_signal[start_ecg:end_ecg]

        # Calculate the energy of the ECG segment (Er_i)
        e_r_i = np.sum(ecg_segment**2)
        qrs_energies.append(e_r_i)

    # Calculate eSQI
    e_sqi = np.sum(qrs_energies) / e_a
    return e_sqi


def calculate_i_sqi(ecg_signal, fs):
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]
    if len(rpeaks) == 0:
        return -1  # Return 0 if no R-peaks are detected
    # Ensure there are at least 2 R-peaks to compute RR intervals
    if len(rpeaks) < 2:
        return 0  # Return 0 if not enough R-peaks

    # Step 1: Calculate RR intervals in seconds
    rr_intervals = (
        np.diff(rpeaks) / fs
    )  # Divide by sampling frequency to get time in seconds

    # Step 2: Sort RR intervals in ascending order
    sorted_rr_intervals = np.sort(rr_intervals)

    # Step 3: Compute 15th and 85th percentiles
    rr_15 = np.percentile(sorted_rr_intervals, 15)
    rr_85 = np.percentile(sorted_rr_intervals, 85)

    # Step 4: Calculate iSQI
    if rr_85 == 0:  # Avoid division by zero
        return 0
    i_sqi = rr_15 / rr_85

    return i_sqi


def calculate_rsd_sqi(ecg_signal, fs):
    """
    Calculate rsdSQI for the given ECG signal.

    Parameters:
    - ecg_signal: The ECG signal as a 1D numpy array.
    - rpeaks: List of R-peak indices detected in the ECG signal.
    - fs: Sampling frequency of the ECG signal.

    Returns:
    - rsd_sqi: The calculated rsdSQI value.
    """
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]
    if len(rpeaks) == 0:
        return -1  # Return -1 if no R-peaks are detected

    rsd_ratios = []

    for rpeak in rpeaks:
        # Define QRS window (R - 0.07s to R + 0.08s)
        start_qrs = max(0, int(rpeak - 0.07 * fs))
        end_qrs = min(len(ecg_signal), int(rpeak + 0.08 * fs))
        qrs_segment = ecg_signal[start_qrs:end_qrs]

        # Define analysis window (R - 0.2s to R + 0.2s)
        start_analysis = max(0, int(rpeak - 0.2 * fs))
        end_analysis = min(len(ecg_signal), int(rpeak + 0.2 * fs))
        analysis_segment = ecg_signal[start_analysis:end_analysis]

        # Compute STDs
        if len(qrs_segment) > 0 and len(analysis_segment) > 0:
            sigma_ri = np.std(qrs_segment)
            sigma_ai = np.std(analysis_segment)

            # Avoid division by zero
            if sigma_ai > 0:
                rsd_ratios.append(sigma_ri / (sigma_ai * 2))

    # Calculate rsdSQI as the mean of the ratios
    if len(rsd_ratios) > 0:
        rsd_sqi = np.mean(rsd_ratios)
    else:
        rsd_sqi = 0  # Return 0 if no valid segments are found

    return rsd_sqi


def calculate_pca_sqi(ecg_signal, fs):
    """
    Calculate pcaSQI for the given ECG signal.

    Parameters:
    - ecg_signal: The ECG signal as a 1D numpy array.
    - rpeaks: List of R-peak indices detected in the ECG signal.
    - fs: Sampling frequency of the ECG signal.

    Returns:
    - pca_sqi: The calculated pcaSQI value.
    """
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]
    if len(rpeaks) == 0:
        return -1  # Return -1 if no R-peaks are detected

    # Define segment length: 100ms before and after R-peak
    segment_samples = int(0.1 * fs)
    aligned_segments = []

    # Extract aligned ECG segments around each R-peak
    for rpeak in rpeaks:
        start = max(0, rpeak - segment_samples)
        end = min(len(ecg_signal), rpeak + segment_samples)
        segment = ecg_signal[start:end]

        # Ensure all segments are of equal length by padding if necessary
        if len(segment) == 2 * segment_samples:
            aligned_segments.append(segment)

    # Convert to matrix form for PCA (n_segments x n_samples_per_segment)
    if len(aligned_segments) < 2:  # PCA requires at least 2 segments
        return 0  # Return 0 if not enough valid segments

    aligned_matrix = np.array(aligned_segments)

    # Apply PCA
    pca = PCA()
    pca.fit(aligned_matrix)

    # Get eigenvalues (explained variance)
    eigenvalues = pca.explained_variance_

    # Calculate pcaSQI: sum of top 5 eigenvalues / sum of all eigenvalues
    sum_top5 = np.sum(eigenvalues[:5])  # Sum of first 5 eigenvalues
    sum_all = np.sum(eigenvalues)  # Sum of all eigenvalues

    if sum_all == 0:  # Avoid division by zero
        return 0

    pca_sqi = sum_top5 / sum_all
    return pca_sqi


def calculate_pica_sqi(ecg_signal, fs):
    """
    Calculate PiCASQI for the given ECG signal.

    Parameters:
    - ecg_signal: The ECG signal as a 1D numpy array.
    - fs: Sampling frequency of the ECG signal.

    Returns:
    - pica_sqi: The calculated PiCASQI value.
    """
    # Step 1: Detect R-peaks
    _, rpeaks_detected = nk.ecg_peaks(ecg_signal, sampling_rate=fs, show=False)
    rpeaks = rpeaks_detected["ECG_R_Peaks"]

    if len(rpeaks) < 2:
        return -1  # Return -1 if not enough R-peaks

    # Step 2: Segment ECG signal into beat-to-beat cycles
    segments = []
    for i in range(len(rpeaks) - 1):
        segment = ecg_signal[
            rpeaks[i] : rpeaks[i + 1]
        ]  # Segment between consecutive R-peaks
        segments.append(segment)

    if len(segments) < 2:
        return 0  # Return 0 if not enough valid segments

    # Step 3: Calculate covariance values
    cx_tau_list = []
    cx_0_list = []

    for i in range(len(segments) - 1):
        current_segment = segments[i]
        next_segment = segments[i + 1]

        # Ensure both segments are of equal length
        min_length = min(len(current_segment), len(next_segment))
        current_segment = current_segment[:min_length]
        next_segment = next_segment[:min_length]

        # Calculate τ-aligned covariance
        cx_tau = np.mean(current_segment * next_segment)  # Covariance for τ_t
        cx_0 = np.mean(current_segment**2)  # Variance of the current segment

        cx_tau_list.append(cx_tau)
        cx_0_list.append(cx_0)

    # Step 4: Compute averaged covariances
    cx_tau_avg = np.mean(cx_tau_list)
    cx_0_avg = np.mean(cx_0_list)

    if cx_0_avg == 0:  # Avoid division by zero
        return 0

    # Step 5: Calculate PiCASQI
    pica_sqi = cx_tau_avg / cx_0_avg

    return pica_sqi


def calculate_sampen_sqi(ecg_signal):
    sampen_sqi, _, _ = eh.SampEn(ecg_signal, m=2, r=0.15 * np.std(ecg_signal))
    return sampen_sqi[0]


def calculate_apen_sqi(ecg_signal):
    apen_sqi, _ = eh.ApEn(ecg_signal, m=2, r=0.15 * np.std(ecg_signal))
    return apen_sqi[0]


def calculate_fuzzen_sqi(ecg_signal):
    fuzzen_sqi, _, _ = eh.FuzzEn(ecg_signal, m=2, r=(0.15 * np.std(ecg_signal), 2))
    return fuzzen_sqi[0]


def calculate_disten_sqi(ecg_signal):
    disten, _ = eh.DistEn(ecg_signal)
    return disten


def calculate_specen_sqi(ecg_signal, fs, freqs):
    nyquist = fs // 2
    specen, _ = eh.SpecEn(
        ecg_signal, N=4 * fs, Freqs=(freqs[0] / nyquist, freqs[1] / nyquist)
    )
    return -1 if np.isnan(specen) else specen


def calculate_phasen_sqi(ecg_signal):
    phasen = eh.PhasEn(ecg_signal)
    return phasen


# folders = {'good': '../../dataset/An open-access arrhythmia database of wearable dynamic electrocardiogram/'
#                    'Signal_quality_database/good',
#            'medium': '../../dataset/An open-access arrhythmia database of wearable dynamic electrocardiogram/'
#                      'Signal_quality_database/medium',
#            'poor': '../../dataset/An open-access arrhythmia database of wearable dynamic electrocardiogram/'
#                    'Signal_quality_database/poor'}

# for i, folder in folders.items():
#     for filename in os.listdir(folder):
#         if filename.endswith('.mat'):
#             mat = sio.loadmat(os.path.join(folder, filename))
#             ecg_lead = mat['ECG'].squeeze().reshape(1, -1)
# s_sqi = calculate_s_sqi(ecg_lead.squeeze())
# print(i, s_sqi)

# k_sqi = calculate_k_sqi(ecg_lead.squeeze())
# print(i, k_sqi)

# sdn_sqi = calculate_sdn_sqi(ecg_lead, fs=400)
# print(i, sdn_sqi)

# pur_sqi = calculate_pur_sqi(ecg_lead, fs=400)
# print(i, pur_sqi)

# bas_sqi = calculate_bas_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, bas_sqi)

# p_sqi = calculate_p_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, p_sqi)

# lp_sqi = calculate_lp_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, lp_sqi)

# mp_sqi = calculate_mp_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, mp_sqi)

# hp_sqi = calculate_hp_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, hp_sqi)

# t_sqi = calculate_t_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, t_sqi)

# bs_sqi = calculate_bs_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, bs_sqi)

# hf_sqi = calculate_hf_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, hf_sqi)

# e_sqi = calculate_e_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, e_sqi)

# i_sqi = calculate_i_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, i_sqi)

# rsd_sqi = calculate_rsd_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, rsd_sqi)

# pca_sqi = calculate_pca_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, pca_sqi)

# pica_sqi = calculate_pica_sqi(ecg_lead.squeeze(), fs=400)
# print(i, filename, pica_sqi)

# sampen_sqi = calculate_sampen_sqi(ecg_lead.squeeze())
# print(i, filename, sampen_sqi)

# apen_sqi = calculate_apen_sqi(ecg_lead.squeeze())
# print(i, filename, apen_sqi)

# fuzzen_sqi = calculate_fuzzen_sqi(ecg_lead.squeeze())
# print(i, filename, fuzzen_sqi)

# disten_sqi = calculate_disten_sqi(ecg_lead.squeeze())
# print(i, filename, disten_sqi)

# specen_sqi = calculate_specen_sqi(ecg_lead.squeeze(), 400, [3, 30])
# print(i, filename, specen_sqi)

# phasen_sqi = calculate_phasen_sqi(ecg_lead.squeeze())
# print(i, filename, phasen_sqi)

# import matplotlib.pyplot as plt
# def plot_ecg(folder_, file_name):
#     mat = sio.loadmat(os.path.join(folders[folder_], file_name))
#     ecg_lead = mat['ECG'].squeeze().reshape(1, -1)
#     plt.plot(ecg_lead[0, :])
#     plt.grid()
#     plt.show()
# # plot_ecg('good', 'A_003.mat')
# # plot_ecg('good', 'A_057.mat')
# # plot_ecg('medium', 'B_044.mat')
# # plot_ecg('medium', 'B_077.mat')
# # plot_ecg('poor', 'C_080.mat')
# # plot_ecg('poor', 'C_091.mat')
