import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import wfdb

_LAST_ANNOTATION_CACHE = {}

def preprocess(signal,fs):
    from preprocessing import ECGPreprocessor
    ecg=ECGPreprocessor(sampling_rate=fs, baseline_sg_window_sec=1)
    signal,_=ecg.preprocess(signal,do_normalize=False)
    return(signal)



# def r_detection(sig: np.ndarray, fs: float) -> np.ndarray:
#     """Robust R-peak detection respecting arbitrary sampling rates.
#     Eliminates the hardcoded 180 Hz chunking bug that caused peak drift.
#     """
#     fs = float(fs)
#     from prominence_delineator import ProminenceDelineator

#     prom_delineator = ProminenceDelineator(sampling_frequency=fs)

#     chunk_samples = int(round(60.0 * fs))
#     if len(sig) <= chunk_samples:
#         peaks = prom_delineator.find_rpeaks(sig)
#         return np.asarray(peaks, dtype=np.int64)

#     overlap = int(round(1.0 * fs))
#     step = chunk_samples - overlap
#     total_len = len(sig)
#     peaks = []

#     for start in range(0, total_len, step):
#         end = min(total_len, start + chunk_samples)
#         if end - start < int(2.0 * fs):
#             break

#         chunk_sig = sig[start:end]
#         raw_peaks = prom_delineator.find_rpeaks(chunk_sig)
#         raw_peaks = np.asarray(raw_peaks, dtype=np.int64)

#         margin = 0 if start == 0 else overlap // 2
#         right_margin = (end - start) if end == total_len else (chunk_samples - overlap // 2)

#         valid = (raw_peaks >= margin) & (raw_peaks < right_margin)
#         for p in raw_peaks[valid]:
#             peaks.append(start + p)

#     return np.unique(np.asarray(peaks, dtype=np.int64))



def r_detection(sig,fs):
    signal=sig
    fs=int(fs)
    print(f'fs is {fs} and signal length is {len(sig)/(fs*3600)} hours')
    # signal=preprocess(signal=sig,fs=fs)
    # print("finished preprocessing")
    from prominence_delineator import ProminenceDelineator 
    PromDelineator = ProminenceDelineator(sampling_frequency=fs)
    # chunk it to 60 seconds segments (with overlap)
    chunks=int(len(signal)/(fs*60))
    print(f'chunks={chunks} len_sig={len(signal)}')
    segment=int(len(signal)/chunks)
    overlap=fs # 1 Second
    overlap_start=overlap
    overlap_end=overlap
    peaks=[] 
    for i in range(chunks):
        if(i==0):
            overlap_start=0
        elif(i==chunks-1):
            overlap_end=0
        else:
            overlap_start,overlap_end=overlap,overlap
        start=i*segment-overlap_start
        end=(i+1)*segment+overlap_end
        r_peaks=PromDelineator.find_rpeaks(signal[start:end])
        r_peaks=np.array(r_peaks)
        r_peaks=r_peaks.astype(int)
        mask= (r_peaks > overlap_start) & (r_peaks < segment+overlap_start)
        r_valid=r_peaks[mask]
        r_valid=r_valid+i*segment-overlap_start
        r_valid=r_valid.astype(int)     
        for j in r_valid:
            peaks.append(j)
        # print(f'processed chunk{i+1}')
    peaks=np.array(peaks)
    return peaks


def wfdb_reader(folder_path, annotation=None):
    record = wfdb.rdrecord(folder_path)
    sig1 = record.p_signal[:, 0]
    sig2 = record.p_signal[:, 1]
    fs = record.fs
    if annotation:
        ann = wfdb.rdann(folder_path, annotation)
        return sig1, sig2, fs, ann.sample
    return sig1, sig2, fs



def dat_reader(path, fs, name, data, return_labels=False):
    import os 
    sample_rate = fs
    state_path = path
    samples = []
    labels = []
    in_block = False

    with open(state_path, "r", errors="ignore") as f:
        for line in f:
            if "Beat    Beat     Beat" in line or "Beat    Beat    Beat" in line:
                in_block = True
                continue

            if not in_block:
                continue

            if "Beat type" in line or "******" in line:
                continue

            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s*$", line)
            if m:
                beat_num, beat_time, label = m.groups()
                samples.append(int(beat_time))
                labels.append(int(label))

    peak_samples = np.array(samples, dtype=np.int64)
    label_vals = np.array(labels, dtype=np.int64)

    # Cache globally for automatic label resolution
    _LAST_ANNOTATION_CACHE["last"] = (peak_samples, label_vals)
    _LAST_ANNOTATION_CACHE[name] = (peak_samples, label_vals)
    _LAST_ANNOTATION_CACHE["path"] = path
    try:
        _GLOBAL_ANNOTATION_CACHE["last"] = (peak_samples, label_vals)
        _GLOBAL_ANNOTATION_CACHE[name] = (peak_samples, label_vals)
        _GLOBAL_ANNOTATION_CACHE["path"] = path
    except Exception:
        pass

    signal = np.fromfile(f'Pats/{name}/{data}.dat', dtype="<i2").astype(np.float64)
    t = np.arange(signal.size) / sample_rate

    valid = (peak_samples >= 0) & (peak_samples < signal.size)
    peak_samples = peak_samples[valid]
    label_vals = label_vals[valid]
    peak_times = peak_samples / sample_rate

    start_s, end_s = 0, 100
    mask_sig = (t >= start_s) & (t <= end_s)
    mask_pk = (peak_times >= start_s) & (peak_times <= end_s)

    plt.figure(figsize=(16, 5))

    plt.plot(t[mask_sig], signal[mask_sig], color="black", linewidth=1.2, label="Lead II")
    plt.plot(peak_times[mask_pk], signal[peak_samples[mask_pk]], "ro", label="state-file beat markers")

    for x, lab in zip(peak_times[mask_pk], label_vals[mask_pk]):
        y = signal[int(x * sample_rate)]
        plt.text(x, y + 50, str(lab), color="red", fontsize=8, ha="center", va="bottom")

    plt.xlabel("Time (s)")
    plt.ylabel("ADC")
    plt.title("Lead II with state-file beat markers and labels")
    plt.legend()
    plt.tight_layout()
    plt.close()

    if return_labels:
        return signal, peak_samples, label_vals
    return signal, peak_samples