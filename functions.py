import os
import numpy as np 
import matplotlib.pyplot as plt 
import re
import wfdb
from qrs_classifier import evaluate_qrs_classification, QRSClassifier, _GLOBAL_ANNOTATION_CACHE

_LAST_ANNOTATION_CACHE = {}


def qrs_classification_assesment(
    signal,
    r_peaks,
    peak_samples,
    peak_labels=None,
    qrs_abnormalities=None,
    annotation_path=None,
    fs=180.0,
    tolerance=10,
    sample_offset=0,
    print_report=True,
):
    """
    Evaluates QRS classification TP, TN, FP, FN strictly on beats where detected
    and labeled R-peaks match within tolerance (default <= 10 samples).

    If peak_labels is not passed, it automatically resolves labels from the last
    annotation read by dat_reader, or from annotation_path/Pats folder.
    """
    if peak_labels is None:
        if annotation_path is None and "path" in _LAST_ANNOTATION_CACHE:
            annotation_path = _LAST_ANNOTATION_CACHE["path"]
        if "last" in _LAST_ANNOTATION_CACHE:
            cached_p, cached_l = _LAST_ANNOTATION_CACHE["last"]
            p_map = dict(zip(cached_p, cached_l))
            p_arr = np.asarray(peak_samples) if not isinstance(peak_samples, str) else None
            if p_arr is not None and len(p_arr) > 0:
                matches = sum(1 for p in p_arr[:50] if p in p_map)
                if matches > 0:
                    peak_labels = np.array([p_map.get(p, 0) for p in p_arr], dtype=np.int32)

    return evaluate_qrs_classification(
        ecg_signal=signal,
        detected_r_peaks=r_peaks,
        labeled_r_peaks=peak_samples,
        labeled_classes=peak_labels,
        predicted_labels=qrs_abnormalities,
        annotation_path=annotation_path,
        fs=fs,
        tolerance_samples=tolerance,
        sample_offset=sample_offset,
        print_report=print_report,
    )


# Spelling alias (supports both 'assesment' and standard 'assessment')
qrs_classification_assessment = qrs_classification_assesment


def detector_assesment(signal,r_peaks,peak_samples,ACCURACY_PROCESS,IMAGE_PROCESS,color=None,ax=None,fig=None) : 

    if ACCURACY_PROCESS:    
        good_detection=[]
        bad_detection=[]
        not_detected=[]
        self_detected=[]
        tolerance=10

        # ----- Self Detected -----
        arr=peak_samples
        for i in r_peaks : 
            check = np.any((arr >= i - tolerance) & (arr <= i + tolerance))
            seconde_check=np.any((arr >= i - 2*tolerance) & (arr <= i + 2*tolerance))
            if check : 
                good_detection.append(i)
            # elif seconde_check : 
            #     bad_detection.append(i)
            else:
                self_detected.append(i)
            print(len(good_detection),len(bad_detection),len(not_detected),len(peak_samples),len(self_detected),len(r_peaks))

        good_detection=np.array(good_detection)
        bad_detection=np.array(bad_detection)
        self_detected=np.array(self_detected)

        # np.savez_compressed('Detection_Results/Good.npz',good_detection,allow_pickle=True)
        # np.savez_compressed('Detection_Results/Bad.npz',bad_detection,allow_pickle=True)
        # np.savez_compressed('Detection_Results/Self_Detect.npz',self_detected,allow_pickle=True)

        # ------ Not Detected -----
        arr=r_peaks
        for i in peak_samples : 
            check = np.any((arr >= i - tolerance) & (arr <= i + tolerance))
            if check : 
                pass
            else :
                not_detected.append(i)
            print(len(good_detection),len(bad_detection),len(not_detected),len(peak_samples),len(self_detected),len(r_peaks))

        not_detected=np.array(not_detected)
        #np.savez_compressed('Detection_Results/Not_Detect.npz',not_detected,allow_pickle=True)
    


    # ------ Print Results ------
    # good=np.load('Detections/Accuracy/Good.npz',allow_pickle=True)['arr_0']
    # bad=np.load('Detections/Accuracy/Bad.npz',allow_pickle=True)['arr_0']
    # no=np.load('Detections/Accuracy/Not_Detect.npz',allow_pickle=True)['arr_0']
    # sel=np.load('Detections/Accuracy/Self_Detect.npz',allow_pickle=True)['arr_0']
    print(f'{len(good_detection)} samples detected correctly!')
    #print(f'{len(bad_detection)} samples detected badly!')
    print(f'{len(not_detected)} samples havent detected!')
    print(f'{len(r_peaks)-len(good_detection)-len(bad_detection)} samples detected but werent in the label') 
    return good_detection,not_detected,self_detected


    sel=self_detected
    no=not_detected
    pad=1000
    #image_Process(ax,fig,pad=700,self_detect=self_detected,not_detect=not_detected)
    #------- Plot Saver -----
    if IMAGE_PROCESS : 
        print("im here")
        arr=peak_samples
        for i in range(5,min(300,len(sel))):              # Self Detection
            center=sel[i]
            mask=(arr>center-pad)&(arr<center+pad)
            label=arr[mask]
            label=label-center+pad
            sig=signal[center-pad:center+pad]
            plt.plot(sig)
            start=(center-pad)/90
            end=(center+pad)/90
            print(i,end-start)
            for j in range(int(start)+1,int(end)-1):
                r=color[j][0]
                g=color[j][1]
                b=color[j][2]
                col=(r/255,g/255,b/255)
                k=j-int(start)-1
                init=int(start)+1-start
                init=int(init*180)
                p=np.arange(k*90+init,init+(k+1)*90)
                #print(end-start,j)
                plt.plot(p,sig[p],color=col)
            plt.scatter(pad,sig[pad],color='black',s=100)
            plt.scatter(label,sig[label],color='red')
            plt.savefig(f'Pics/Self_Detected/{i+1}.png')
            plt.close()


        arr=r_peaks
        for i in range(5,min(300,len(no))):         # No Detection
            center=no[i]
            mask=(arr>center-pad)&(arr<center+pad)
            label=arr[mask]
            label=label-center+pad
            sig=signal[center-pad:center+pad]
            plt.plot(sig)
            start=int((center-pad)/90)
            end=int((center+pad)/90)
            for j in range(int(start)+1,int(end)-1):
                r=color[j][0]
                g=color[j][1]
                b=color[j][2]
                col=(r/255,g/255,b/255)
                k=j-int(start)-1
                init=int(start)+1-start
                init=int(init*180)+1
                p=np.arange(k*90+init,init+(k+1)*90)
                #print(end-start,j)
                plt.plot(p,sig[p],color=col)
            plt.scatter(pad,sig[pad],color='red',s=100)
            plt.scatter(label,sig[label],color='black')
            plt.savefig(f'Pics/Not_Detected/{i+1}.png')
            plt.close()

def image_Process(ax,fig,pad,name,self_detect=None,not_detect=None):
    from pathlib import Path
    self_path=f'Pics/{name}/Self_Detected'
    not_path=f'Pics/{name}/Not_Detected'
    Path(self_path).mkdir(parents=True, exist_ok=True)
    for i in range(5,min(300,len(self_detect))):
        ax.set_xlim(self_detect[i]-pad,self_detect[i]+pad)
        fig.savefig(f'{self_path}/{i+1}.png', dpi=300, bbox_inches="tight")
    for i in range(5,min(300,len(not_detect))):
        ax.set_xlim(not_detect[i]-pad,not_detect[i]+pad)
        fig.savefig(f'{not_path}/{i+1}.png', dpi=300, bbox_inches="tight")


def cluster_maker_with_ann(peaks,thresh):
    finder_names=['nk_nk','nk_pantom','nk_nabian','nk_hamilton','nk_martine','nk_christove','nk_elgendi','nk_engzee','nk_khamis','nk_kalidas','nk_rodri','nk_emrich','bern_hamilton','bren_christov','bren_engzee','bren_swt','bern_pantom','bern_two_avg','Prominence']

    data = list(zip(peaks, finder_names))

    # Sort according to peak position
    data.sort(key=lambda x: x[0])

    groups = []
    current_group = [data[0]]

    for item in data[1:]:
        peak = item[0]

        if peak - current_group[-1][0] <= thresh:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]

    groups.append(current_group)
    return groups


def cluster_maker(peaks,thresh):
    peaks = np.sort(peaks)
    groups = []
    current_group = [peaks[0]]

    for peak in peaks[1:]:
        if peak - current_group[-1] <= thresh:
            current_group.append(peak)
        else:
            groups.append(current_group)
            current_group = [peak]

    groups.append(current_group)

    groups = [np.array(group) for group in groups]
    return groups


def read_holter_labels(
    file_path: str,
    upto_sample=None,
    start_sample: int = 0,
    return_peaks: bool = False,
    as_binary: bool = False,
):
    """
    Reads labeled beats of a Holter recording from a .txt file and returns the
    labels as a NumPy array (0, 1, 2, ...), filtered up to a determined sample.

    Clinical Label Mapping:
        0 = Normal
        1 = SVPB (Supraventricular Premature Beat / PAC)
        2 = VPB  (Ventricular Premature Beat / PVC)
        4 = Paced
        6 = AFib
        7 = R on T
        8 = Artifact
        9 = Unknown

    Parameters:
    -----------
    file_path : str
        Path to the Holter .txt annotation file (e.g., 'pat2.txt' or 'Pats/pat2/pat2.txt').
    upto_sample : int, np.ndarray, or None, optional
        Upper sample limit. Can be an integer sample index (e.g., 50000),
        or directly the truncated ECG signal array (len(signal) will be used).
        If None, all beats in the file are returned.
    start_sample : int, optional
        Lower sample limit (default 0). Beats before this sample are excluded.
    return_peaks : bool, optional
        If False (default), returns only the labels array `labels`.
        If True, returns a tuple `(peak_samples, labels)`.
    as_binary : bool, optional
        If True, maps 0 -> 0 (Normal) and all non-zero labels (1, 2, 4, 6, ...) -> 1 (Abnormal).
        If False (default), returns the raw clinical codes (0, 1, 2, ...).

    Returns:
    --------
    labels : np.ndarray
        1D NumPy array of beat labels (dtype=np.int32).
    (peak_samples, labels) : Tuple[np.ndarray, np.ndarray]
        If return_peaks=True, returns peak sample indices and corresponding labels.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Annotation file not found: {file_path}")

    # If user passed an array or list of signal samples directly
    if isinstance(upto_sample, (list, tuple, np.ndarray)):
        upto_sample = len(upto_sample)

    peak_samples = []
    labels = []
    in_mortara_table = False

    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            if "Beat    Beat     Beat" in line or "Beat    Beat    Beat" in line:
                in_mortara_table = True
                continue
            if in_mortara_table:
                if "Beat type" in line or "******" in line or "---" in line:
                    continue
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                    peak_samples.append(int(parts[1]))
                    labels.append(int(parts[2]))

    # Fallback: if not standard Mortara table, parse generic tabular txt (sample, label)
    if not in_mortara_table or len(labels) == 0:
        with open(file_path, "r", errors="ignore") as f:
            for line in f:
                line_str = line.strip().replace(",", " ").replace("\t", " ")
                parts = line_str.split()
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    peak_samples.append(int(parts[0]))
                    labels.append(int(parts[1]))
                elif len(parts) == 1 and parts[0].isdigit():
                    labels.append(int(parts[0]))

    peaks_arr = np.array(peak_samples, dtype=np.int64) if peak_samples else np.array([], dtype=np.int64)
    labels_arr = np.array(labels, dtype=np.int32)

    # Filter beats within [start_sample, upto_sample]
    if len(peaks_arr) == len(labels_arr) and len(peaks_arr) > 0:
        mask = peaks_arr >= start_sample
        if upto_sample is not None:
            mask = mask & (peaks_arr <= upto_sample)
        peaks_arr = peaks_arr[mask]
        labels_arr = labels_arr[mask]

    # Optional binary conversion: 0 = Normal, 1 = Abnormal
    if as_binary:
        labels_arr = (labels_arr != 0).astype(np.int32)

    # Cache for downstream automatic label resolution
    _LAST_ANNOTATION_CACHE["last"] = (peaks_arr, labels_arr)
    _LAST_ANNOTATION_CACHE["path"] = file_path
    try:
        _GLOBAL_ANNOTATION_CACHE["last"] = (peaks_arr, labels_arr)
        _GLOBAL_ANNOTATION_CACHE["path"] = file_path
    except Exception:
        pass

    if return_peaks:
        return peaks_arr, labels_arr
    return labels_arr


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

def wfdb_reader(folder_path,annotation=None):
    record=wfdb.rdrecord(folder_path)
    sig1=record.p_signal[:,0]
    sig2=record.p_signal[:,1]
    fs=record.fs
    if annotation :
        ann=wfdb.rdann
        samples=ann.sample
        return sig1,sig2,fs,samples
    else :
        return sig1,sig2,fs

def preprocess(signal,fs):
    from preprocessing import ECGPreprocessor
    ecg=ECGPreprocessor(sampling_rate=fs, baseline_sg_window_sec=1)
    signal,_=ecg.preprocess(signal,do_normalize=False)
    return(signal)


def align_rows_to_reference_no_fill(A, B_rows, threshold=20, fill_value=np.nan):
    """
    Aligns each row from B_rows to reference A.
    - NaNs in the input rows are ignored (dropped) – not turned into zeros.
    - For each a in A, finds the closest b in the row's valid numbers.
    - If |a - b| < threshold, keeps b; otherwise puts fill_value (default NaN).

    Parameters:
    A (array-like): 1D reference array.
    B_rows (list of lists or a single list): each row can have different length and contain NaNs.
    threshold (float): maximum allowed difference (default=20).
    fill_value (float): value to put when no match is found (default np.nan).

    Returns:
    np.ndarray: aligned matrix with shape (len(B_rows), len(A)).
    """
    A = np.asarray(A)
    
    # Guard against a flat list (treat as a single row)
    if not isinstance(B_rows, (list, tuple, np.ndarray)):
        raise TypeError("B_rows must be a list, tuple, or numpy array")
    if len(B_rows) > 0 and not isinstance(B_rows[0], (list, tuple, np.ndarray)):
        B_rows = [B_rows]

    aligned_rows = []

    for row in B_rows:
        # Convert to numpy array and drop NaNs (do NOT replace with zero)
        row_arr = np.asarray(row, dtype=float)
        valid = row_arr[~np.isnan(row_arr)]

        # If no valid numbers, fill the whole row with NaN
        if len(valid) == 0:
            aligned_rows.append(np.full(len(A), fill_value))
            continue

        # Sort for binary search
        row_sorted = np.sort(valid)

        # Initialize new row with fill_value (e.g., NaN)
        new_row = np.full(len(A), fill_value, dtype=float)

        for j, a in enumerate(A):
            idx = np.searchsorted(row_sorted, a)
            best_dist = threshold   # strictly less than threshold
            best_val = fill_value   # if no match, keep fill_value

            # Check left neighbour
            if idx > 0:
                d = abs(a - row_sorted[idx - 1])
                if d < best_dist:
                    best_dist = d
                    best_val = row_sorted[idx - 1]

            # Check right neighbour
            if idx < len(row_sorted):
                d = abs(a - row_sorted[idx])
                if d < best_dist:
                    best_dist = d
                    best_val = row_sorted[idx]

            # Only assign if a match was found (distance < threshold)
            if best_val is not fill_value:   # could be 0.0 if that's a real value; we need a better check
                # Better: check if best_dist was updated (i.e., < threshold)
                if best_dist < threshold:
                    new_row[j] = best_val

        aligned_rows.append(new_row)

    return np.vstack(aligned_rows)

def check_detected_label(aligned,peak_samples,tolerance):
    tol=5
    aligned_checker=aligned
    for i in range(len(aligned)):
        for j in range(len(aligned[i])):
            r=aligned[i][j]
            if (r==np.nan):
                aligned_checker[i][j]=0
            else: 
                if(np.any((peak_samples>r-tol)&(peak_samples<r+tol))):
                    aligned_checker[i][j]=1
                else:
                    aligned_checker[i][j]=0
    return aligned_checker.T

import math
def bin_color_finder(bin,window) : 
        blue=255
        bin_len=bin[1]-bin[0]
        interval=window[-1]['end_s']-window[-1]['start_s']
        overlap=window[-1]['start_s']-window[-2]['start_s']
        max_effect=int(255/(math.ceil(interval/overlap)))
        for i in window:
            start=i['start_s']
            if((abs(bin[0]-start)<interval-bin_len) & (start<=bin[0])):
                if(i['label']=='poor'):
                    blue-=max_effect
                elif(i['label']=='medium'):
                    blue-=(max_effect/3)
        return(blue)

def noise_detection(signal,fs,windows,peaks,name,interval=0.5,col_threshold=40,show_plot=False,plot=None):
    bin_number=int(len(signal)/(interval*fs))-1
    new_peaks=[]
    bin_colors=[]
    fig, ax = plt.subplots()
    for i in range(bin_number):
        bin=(i*interval,(i+1)*interval)
        blue=bin_color_finder(bin,windows)
        start=int(i*fs*interval)
        end=int((i+1)*fs*interval)
        p=np.arange(start,end)
        bin_colors.append((blue,start,end))
        plt.plot(p,signal[start:end],color=(0,0,blue/255))
        if (blue<col_threshold):
            new_peaks=peaks[(peaks<start)|(peaks>end)]
            # plt.plot(p,signal[start:end],color='pink')
    if show_plot:
        fig.savefig(f'Pat/pat1/MII/{name}')
        plt.show()
    return new_peaks,fig,ax
import numpy as np
import pywt
from scipy import signal, stats

TARGET_FS = 360.0  # Standard reference sampling rate for all datasets


def extract_robust_qrs_features(
    raw_window: np.ndarray, source_fs: float, target_fs: float = TARGET_FS
) -> dict:
    """Extracts features robust to varying sampling rates and arbitrary voltage gains.

    Parameters:
    -----------
    raw_window : np.ndarray
        ECG window from -100 ms to +150 ms at the original source_fs.
    source_fs : float
        The sampling rate of the current dataset (e.g., 250, 500, 1000).
    target_fs : float
        Universal target sampling rate to which all data is normalized (default: 360 Hz).
    """
    x = np.asarray(raw_window, dtype=np.float64)

    # =========================================================================
    # Step 1: Resample to Target Sampling Rate (Eliminates fs differences)
    # =========================================================================
    if source_fs != target_fs:
        num_target_samples = int(round(len(x) * (target_fs / source_fs)))
        # Polyphase filtering preserves waveform shape without edge artifacts
        x_resampled = signal.resample(x, num_target_samples)
    else:
        x_resampled = x

    # =========================================================================
    # Step 2: Z-Score Normalization (Eliminates arbitrary voltage/gain differences)
    # =========================================================================
    # This standardizes the scale: mean = 0, std = 1
    # Now raw ADC counts, microvolts (uV), or millivolts (mV) produce identical shapes!
    std_dev = np.std(x_resampled)
    if std_dev < 1e-6:
        std_dev = 1e-6  # Prevent division by zero on flat/dead signals
    x_norm = (x_resampled - np.mean(x_resampled)) / std_dev

    N = len(x_norm)
    features = {}

    # =========================================================================
    # Step 3: Scale-Invariant Morphological & Energy Features
    # =========================================================================
    # Mean-normalized energy and AUC (independent of sample count)
    features["norm_auc"] = float(np.sum(np.abs(x_norm)) / N)
    features["norm_energy"] = float(np.sum(x_norm**2) / N)

    # Relative amplitudes on normalized scale
    features["norm_peak_to_peak"] = float(np.max(x_norm) - np.min(x_norm))
    features["norm_max"] = float(np.max(x_norm))
    features["norm_min"] = float(np.min(x_norm))

    # =========================================================================
    # Step 4: Higher-Order Statistics (Naturally scale and shift invariant)
    # =========================================================================
    features["skewness"] = float(stats.skew(x_norm))
    features["kurtosis"] = float(stats.kurtosis(x_norm))

    # =========================================================================
    # Step 5: Slope & Fragmentation Features
    # =========================================================================
    # Since dx uses normalized amplitude and fixed target_fs, slopes are comparable
    dx = np.diff(x_norm) * target_fs
    features["max_positive_slope"] = float(np.max(dx))
    features["max_negative_slope"] = float(np.min(dx))
    features["mean_abs_slope"] = float(np.mean(np.abs(dx)))

    # Sign changes / Notching count
    features["num_notches_proxy"] = int(np.sum(np.diff(np.signbit(dx))))

    # =========================================================================
    # Step 6: Wavelet Sub-Bands (Now mapped to identical frequency brackets!)
    # =========================================================================
    # Because target_fs = 360 Hz everywhere:
    # D1 = 90-180 Hz, D2 = 45-90 Hz, D3 = 22.5-45 Hz, D4 = 11.25-22.5 Hz (QRS core)
    coeffs = pywt.wavedec(x_norm, "db4", level=4)
    approx_A4, detail_D4, detail_D3, detail_D2, detail_D1 = coeffs

    e_d1 = float(np.sum(detail_D1**2))
    e_d2 = float(np.sum(detail_D2**2))
    e_d3 = float(np.sum(detail_D3**2))
    e_d4 = float(np.sum(detail_D4**2))
    e_a4 = float(np.sum(approx_A4**2))
    total_energy = e_d1 + e_d2 + e_d3 + e_d4 + e_a4 + 1e-10

    features["dwt_energy_D1_ratio"] = e_d1 / total_energy
    features["dwt_energy_D2_ratio"] = e_d2 / total_energy
    features["dwt_energy_D3_ratio"] = e_d3 / total_energy
    features["dwt_energy_D4_ratio"] = e_d4 / total_energy
    features["dwt_energy_A4_ratio"] = e_a4 / total_energy
    features["hf_to_lf_ratio"] = float((e_d1 + e_d2) / (e_d4 + e_a4 + 1e-10))

    return features



import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats

TARGET_FS = 360.0


def build_dataset_features(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    template: np.ndarray,
    source_fs: float,
    labels: np.ndarray = None,
    lead: str = "II",
    patient_id: str = None,
    target_fs: float = TARGET_FS,
    local_window_size: int = 10,
) -> pd.DataFrame:
    """Builds a standardized feature table for all beats in an ECG signal,

    appending 'lead' and 'label' as the last two columns.

    Parameters:
    -----------
    ecg_signal : np.ndarray
        1D array of the ECG signal.
    r_peaks : np.ndarray
        1D array of detected R-peak sample indices.
    template : np.ndarray
        The patient's normalized baseline QRS template.
    source_fs : float
        Original sampling rate of the signal.
    labels : np.ndarray, optional
        Array of ground-truth labels (0 = Normal, 1 = Abnormal) for each R-peak.
        If None (e.g. during inference), NaN is assigned.
    lead : str
        Lead name (e.g. 'II', 'V1', 'MLII', 'V5').
    patient_id : str, optional
        Unique patient identifier (helps prevent data leakage during multi-patient training).
    target_fs : float
        Universal target sampling rate (default: 360.0 Hz).
    """
    n_peaks = len(r_peaks)
    if n_peaks == 0:
        return pd.DataFrame()

    total_len = len(ecg_signal)
    target_window_len = int(round(0.250 * target_fs))  # 250 ms
    pre_samples = int(round(0.100 * source_fs))
    post_samples = int(round(0.150 * source_fs))

    # Standardize template
    template_norm = (template - np.mean(template)) / (np.std(template) + 1e-6)
    if len(template_norm) != target_window_len:
        template_norm = signal.resample(template_norm, target_window_len)

    # 1. Precompute RR intervals in SECONDS
    if n_peaks > 1:
        rr_intervals = np.diff(r_peaks) / float(source_fs)
        global_median_rr = float(np.median(rr_intervals))
    else:
        rr_intervals = np.array([0.8])
        global_median_rr = 0.8

    dataset_rows = []

    for i, r_idx in enumerate(r_peaks):
        # ---------------- Timing Features ----------------
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
        local_rr = (
            float(np.median(local_slice))
            if len(local_slice) > 0
            else global_median_rr
        )
        if local_rr < 0.2:
            local_rr = 0.2

        pre_rr_ratio = pre_rr / local_rr
        post_rr_ratio = post_rr / local_rr
        pre_post_ratio = pre_rr / (post_rr + 1e-6)

        # ---------------- Windowing & Normalization ----------------
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

        win_resampled = signal.resample(raw_win, target_window_len)
        std_val = np.std(win_resampled)
        if std_val < 1e-6:
            std_val = 1e-6
        x_norm = (win_resampled - np.mean(win_resampled)) / std_val

        # ---------------- Template Comparison ----------------
        corr = np.corrcoef(x_norm, template_norm)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        rmse = np.sqrt(np.mean((x_norm - template_norm) ** 2))

        # ---------------- Morphology Features ----------------
        norm_auc = np.mean(np.abs(x_norm))
        norm_energy = np.mean(x_norm**2)
        kurt = float(stats.kurtosis(x_norm))
        skew = float(stats.skew(x_norm))

        dx = np.diff(x_norm) * target_fs
        max_slope = float(np.max(np.abs(dx)))
        mean_slope = float(np.mean(np.abs(dx)))
        notches = int(np.sum(np.diff(np.signbit(dx))))

        # Wavelet Decomposition (db4)
        coeffs = pywt.wavedec(x_norm, "db4", level=4)
        approx_A4, detail_D4, detail_D3, detail_D2, detail_D1 = coeffs
        e_d1 = float(np.sum(detail_D1**2))
        e_d2 = float(np.sum(detail_D2**2))
        e_d3 = float(np.sum(detail_D3**2))
        e_d4 = float(np.sum(detail_D4**2))
        e_a4 = float(np.sum(approx_A4**2))
        total_e = e_d1 + e_d2 + e_d3 + e_d4 + e_a4 + 1e-10

        # Build row
        row = {
            # Patient ID (if provided)
            **({"patient_id": str(patient_id)} if patient_id is not None else {}),
            # Timing
            "pre_rr": float(pre_rr),
            "post_rr": float(post_rr),
            "local_rr": float(local_rr),
            "pre_rr_ratio": float(pre_rr_ratio),
            "post_rr_ratio": float(post_rr_ratio),
            "pre_post_ratio": float(pre_post_ratio),
            "is_boundary_beat": is_boundary,
            # Template
            "template_corr": float(corr),
            "template_rmse": float(rmse),
            # Morphology
            "norm_auc": float(norm_auc),
            "norm_energy": float(norm_energy),
            "kurtosis": kurt,
            "skewness": skew,
            "max_abs_slope": max_slope,
            "mean_abs_slope": mean_slope,
            "num_notches": notches,
            "dwt_energy_D3_ratio": e_d3 / total_e,
            "dwt_energy_D4_ratio": e_d4 / total_e,
            "hf_to_lf_ratio": float((e_d1 + e_d2) / (e_d4 + e_a4 + 1e-10)),
            # LAST TWO COLUMNS:
            "lead": str(lead),
            "label": int(labels[i])
            if (labels is not None and i < len(labels))
            else np.nan,
        }
        dataset_rows.append(row)

    df = pd.DataFrame(dataset_rows)

    # Ensure 'lead' and 'label' are strictly the final two columns
    cols = [c for c in df.columns if c not in ["lead", "label"]] + [
        "lead",
        "label",
    ]
    return df[cols]




from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import xgboost as xgb


def train_qrs_model(
    df_combined: pd.DataFrame, patient_col: str = "patient_id"
):
    """Trains an XGBoost model on a concatenated DataFrame of multiple patients.

    Automatically handles the 'lead' categorical column and 'label' target.
    """
    # 1. Separate ground-truth labels
    if "label" not in df_combined.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    y = df_combined["label"].values.astype(int)

    # 2. Extract patient_ids for group-splitting (prevents patient data leakage)
    patient_ids = None
    cols_to_drop = ["label"]
    if patient_col in df_combined.columns:
        patient_ids = df_combined[patient_col].values
        cols_to_drop.append(patient_col)

    X_df = df_combined.drop(columns=cols_to_drop).copy()

    # 3. Format 'lead' as categorical for XGBoost's native categorical support
    if "lead" in X_df.columns:
        X_df["lead"] = X_df["lead"].astype("category")

    # 4. Patient-Wise Train/Validation Split
    if patient_ids is not None and len(np.unique(patient_ids)) > 1:
        splitter = StratifiedGroupKFold(n_splits=5)
        train_idx, val_idx = next(splitter.split(X_df, y, groups=patient_ids))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(splitter.split(X_df, y))

    X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 5. Cost-Sensitive Imbalance Weight
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / max(1, num_pos)

    print(
        f"Training Set: {num_neg} Normal, {num_pos} Abnormal (Ratio:"
        f" {scale_pos_weight:.2f}:1)"
    )

    # 6. Train XGBoost with enable_categorical=True
    clf = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Balances >90% normal data
        eval_metric="aucpr",  # PR-AUC for minority abnormal beats
        early_stopping_rounds=30,
        enable_categorical=True,  # Natively splits on 'lead'
        random_state=42,
    )

    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 7. Find Optimal Decision Threshold (Maximizing F1 of Abnormal Class)
    val_probs = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    f1_scores = (
        2
        * (precisions * recalls)
        / (precisions + recalls + 1e-10)[:-1]  # Drop last
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    pr_auc = average_precision_score(y_val, val_probs)

    print("\n" + "=" * 55)
    print(f"Optimal Decision Threshold : {best_threshold:.4f}")
    print(f"Validation PR-AUC          : {pr_auc:.4f}")
    print(f"Validation F1 (Abnormal)   : {best_f1:.4f}")
    print("=" * 55)

    # 8. Evaluation
    val_preds = (val_probs >= best_threshold).astype(int)
    print("\nClassification Report:")
    print(
        classification_report(
            y_val, val_preds, target_names=["Normal (0)", "Abnormal (1)"]
        )
    )

    cm = confusion_matrix(y_val, val_preds)
    print(f"Confusion Matrix:\nTN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")

    # 9. Feature Importance
    importance_df = pd.DataFrame({
        "Feature": X_df.columns,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return clf, best_threshold, importance_df


from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import xgboost as xgb


def train_qrs_model(
    df_combined: pd.DataFrame, patient_col: str = "patient_id"
):
    """Trains an XGBoost model on a concatenated DataFrame of multiple patients.

    Automatically handles the 'lead' categorical column and 'label' target.
    """
    # 1. Separate ground-truth labels
    if "label" not in df_combined.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    y = df_combined["label"].values.astype(int)

    # 2. Extract patient_ids for group-splitting (prevents patient data leakage)
    patient_ids = None
    cols_to_drop = ["label"]
    if patient_col in df_combined.columns:
        patient_ids = df_combined[patient_col].values
        cols_to_drop.append(patient_col)

    X_df = df_combined.drop(columns=cols_to_drop).copy()

    # 3. Format 'lead' as categorical for XGBoost's native categorical support
    if "lead" in X_df.columns:
        X_df["lead"] = X_df["lead"].astype("category")

    # 4. Patient-Wise Train/Validation Split
    if patient_ids is not None and len(np.unique(patient_ids)) > 1:
        splitter = StratifiedGroupKFold(n_splits=5)
        train_idx, val_idx = next(splitter.split(X_df, y, groups=patient_ids))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(splitter.split(X_df, y))

    X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 5. Cost-Sensitive Imbalance Weight
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / max(1, num_pos)

    print(
        f"Training Set: {num_neg} Normal, {num_pos} Abnormal (Ratio:"
        f" {scale_pos_weight:.2f}:1)"
    )

    # 6. Train XGBoost with enable_categorical=True
    clf = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Balances >90% normal data
        eval_metric="aucpr",  # PR-AUC for minority abnormal beats
        early_stopping_rounds=30,
        enable_categorical=True,  # Natively splits on 'lead'
        random_state=42,
    )

    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 7. Find Optimal Decision Threshold (Maximizing F1 of Abnormal Class)
    # -------------------- NEW (FIXED) --------------------
    val_probs = clf.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    # Slice precisions and recalls so they match the length of thresholds
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Now all arrays have the exact same shape!
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
    else:
        best_threshold = 0.5
        best_f1 = 0.0
    pr_auc = average_precision_score(y_val, val_probs)

    print("\n" + "=" * 55)
    print(f"Optimal Decision Threshold : {best_threshold:.4f}")
    print(f"Validation PR-AUC          : {pr_auc:.4f}")
    print(f"Validation F1 (Abnormal)   : {best_f1:.4f}")
    print("=" * 55)

    # 8. Evaluation
    val_preds = (val_probs >= best_threshold).astype(int)
    print("\nClassification Report:")
    print(
        classification_report(
            y_val, val_preds, target_names=["Normal (0)", "Abnormal (1)"]
        )
    )

    cm = confusion_matrix(y_val, val_preds)
    print(f"Confusion Matrix:\nTN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")

    # 9. Feature Importance
    importance_df = pd.DataFrame({
        "Feature": X_df.columns,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return clf, best_threshold, importance_df

import joblib


def save_qrs_pipeline(
    model,
    best_threshold: float,
    feature_columns: list,
    filepath: str = "qrs_model_pipeline.joblib",
):
    """Saves the trained XGBoost model along with its decision threshold and feature

    metadata into a single file.
    """
    pipeline_bundle = {
        "model": model,
        "threshold": float(best_threshold),
        "feature_columns": list(feature_columns),
        "target_fs": 360.0,
    }

    joblib.dump(pipeline_bundle, filepath)
    print(f" Pipeline successfully saved to: {filepath}")
    print(f" Saved Operating Threshold: {best_threshold:.4f}")




import joblib
import numpy as np
import pandas as pd


def predict_new_recording(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    template: np.ndarray,
    source_fs: float,
    lead: str = "II",
    model_path: str = "qrs_classifier_v1.joblib",
) -> pd.DataFrame:
    """Loads the trained model pipeline and classifies every beat in a new ECG recording.

    Returns a DataFrame with R-peak indices, predicted labels, and abnormal probabilities.
    """
    # -------------------------------------------------------------
    # 1. Load the Saved Pipeline Bundle
    # -------------------------------------------------------------
    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = bundle["threshold"]
    expected_features = bundle["feature_columns"]
    target_fs = bundle.get("target_fs", 360.0)

    # -------------------------------------------------------------
    # 2. Extract Features for the New Patient
    # -------------------------------------------------------------
    # Pass labels=None since we are predicting on unseen data
    df_features = build_dataset_features(
        ecg_signal=ecg_signal,
        r_peaks=r_peaks,
        template=template,
        source_fs=source_fs,
        labels=None,  # No ground truth during inference
        lead=lead,
        target_fs=target_fs,
    )

    # Prepare X (drop 'label' which is all NaNs)
    X = df_features[expected_features].copy()

    # Ensure 'lead' is categorical matching XGBoost requirements
    if "lead" in X.columns:
        X["lead"] = X["lead"].astype("category")

    # -------------------------------------------------------------
    # 3. Predict Probabilities & Apply Saved Threshold
    # -------------------------------------------------------------
    # Get probability of being Abnormal (Class 1)
    abnormal_probs = model.predict_proba(X)[:, 1]

    # Apply optimal threshold (NOT default 0.5)
    predictions = (abnormal_probs >= threshold).astype(int)

    # -------------------------------------------------------------
    # 4. Format Results
    # -------------------------------------------------------------
    results_df = pd.DataFrame({
        "r_peak_sample": r_peaks,
        "time_seconds": r_peaks / float(source_fs),
        "prediction": predictions,  # 0 = Normal, 1 = Abnormal
        "prediction_label": [
            "Abnormal" if p == 1 else "Normal" for p in predictions
        ],
        "abnormal_probability": abnormal_probs.round(4),
    })

    return results_df


# -----------------------------------------------------------------------------
# Modern Arbitrary-Length Deep Learning QRS Assessment Engine
# -----------------------------------------------------------------------------
try:
    from assess_qrs import assess_patient_qrs, plot_patient_qrs, QRSAssessmentResult, get_patient_sqi
    assess_patient = assess_patient_qrs
    qrs_assesment = assess_patient_qrs
    qrs_assessment = assess_patient_qrs
    plot_qrs = plot_patient_qrs
    patient_sqi = get_patient_sqi
except ImportError:
    pass




def r_detection(sig,fs):
    sig=preprocess(signal=sig,fs=fs)
    print("finished preprocessing")
    from prominence_delineator import ProminenceDelineator 
    PromDelineator = ProminenceDelineator(sampling_frequency=fs)
    # chunk it to 60 seconds segments (with overlap)
    chunks=int(len(sig)/(180*60))
    segment=int(len(sig)/chunks)
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
        r_peaks=PromDelineator.find_rpeaks(sig[start:end])
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

 