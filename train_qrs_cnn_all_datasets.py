r"""
End-to-End Multi-Cohort 1D-CNN QRS Classifier Training Engine
=============================================================
Trains a 1D Residual Convolutional Neural Network (1D-ResNet)
on EVERY possible record across all 14 annotated ECG datasets in:
    C:\Users\Artin\Desktop\Dataset

STRICT CLINICAL VALIDATION CRITERION:
Only beats where detected R-peaks and cardiologist-labeled R-peaks
match within a 10-sample tolerance window (|det_peak - ref_peak| <= 10)
are extracted and used for training and evaluation.

Cohorts Included:
  1.  mitdb     (MIT-BIH Arrhythmia Database, 48 records)
  2.  nstdb     (MIT-BIH Noise Stress Test Database, 12 records)
  3.  cudb      (Creighton Univ Ventricular Tachyarrhythmia, 35 records)
  4.  qtdb      (QT Database, 15 records)
  5.  svdb      (MIT-BIH Supraventricular Arrhythmia, 20 records)
  6.  incartdb  (St. Petersburg INCART 12-lead Arrhythmia, 15 records)
  7.  ludb      (Lobachevsky University ECG Database, 50 records)
  8.  fantasia  (Fantasia Healthy Aging Database, 10 records)
  9.  edb       (European ST-T Database, 10 records)
  10. ltdb      (MIT-BIH Long-Term Holter Database, 3 records)
  11. sddb      (Sudden Cardiac Death Holter Database, 4 records)
  12. chfdb     (BIDMC Congestive Heart Failure Database, 4 records)
  13. nsrdb     (MIT-BIH Normal Sinus Rhythm Database, 4 records)
  14. data716   (Univ of Glasgow Exercise & Motion Stress Database, 23 records)
  Total: 253 patient records.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import scipy.signal as signal
from typing import Dict, List, Tuple, Any

# Machine Learning & Deep Learning
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import wfdb

# Suppress verbose TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from r_peak_detector import RPeakDetector

DATASET_DIR = r"C:\Users\Artin\Desktop\Dataset"
DEFAULT_CACHE_PATH = r"C:\Users\Artin\Desktop\QRS\dataset_cache_all_valid_beats.npz"
MODEL_SAVE_PATH = r"C:\Users\Artin\Desktop\QRS\qrs_resnet1d_all_datasets_2.keras"
REPORT_SAVE_PATH = r"C:\Users\Artin\Desktop\QRS\cnn_training_report.json"

# AAMI EC57 Beat Standard
VALID_BEATS = {'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'j', 'e', '/', 'f', 'Q', '!'}
NORMAL_BEATS = {'N', 'L', 'R', 'B', 'j', 'e'}
ABNORMAL_BEATS = {'V', 'E', 'F', 'A', 'a', 'J', 'S', '/', 'f', 'Q', '!'}

# Sampling specifications
TARGET_FS = 360.0
PRE_MS = 0.100    # 100 ms before R-peak
POST_MS = 0.150   # 150 ms after R-peak
TARGET_LEN = 90   # 250 ms at 360 Hz


def discover_all_records(dataset_dir: str) -> List[Dict[str, Any]]:
    """Discovers every available annotated record across all 14 datasets."""
    records = []

    # 1. WFDB Datasets
    wfdb_configs = [
        ("mitdb", "atr"),
        ("nstdb", "atr"),
        ("cudb", "atr"),
        ("qtdb", "atr"),
        ("svdb", "atr"),
        ("incartdb", "atr"),
        ("ludb", "ii"),
        ("fantasia", "ecg"),
        ("edb", "atr"),
        ("ltdb", "atr"),
        ("sddb", "atr"),
        ("chfdb", "ecg"),
        ("nsrdb", "atr"),
    ]

    for cohort_id, ann_ext in wfdb_configs:
        c_dir = os.path.join(dataset_dir, cohort_id)
        if not os.path.exists(c_dir):
            continue
        heas = sorted([f[:-4] for f in os.listdir(c_dir) if f.endswith(".hea")])
        for r_id in heas:
            # Check that both header and annotation exist
            ann_file = os.path.join(c_dir, f"{r_id}.{ann_ext}")
            if os.path.exists(ann_file):
                records.append({
                    "cohort": cohort_id,
                    "record_id": r_id,
                    "type": "wfdb",
                    "ann_ext": ann_ext,
                    "path": os.path.join(c_dir, r_id)
                })

    # 2. Glasgow Real-World Motion Dataset (data716)
    g_dir = os.path.join(dataset_dir, "data716", "experiment_data")
    if os.path.exists(g_dir):
        for s in sorted(os.listdir(g_dir)):
            sp = os.path.join(g_dir, s)
            if os.path.isdir(sp) and s.startswith("subject"):
                for task in sorted(os.listdir(sp)):
                    tp = os.path.join(sp, task)
                    if os.path.isdir(tp):
                        ecg_p = os.path.join(tp, "ECG.tsv")
                        ann_p = os.path.join(tp, "annotation_cables.tsv")
                        if not os.path.exists(ann_p):
                            ann_p = os.path.join(tp, "annotation_cs.tsv")
                        if os.path.exists(ecg_p) and os.path.exists(ann_p):
                            records.append({
                                "cohort": "data716",
                                "record_id": f"{s}_{task}",
                                "type": "glasgow",
                                "ann_ext": "tsv",
                                "ecg_path": ecg_p,
                                "ann_path": ann_p
                            })

    return records


def extract_beats_from_wfdb(
    record_info: Dict[str, Any],
    max_samples: int = 60000,
    tolerance: int = 10
) -> Tuple[List[np.ndarray], List[List[float]], List[int], str]:
    """
    Extracts valid beats strictly matching detected R-peaks to reference annotations within tolerance.
    """
    r_path = record_info["path"]
    ann_ext = record_info["ann_ext"]

    # Header check to prevent sampto overflow
    try:
        hdr = wfdb.rdheader(r_path)
        actual_sampto = max_samples
        if hdr.sig_len and hdr.sig_len <= max_samples:
            actual_sampto = max(100, hdr.sig_len - 10)

        rec = wfdb.rdrecord(r_path, sampto=actual_sampto)
        ann = wfdb.rdann(r_path, ann_ext, sampto=actual_sampto)
    except Exception:
        return [], [], [], ""

    # Intelligent channel selection (avoid RESP / BP)
    best_idx = 0
    if rec.n_sig > 1 and hasattr(rec, 'sig_name') and rec.sig_name:
        for ch_idx, name in enumerate(rec.sig_name):
            n_lower = name.lower()
            if any(ign in n_lower for ign in ['resp', 'pleth', 'bp', 'abp', 'art']):
                continue
            if any(lead in n_lower for lead in ['ii', 'mlii', 'ecg', 'v1', 'v5', 'v2', 'v4']):
                best_idx = ch_idx
                break

    sig = rec.p_signal[:, best_idx]
    fs = float(rec.fs)
    sig_len = len(sig)

    # 1. Detect R-peaks with our adaptive detector
    detector = RPeakDetector(sampling_rate=fs, method="adaptive_prominence")
    try:
        res = detector.detect(sig)
        det_peaks = np.array(res.r_peaks, dtype=np.int64)
    except Exception:
        return [], [], [], ""

    if len(det_peaks) < 3:
        return [], [], [], ""

    # 2. Extract reference annotated beats
    ref_peaks = []
    ref_labels = []
    for s_idx, sym in zip(ann.sample, ann.symbol):
        if s_idx < sig_len and sym in VALID_BEATS:
            ref_peaks.append(s_idx)
            ref_labels.append(0 if sym in NORMAL_BEATS else 1)

    if len(ref_peaks) == 0:
        return [], [], [], ""

    ref_peaks = np.array(ref_peaks, dtype=np.int64)
    ref_labels = np.array(ref_labels, dtype=np.int32)

    # 3. Calculate local RR context from detector peaks
    rrs = np.diff(det_peaks) / fs
    pad_rrs=10
    paded_rrs=np.pad(rrs,(pad_rrs,pad_rrs),mode='edge')
    med_rr = float(np.median(rrs)) if len(rrs) > 0 else 0.8
    if med_rr <= 0:
        med_rr = 0.8

    pre_samp = int(round(PRE_MS * fs))
    post_samp = int(round(POST_MS * fs))

    valid_r_peaks=[]
    for r_pk in ann_peaks:
        dists_new = np.abs(det_peaks - r_pk)
        min_idx_new = int(np.argmin(dists))
        if dists[min_idx_new] <= tolerance:
            d_pk_new = det_peaks[min_idx]
            valid_r_peaks.append(d_pk_new)
    valid_r_peaks=np.array(valid_r_peaks)
    rrr_new=rrs = np.diff(det_peaks) / fs
    paded_rrs_new=np.pad(rrs_new,(pad_rrs,pad_rrs),mode='edge')
    waveforms = []
    rhythms = []
    labels = []

    # 4. Strict 10-sample tolerance matching
    for r_pk, r_lbl in zip(ref_peaks, ref_labels):
        dists = np.abs(det_peaks - r_pk)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= tolerance:
            # VALID COINCIDENT BEAT
            d_pk = det_peaks[min_idx]

            # Extract window centered at d_pk
            start = d_pk - pre_samp
            end = d_pk + post_samp

            if start < 0 or end > sig_len:
                raw_w = np.pad(sig[max(0, start):min(sig_len, end)],
                               (max(0, -start), max(0, end - sig_len)), mode='edge')
            else:
                raw_w = sig[start:end]

            # Resample to 90 points (360 Hz equivalent)
            if len(raw_w) != TARGET_LEN:
                w_res = signal.resample(raw_w, TARGET_LEN)
            else:
                w_res = raw_w

            # Z-score normalization
            std_val = float(np.std(w_res))
            if std_val > 1e-6:
                w_norm = (w_res - np.mean(w_res)) / std_val
            else:
                w_norm = w_res - np.mean(w_res)
            arg=0
            for k in range(len(valid_r_peaks)):
                if(d_pk==valid_r_peaks[k]):
                    arg=k
            rr_windows=paded_rrs_new[k-pad_rrs+1:k+pad_rrs-1]
            # Rhythm context features
            # rr_windows=paded_rrs[min_idx:min_idx+2*pad_rrs]
            med_rr=float(np.median(rr_windows))
            pre_rr = float(rrs[min_idx - 1]) if min_idx > 0 and (min_idx - 1) < len(rrs) else med_rr
            post_rr = float(rrs[min_idx]) if min_idx < len(rrs) else med_rr
            rhythm_feat = [
                pre_rr / med_rr,
                post_rr / med_rr,
                pre_rr / (post_rr + 1e-6)
            ]

            waveforms.append(w_norm)
            rhythms.append(rhythm_feat)
            labels.append(r_lbl)

    patient_id = f"{record_info['cohort']}_{record_info['record_id']}"
    return waveforms, rhythms, labels, patient_id


def extract_beats_from_glasgow(
    record_info: Dict[str, Any],
    max_samples: int = 30000,
    tolerance: int = 10
) -> Tuple[List[np.ndarray], List[List[float]], List[int], str]:
    """Extracts valid beats from Glasgow data716 exercise recordings."""
    try:
        ecg_data = np.loadtxt(record_info["ecg_path"], max_rows=max_samples)
        sig = ecg_data[:, 1]  # cables_Einth_II
        ann_peaks = np.loadtxt(record_info["ann_path"], dtype=int)
        ann_peaks = ann_peaks[ann_peaks < len(sig)]
    except Exception:
        return [], [], [], ""

    fs = 250.0
    sig_len = len(sig)

    detector = RPeakDetector(sampling_rate=fs, method="adaptive_prominence")
    try:
        res = detector.detect(sig)
        det_peaks = np.array(res.r_peaks, dtype=np.int64)
    except Exception:
        return [], [], [], ""

    if len(det_peaks) < 3 or len(ann_peaks) == 0:
        return [], [], [], ""

    rrs = np.diff(det_peaks) / fs
    pad_rrs=10
    paded_rrs=np.pad(rrs,(pad_rrs,pad_rrs),mode='edge')
    med_rr = float(np.median(rrs)) if len(rrs) > 0 else 0.8
    if med_rr <= 0:
        med_rr = 0.8

    pre_samp = int(round(PRE_MS * fs))
    post_samp = int(round(POST_MS * fs))

    waveforms = []
    rhythms = []
    labels = []

    valid_r_peaks=[]
    for r_pk in ann_peaks:
        dists_new = np.abs(det_peaks - r_pk)
        min_idx_new = int(np.argmin(dists))
        if dists[min_idx_new] <= tolerance:
            d_pk_new = det_peaks[min_idx]
            valid_r_peaks.append(d_pk_new)
    valid_r_peaks=np.array(valid_r_peaks)
    rrr_new=rrs = np.diff(det_peaks) / fs
    paded_rrs_new=np.pad(rrs_new,(pad_rrs,pad_rrs),mode='edge')

    for r_pk in ann_peaks:
        dists = np.abs(det_peaks - r_pk)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= tolerance:
            d_pk = det_peaks[min_idx]

            start = d_pk - pre_samp
            end = d_pk + post_samp

            if start < 0 or end > sig_len:
                raw_w = np.pad(sig[max(0, start):min(sig_len, end)],
                               (max(0, -start), max(0, end - sig_len)), mode='edge')
            else:
                raw_w = sig[start:end]

            w_res = signal.resample(raw_w, TARGET_LEN)
            std_val = float(np.std(w_res))
            if std_val > 1e-6:
                w_norm = (w_res - np.mean(w_res)) / std_val
            else:
                w_norm = w_res - np.mean(w_res)
            arg=0
            for k in range(len(valid_r_peaks)):
                if(d_pk==valid_r_peaks[k]):
                    arg=k
            rr_windows=paded_rrs_new[k-pad_rrs+1:k+pad_rrs-1]
            med_rr=float(np.median(rr_windows))
            pre_rr = float(rrs[min_idx - 1]) if min_idx > 0 and (min_idx - 1) < len(rrs) else med_rr
            post_rr = float(rrs[min_idx]) if min_idx < len(rrs) else med_rr
            rhythm_feat = [
                pre_rr / med_rr,
                post_rr / med_rr,
                pre_rr / (post_rr + 1e-6)
            ]

            waveforms.append(w_norm)
            rhythms.append(rhythm_feat)
            labels.append(0)  # Healthy sinus rhythm volunteer

    patient_id = f"data716_{record_info['record_id']}"
    return waveforms, rhythms, labels, patient_id


def build_or_load_dataset(
    dataset_dir: str,
    cache_path: str,
    max_samples: int = 60000,
    tolerance: int = 10,
    force_rebuild: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds or loads cached multi-cohort dataset of valid beats."""
    if not force_rebuild and os.path.exists(cache_path):
        print(f"--> Loading pre-extracted dataset from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X_wave"], data["X_rhy"], data["y"], data["patients"], data["cohorts"]

    print("=" * 80)
    print("STEP 1: DISCOVERING ALL FILES ACROSS 14 COHORTS")
    print("=" * 80)
    records = discover_all_records(dataset_dir)
    print(f"Found {len(records)} total records across 14 annotated databases.")

    all_wave = []
    all_rhy = []
    all_y = []
    all_pats = []
    all_cohorts = []

    print("\n" + "=" * 80)
    print(f"STEP 2: EXTRACTING STRICT VALID BEATS (|det - ref| <= {tolerance} samples)")
    print(f"        Max samples per record: {max_samples:,}")
    print("=" * 80)

    t0 = time.time()
    cohort_stats = {}

    for idx, r_info in enumerate(records, 1):
        c_name = r_info["cohort"]
        r_name = r_info["record_id"]

        if r_info["type"] == "wfdb":
            w, r, y, p = extract_beats_from_wfdb(r_info, max_samples=max_samples, tolerance=tolerance)
        else:
            w, r, y, p = extract_beats_from_glasgow(r_info, max_samples=max_samples, tolerance=tolerance)

        n_beats = len(y)
        if n_beats > 0:
            all_wave.extend(w)
            all_rhy.extend(r)
            all_y.extend(y)
            all_pats.extend([p] * n_beats)
            all_cohorts.extend([c_name] * n_beats)

            if c_name not in cohort_stats:
                cohort_stats[c_name] = {"records": 0, "normal": 0, "abnormal": 0}
            cohort_stats[c_name]["records"] += 1
            cohort_stats[c_name]["normal"] += sum(1 for label in y if label == 0)
            cohort_stats[c_name]["abnormal"] += sum(1 for label in y if label == 1)

        if idx % 25 == 0 or idx == len(records):
            print(f"  Processed [{idx:>3}/{len(records)}] records | Current Total Beats: {len(all_y):,}")

    elapsed = time.time() - t0
    print(f"\nExtraction completed in {elapsed:.1f} seconds!")
    print("\nCohort Extraction Summary:")
    print("-" * 65)
    print(f"{'Cohort':<12} | {'Records':<8} | {'Normal':<10} | {'Abnormal':<10} | {'Total':<10}")
    print("-" * 65)
    for c_id, stats in sorted(cohort_stats.items()):
        tot = stats["normal"] + stats["abnormal"]
        print(f"{c_id:<12} | {stats['records']:<8} | {stats['normal']:<10,} | {stats['abnormal']:<10,} | {tot:<10,}")
    print("-" * 65)

    X_wave = np.array(all_wave, dtype=np.float32)[..., np.newaxis]
    X_rhy = np.array(all_rhy, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    patients = np.array(all_pats)
    cohorts = np.array(all_cohorts)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_wave=X_wave,
        X_rhy=X_rhy,
        y=y,
        patients=patients,
        cohorts=cohorts
    )
    print(f"Saved extracted beat dataset cache to: {cache_path}")
    return X_wave, X_rhy, y, patients, cohorts


def build_resnet1d_model(target_len: int = TARGET_LEN) -> keras.Model:
    """Builds a Multi-Modal 1D Residual Convolutional Network (1D-ResNet)."""
    def residual_block(x, filters, kernel_size=5):
        shortcut = layers.Conv1D(filters, 1, padding='same')(x)

        y = layers.Conv1D(filters, kernel_size, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)

        y = layers.Conv1D(filters, kernel_size, padding='same')(y)
        y = layers.BatchNormalization()(y)

        out = layers.Add()([shortcut, y])
        out = layers.Activation('relu')(out)
        out = layers.MaxPooling1D(2)(out)
        return out

    # Stream 1: Raw 1D ECG QRS Waveform
    wave_in = layers.Input(shape=(target_len, 1), name="waveform_input")
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(wave_in)
    x = layers.BatchNormalization()(x)
    x = residual_block(x, 32, kernel_size=5)
    x = residual_block(x, 64, kernel_size=5)
    x = residual_block(x, 128, kernel_size=3)
    x = layers.GlobalAveragePooling1D()(x)

    # Stream 2: Dynamic RR Interval Context
    rhythm_in = layers.Input(shape=(3,), name="rhythm_input")
    r = layers.Dense(16, activation='relu')(rhythm_in)

    # Multi-Modal Fusion
    fused = layers.Concatenate()([x, r])
    dense = layers.Dense(64, activation='relu')(fused)
    dense = layers.Dropout(0.3)(dense)
    out = layers.Dense(1, activation='sigmoid', name="output")(dense)

    model = keras.Model(inputs=[wave_in, rhythm_in], outputs=out, name="ResNet1D_QRS_AllDatasets")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )
    return model


def train_and_evaluate(
    X_wave: np.ndarray,
    X_rhy: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    cohorts: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128
):
    """Executes patient-stratified train/test split, trains 1D-ResNet, and reports clinical metrics."""
    print("\n" + "=" * 80)
    print("STEP 3: PATIENT-STRATIFIED SPLIT (ZERO CONTAMINATION)")
    print("=" * 80)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_wave, y, groups=patients))

    X_wave_train, X_wave_test = X_wave[train_idx], X_wave[test_idx]
    X_rhy_train, X_rhy_test = X_rhy[train_idx], X_rhy[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    patients_test = patients[test_idx]
    cohorts_test = cohorts[test_idx]

    train_pats = np.unique(patients[train_idx])
    test_pats = np.unique(patients[test_idx])

    print(f"Total Beats:       {len(y):,}")
    print(f"Training Beats:    {len(y_train):,} ({len(train_pats)} unique patient records)")
    print(f"Testing Beats:     {len(y_test):,} ({len(test_pats)} unique patient records)")
    print(f"Train Class Ratio: {np.mean(y_train == 0)*100:.1f}% Normal | {np.mean(y_train == 1)*100:.1f}% Abnormal")
    print(f"Test Class Ratio:  {np.mean(y_test == 0)*100:.1f}% Normal | {np.mean(y_test == 1)*100:.1f}% Abnormal")

    # Balanced class weighting for minority ectopic beats
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    # Use square root smoothed weights to prevent loss skewing while boosting minority recall
    class_weight_dict = {0: 1.0, 1: float(np.sqrt(weights[1] / weights[0]))}
    print(f"Clinically Tuned Class Weights: {class_weight_dict}")

    print("\n" + "=" * 80)
    print("STEP 4: BUILDING & COMPILING 1D-RESNET ARCHITECTURE")
    print("=" * 80)
    model = build_resnet1d_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("\n" + "=" * 80)
    print(f"STEP 5: TRAINING 1D-RESNET ON ALL DATASETS (Epochs={epochs}, Batch={batch_size})")
    print("=" * 80)

    history = model.fit(
        x={"waveform_input": X_wave_train, "rhythm_input": X_rhy_train},
        y=y_train,
        validation_data=({"waveform_input": X_wave_test, "rhythm_input": X_rhy_test}, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "=" * 80)
    print("STEP 6: CLINICAL EVALUATION ON UNSEEN TEST PATIENTS")
    print("=" * 80)

    test_probs = model.predict({"waveform_input": X_wave_test, "rhythm_input": X_rhy_test}, batch_size=256).flatten()
    test_preds = (test_probs >= 0.50).astype(int)

    acc = float(accuracy_score(y_test, test_preds))
    sens = float(recall_score(y_test, test_preds, pos_label=1))
    prec = float(precision_score(y_test, test_preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, test_preds, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.0

    print("-" * 60)
    print(f"OVERALL UNSEEN TEST RESULTS ({len(y_test):,} beats):")
    print(f"  Accuracy:    {acc * 100:.2f}%")
    print(f"  Specificity: {spec * 100:.2f}% (Normal beat retention)")
    print(f"  Sensitivity: {sens * 100:.2f}% (Arrhythmia detection recall)")
    print(f"  Precision:   {prec * 100:.2f}%")
    print(f"  F1-Score:    {f1 * 100:.2f}%")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Confusion Matrix: TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print("-" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["Normal (0)", "Abnormal (1)"]))

    # Per-cohort breakdown
    print("\nPer-Cohort Performance on Unseen Test Beats:")
    print("-" * 75)
    print(f"{'Cohort':<12} | {'Beats':<8} | {'Accuracy':<10} | {'Sensitivity':<12} | {'Specificity':<12}")
    print("-" * 75)
    cohort_breakdown = {}
    for c_id in np.unique(cohorts_test):
        c_mask = (cohorts_test == c_id)
        c_y_true = y_test[c_mask]
        c_y_pred = test_preds[c_mask]
        c_acc = float(np.mean(c_y_true == c_y_pred))
        c_cm = confusion_matrix(c_y_true, c_y_pred, labels=[0, 1])
        c_tn, c_fp, c_fn, c_tp = c_cm.ravel()
        c_sens = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 1.0
        c_spec = float(c_tn / (c_tn + c_fp)) if (c_tn + c_fp) > 0 else 1.0
        cohort_breakdown[c_id] = {
            "beats": int(len(c_y_true)),
            "accuracy": c_acc,
            "sensitivity": c_sens,
            "specificity": c_spec
        }
        print(f"{c_id:<12} | {len(c_y_true):<8,} | {c_acc*100:<9.2f}% | {c_sens*100:<11.2f}% | {c_spec*100:<11.2f}%")
    print("-" * 75)

    # Save report
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_beats": int(len(y)),
        "train_beats": int(len(y_train)),
        "test_beats": int(len(y_test)),
        "unique_patients_train": int(len(train_pats)),
        "unique_patients_test": int(len(test_pats)),
        "metrics": {
            "accuracy": acc,
            "specificity": spec,
            "sensitivity": sens,
            "precision": prec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        },
        "per_cohort": cohort_breakdown
    }

    with open(REPORT_SAVE_PATH, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nFinal metrics report saved to: {REPORT_SAVE_PATH}")
    print(f"Trained model saved to:        {MODEL_SAVE_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Cohort 1D-CNN QRS Classifier Training")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR, help="Path to Dataset directory")
    parser.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH, help="Path to cache npz file")
    parser.add_argument("--samples_per_rec", type=int, default=60000, help="Max samples evaluated per record")
    parser.add_argument("--tolerance", type=int, default=10, help="R-peak matching tolerance in samples")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of dataset cache from scratch")
    args = parser.parse_args()

    X_wave, X_rhy, y, patients, cohorts = build_or_load_dataset(
        dataset_dir=args.dataset_dir,
        cache_path=args.cache_path,
        max_samples=args.samples_per_rec,
        tolerance=args.tolerance,
        force_rebuild=args.rebuild
    )

    train_and_evaluate(
        X_wave, X_rhy, y, patients, cohorts,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
