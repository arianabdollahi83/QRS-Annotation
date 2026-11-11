import os
import glob
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple

import wfdb
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functions import r_detection
from features import (bessel_bandpass_filter, extract_all_features,
                      BEAT_SAMPLES, N_NEIGHBORS, CLASS_NAMES)

AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,        # N
    'V': 1, 'E': 1,                                # V
    'A': 2, 'a': 2, 'J': 2, 'S': 2,                # S
    'F': 3,                                        # F
    # All remaining annotations default to Q (4)
}
N_CLASSES = 5

DS1_PATIENTS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"
]
DS2_PATIENTS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"
]


def load_and_filter_record(rec_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, 'atr')
    fs = float(record.fs)
    raw_sig = record.p_signal[:, 0]
    filt_sig = bessel_bandpass_filter(raw_sig, fs, lowcut=0.5, highcut=40.0)

    ann_peaks = np.asarray(annotation.sample, dtype=np.int64)
    symbols = np.asarray(annotation.symbol)
    classes = np.array([AAMI_MAP.get(s, 4) for s in symbols], dtype=np.int32)
    return filt_sig, ann_peaks, classes, fs


def process_record_training(rec_path: str, tol_samples: int = 20) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    filt_sig, ann_peaks, ann_classes, fs = load_and_filter_record(rec_path)
    
    # 1. Run your detector on filtered signal
    try:
        det_peaks = np.asarray(r_detection(sig=filt_sig, fs=fs), dtype=np.int64)
        det_peaks = np.sort(det_peaks)
    except Exception as e:
        print(f"Detector error on {rec_path}: {e}")
        det_peaks = np.array([], dtype=np.int64)

    n_ann = len(ann_peaks)
    if len(det_peaks) == 0 or n_ann == 0:
        return {}, {"total_ann": n_ann, "detected": len(det_peaks), "kept_beats": 0, "ignored_beats": n_ann}

    # 2. Match detected peaks with annotation peaks within ±tol_samples
    # For every detected peak, find closest annotation peak
    idx_right = np.searchsorted(ann_peaks, det_peaks)
    idx_left = np.clip(idx_right - 1, 0, n_ann - 1)
    idx_right = np.clip(idx_right, 0, n_ann - 1)

    dist_left = np.abs(det_peaks - ann_peaks[idx_left])
    dist_right = np.abs(det_peaks - ann_peaks[idx_right])
    
    nearest_ann_idx = np.where(dist_left <= dist_right, idx_left, idx_right)
    min_dist = np.minimum(dist_left, dist_right)

    # A detected peak is valid only if within <= tol_samples of an annotation
    is_valid_det = (min_dist <= tol_samples)

    # Deduplicate: if two detected peaks point to same annotation, keep closest
    matched_ann_map = {}
    for d_i, (valid, ann_i, dist) in enumerate(zip(is_valid_det, nearest_ann_idx, min_dist)):
        if valid:
            if ann_i not in matched_ann_map or dist < matched_ann_map[ann_i][1]:
                matched_ann_map[ann_i] = (d_i, dist)

    # Re-flag validity after 1-to-1 matching
    valid_flag = np.zeros(len(det_peaks), dtype=bool)
    det_labels = np.full(len(det_peaks), -1, dtype=np.int32)
    for ann_i, (d_i, _) in matched_ann_map.items():
        valid_flag[d_i] = True
        det_labels[d_i] = ann_classes[ann_i]

    # 3. Find beats where 20 before and 20 after are strictly valid
    # Sequence length required: 20 + 1 + 20 = 41 valid consecutive detected beats
    n_det = len(det_peaks)
    valid_int = valid_flag.astype(np.int32)
    
    # Cumulative sum to check if all 41 beats in [i - 20, i + 20] are valid (sum == 41)
    csum = np.pad(np.cumsum(valid_int), (1, 0), mode='constant')
    
    kept_det_indices = []
    for i in range(N_NEIGHBORS, n_det - N_NEIGHBORS):
        window_sum = csum[i + N_NEIGHBORS + 1] - csum[i - N_NEIGHBORS]
        if window_sum == (2 * N_NEIGHBORS + 1):  # Exactly 41 consecutive valid beats
            kept_det_indices.append(i)

    kept_det_indices = np.array(kept_det_indices, dtype=np.int64)

    # 4. Feature extraction on all detected peaks
    # Using 'inference' mode to extract feature rows across all detected peaks,
    # then selecting strictly the verified 41-beat sequence centers.
    feats = extract_all_features(filt_sig, det_peaks, fs, mode="inference")

    stats = {
        "total_ann": n_ann,
        "detected": n_det,
        "kept_beats": len(kept_det_indices),
        "ignored_beats": n_ann - len(kept_det_indices)
    }

    if len(kept_det_indices) == 0:
        return {}, stats

    final_feats = {
        "beat": feats["beat"][kept_det_indices],
        "diff": feats["diff"][kept_det_indices],
        "p_wave": feats["p_wave"][kept_det_indices],
        "rr_raw": feats["rr_raw"][kept_det_indices],
        "rr_norm": feats["rr_norm"][kept_det_indices],
        "rr_z": feats["rr_z"][kept_det_indices],
        "labels": det_labels[kept_det_indices]
    }
    return final_feats, stats


def build_hierarchical_model() -> keras.Model:
    beat_in = layers.Input(shape=(BEAT_SAMPLES, 1), name="beat_input")
    diff_in = layers.Input(shape=(BEAT_SAMPLES, 1), name="diff_input")
    morph_cat = layers.Concatenate(axis=-1)([beat_in, diff_in])

    x = layers.Conv1D(32, kernel_size=7, padding="same", use_bias=False)(morph_cat)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    res = x
    x = layers.Conv1D(32, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    res = layers.Conv1D(64, 1, padding="same", use_bias=False)(x)
    res = layers.BatchNormalization()(res)
    x = layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    morph_vec = layers.GlobalAveragePooling1D()(x)

    rr_raw_in = layers.Input(shape=(40, 1), name="rr_raw_input")
    rr_norm_in = layers.Input(shape=(40, 1), name="rr_norm_input")
    rr_z_in = layers.Input(shape=(40, 1), name="rr_z_input")
    rr_all = layers.Concatenate(axis=-1)([rr_raw_in, rr_norm_in, rr_z_in])

    r = layers.Conv1D(32, 3, padding="same", activation="relu")(rr_all)
    r = layers.Conv1D(64, 3, padding="same", activation="relu")(r)
    r_vec = layers.GlobalMaxPooling1D()(r)

    p_in = layers.Input(shape=(3,), name="p_wave_input")
    p_vec = layers.Dense(16, activation="relu")(p_in)

    fused = layers.Concatenate()([morph_vec, r_vec, p_vec])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    fused = layers.Dense(64, activation="relu")(fused)

    out_abnormal = layers.Dense(1, activation="sigmoid", name="abnormal_output")(fused)
    out_classes = layers.Dense(N_CLASSES, activation="softmax", name="class_output")(fused)

    model = keras.Model(
        inputs=[beat_in, diff_in, rr_raw_in, rr_norm_in, rr_z_in, p_in],
        outputs=[out_abnormal, out_classes],
        name="Hierarchical_ECG_V4"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "abnormal_output": "binary_crossentropy",
            "class_output": "sparse_categorical_crossentropy"
        },
        loss_weights={"abnormal_output": 1.0, "class_output": 0.5},
        metrics={"abnormal_output": "accuracy", "class_output": "accuracy"}
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=r"C:\Users\Artin\Desktop\Dataset\mitdb")
    parser.add_argument("--save_path", default="ecg_hierarchical_v4.keras")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1: PREPROCESSING & FEATURE EXTRACTION (INTER-PATIENT SPLIT)")
    print("=" * 80)

    train_store = {k: [] for k in ["beat", "diff", "p_wave", "rr_raw", "rr_norm", "rr_z", "labels"]}
    val_store = {k: [] for k in ["beat", "diff", "p_wave", "rr_raw", "rr_norm", "rr_z", "labels"]}

    for pat in DS1_PATIENTS:
        rec_p = os.path.join(args.dataset_dir, pat)
        if not os.path.exists(rec_p + ".hea"):
            continue
        feats, stats = process_record_training(rec_p, tol_samples=10)
        print(f"Record {pat} [TRAIN]: Ann: {stats['total_ann']} | Det: {stats['detected']} | "
              f"Kept (41 consecutive valid): {stats['kept_beats']} | Ignored: {stats['ignored_beats']}")
        if stats["kept_beats"] > 0:
            for k in train_store:
                train_store[k].append(feats[k])

    for pat in DS2_PATIENTS:
        rec_p = os.path.join(args.dataset_dir, pat)
        if not os.path.exists(rec_p + ".hea"):
            continue
        filt_sig, ref_peaks, ref_classes, fs = load_and_filter_record(rec_p)
        
        # In validation, test with padded boundaries
        feats = extract_all_features(filt_sig, ref_peaks, fs, mode="inference")
        print(f"Record {pat} [VAL]: Extracted {len(ref_peaks)} beats (with padding on boundary beats).")
        val_store["beat"].append(feats["beat"])
        val_store["diff"].append(feats["diff"])
        val_store["p_wave"].append(feats["p_wave"])
        val_store["rr_raw"].append(feats["rr_raw"])
        val_store["rr_norm"].append(feats["rr_norm"])
        val_store["rr_z"].append(feats["rr_z"])
        val_store["labels"].append(ref_classes)

    X_train = {
        "beat_input": np.concatenate(train_store["beat"])[..., None],
        "diff_input": np.concatenate(train_store["diff"])[..., None],
        "rr_raw_input": np.concatenate(train_store["rr_raw"])[..., None],
        "rr_norm_input": np.concatenate(train_store["rr_norm"])[..., None],
        "rr_z_input": np.concatenate(train_store["rr_z"])[..., None],
        "p_wave_input": np.concatenate(train_store["p_wave"])
    }
    y_train_cls = np.concatenate(train_store["labels"])
    y_train_abn = (y_train_cls != 0).astype(np.float32)

    X_val = {
        "beat_input": np.concatenate(val_store["beat"])[..., None],
        "diff_input": np.concatenate(val_store["diff"])[..., None],
        "rr_raw_input": np.concatenate(val_store["rr_raw"])[..., None],
        "rr_norm_input": np.concatenate(val_store["rr_norm"])[..., None],
        "rr_z_input": np.concatenate(val_store["rr_z"])[..., None],
        "p_wave_input": np.concatenate(val_store["p_wave"])
    }
    y_val_cls = np.concatenate(val_store["labels"])
    y_val_abn = (y_val_cls != 0).astype(np.float32)

    print("\nBuilding model...")
    model = build_hierarchical_model()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_class_output_loss", mode="min", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_class_output_loss", mode="min", factor=0.5, patience=3)
    ]

    print("Training model...")
    model.fit(
        X_train,
        {"abnormal_output": y_train_abn, "class_output": y_train_cls},
        validation_data=(X_val, {"abnormal_output": y_val_abn, "class_output": y_val_cls}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    preds_abn, preds_cls = model.predict(X_val, batch_size=256)
    preds_cls_labels = np.argmax(preds_cls, axis=-1)

    auc = roc_auc_score(y_val_abn, preds_abn.ravel())
    f1_macro = f1_score(y_val_cls, preds_cls_labels, average="macro")

    print("\n" + "=" * 80)
    print("VALIDATION REPORT (DS2 - UNSEEN PATIENTS)")
    print("=" * 80)
    print(f"ROC-AUC (Abnormal Binary): {auc:.4f}")
    print(f"Macro F1-Score (5-Class): {f1_macro:.4f}")
    print("\nPer-Class Detailed Report:")
    print(classification_report(y_val_cls, preds_cls_labels, target_names=CLASS_NAMES, digits=4))

    model.save(args.save_path)
    print(f"Model saved successfully to: {args.save_path}")


if __name__ == "__main__":
    main()
