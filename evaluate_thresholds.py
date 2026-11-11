r"""
Threshold Evaluator & Diagram Generator for HierarchicalQRSClassifier (v4)
==========================================================================
Evaluates the model across multiple decision thresholds, identifies the optimal
F1 cut-off, computes ROC-AUC / PR-AUC / Sensitivity / Specificity, and renders
professional evaluation diagrams.

Features:
- Sweeps threshold from 0.0 to 1.0 (or continuous unique probability points).
- Renders 4 comprehensive subplots:
  1) ROC Curve with AUC score and Youden J / Best-F1 operating point.
  2) Precision-Recall (PR) Curve with AP / PR-AUC score.
  3) Metric curves vs. Threshold (F1, Precision, Sensitivity/Recall, Specificity).
  4) Confusion Matrix at optimal threshold.
- Saves calibrated threshold to <model>_threshold.json for automated inference.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

# Standard AAMI mappings consistent with training
AAMI_NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}


def match_labels(
    r_peaks: np.ndarray,
    ref_samples: np.ndarray,
    ref_symbols: np.ndarray,
    fs: float,
    tol_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches detected R-peaks to reference annotations within ±tol_samples.
    Returns (binary_ground_truth, match_mask).
    """
    r_peaks = np.asarray(r_peaks, dtype=np.int64)
    ref_samples = np.asarray(ref_samples, dtype=np.int64)
    ref_symbols = np.asarray(ref_symbols)

    if ref_symbols.dtype.kind in "USO":
        ref_binary = (~np.isin(ref_symbols, list(AAMI_NORMAL_SYMBOLS))).astype(np.int32)
    else:
        ref_binary = (ref_symbols != 0).astype(np.int32)

    n_ref = len(ref_samples)
    if n_ref == 0:
        raise ValueError("Reference annotations list is empty.")

    idx_right = np.searchsorted(ref_samples, r_peaks)
    idx_left = np.clip(idx_right - 1, 0, n_ref - 1)
    idx_right = np.clip(idx_right, 0, n_ref - 1)

    dist_left = np.abs(r_peaks - ref_samples[idx_left])
    dist_right = np.abs(r_peaks - ref_samples[idx_right])
    nearest_idx = np.where(dist_left <= dist_right, idx_left, idx_right)
    min_dist = np.minimum(dist_left, dist_right)

    matched = (min_dist <= tol_samples)

    # Resolve duplicates: ensure one reference beat is not claimed twice
    cand = np.flatnonzero(matched)
    if cand.size > 0:
        order = cand[np.lexsort((min_dist[cand], nearest_idx[cand]))]
        _, first = np.unique(nearest_idx[order], return_index=True)
        keep = np.zeros(len(r_peaks), dtype=bool)
        keep[order[first]] = True
        matched &= keep

    y_true = np.where(matched, ref_binary[nearest_idx], 0).astype(np.int32)
    return y_true, matched


def compute_metrics_sweep(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """Computes exact metric sweep across continuous unique threshold values."""
    y = y_true.astype(bool)
    scores = np.asarray(y_probs, dtype=np.float64)
    
    unique_thrs = np.unique(scores)
    # Include bounds
    grid_thrs = np.linspace(0.01, 0.99, 100)
    thrs = np.sort(np.unique(np.concatenate([unique_thrs, grid_thrs])))

    pos, neg = np.sort(scores[y]), np.sort(scores[~y])
    P, N = len(pos), len(neg)

    tp = len(pos) - np.searchsorted(pos, thrs, side="left")
    fp = len(neg) - np.searchsorted(neg, thrs, side="left")
    fn = P - tp
    tn = N - fp

    recall = tp / np.maximum(P, 1)                     # Sensitivity
    specificity = tn / np.maximum(N, 1)                # Specificity
    precision = tp / np.maximum(tp + fp, 1)
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-12)

    # Optimal threshold based on Maximum F1
    best_idx = int(np.argmax(f1))
    best_thr = float(thrs[best_idx])

    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    return {
        "thrs": thrs,
        "f1_curve": f1,
        "prec_curve": precision,
        "rec_curve": recall,
        "spec_curve": specificity,
        "best_thr": best_thr,
        "best_f1": float(f1[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_specificity": float(specificity[best_idx]),
        "tp": int(tp[best_idx]),
        "fp": int(fp[best_idx]),
        "fn": int(fn[best_idx]),
        "tn": int(tn[best_idx]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def plot_evaluation_diagrams(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metrics: Dict[str, Any],
    title: str = "Arrhythmia Abnormality Detection Performance",
    save_path: Optional[str] = "evaluation_report.png",
    show: bool = True
):
    """Generates a 4-panel diagnostic figure."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    # print(type(prec))
    # print(type(2))
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.28, wspace=0.25)

    # 1. ROC Curve
    ax_roc = axs[0, 0]
    ax_roc.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#999999", linestyle="--", lw=1.2, label="Random Guess")
    # Mark optimal operating point
    opt_fpr = 1.0 - metrics["best_specificity"]
    opt_tpr = metrics["best_recall"]
    ax_roc.scatter([opt_fpr], [opt_tpr], color="#d9534f", s=90, zorder=5,
                   label=f"Best F1 (Thr = {metrics['best_thr']:.3f})")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    ax_roc.legend(loc="lower right", fontsize=9)

    # 2. Precision-Recall Curve
    ax_pr = axs[0, 1]
    ax_pr.plot(rec, prec, color="#1b7837", lw=2.5, label=f"PR Curve (AP = {metrics['pr_auc']:.4f})")
    prevalence = float(np.mean(y_true))
    ax_pr.axhline(prevalence, color="#999999", linestyle="--", lw=1.2, label=f"Prevalence ({prevalence:.1%})")
    ax_pr.scatter([metrics["best_recall"]], [metrics["best_precision"]], color="#d9534f", s=90, zorder=5,
                  label=f"Best F1 ({metrics['best_f1']:.3f})")
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=10)
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    ax_pr.legend(loc="lower left", fontsize=9)

    # 3. Metrics vs Threshold Sweep
    ax_thr = axs[1, 0]
    thrs = metrics["thrs"]
    # print(type(thrs))
    # print(len(thrs))
    # print(len(metrics['rec_curve']))
    p=0
    for k in range(len(thrs)):
        if(metrics['rec_curve'][k]<0.95):
            print(f'first optimum rec threshold is {thrs[k]}')
            print(f'precision is{metrics['prec_curve'][k]*100}%')
            p=k
            break
    ax_thr.plot(thrs, metrics["f1_curve"], color="#756bb1", lw=2.2, label="F1 Score")
    ax_thr.plot(thrs, metrics["rec_curve"], color="#3182bd", lw=1.8, label="Sensitivity (Recall)")
    ax_thr.plot(thrs, metrics["spec_curve"], color="#31a354", lw=1.8, label="Specificity")
    ax_thr.plot(thrs, metrics["prec_curve"], color="#e6550d", lw=1.8, label="Precision")
    ax_thr.axvline(metrics["best_thr"], color="#d9534f", linestyle=":", lw=2,
                   label=f"Optimal Thr: {metrics['best_thr']:.3f}")
    ax_thr.set_title("Performance Metrics vs. Decision Threshold", fontsize=12, fontweight="bold")
    ax_thr.set_xlabel("Abnormality Probability Cutoff", fontsize=10)
    ax_thr.set_ylabel("Metric Value", fontsize=10)
    ax_thr.set_xlim([0.0, 1.0])
    ax_thr.set_ylim([0.0, 1.05])
    ax_thr.grid(True, linestyle=":", alpha=0.6)
    ax_thr.legend(loc="lower center", fontsize=9, ncol=2)

    # 4. Confusion Matrix at Best Threshold
    ax_cm = axs[1, 1]
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set_title(f"Confusion Matrix @ Thr = {metrics['best_thr']:.3f}", fontsize=12, fontweight="bold")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Normal (0)", "Abnormal (1)"])
    ax_cm.set_yticklabels(["Normal (0)", "Abnormal (1)"])
    ax_cm.set_xlabel("Predicted Label", fontsize=10)
    ax_cm.set_ylabel("True Ground Truth", fontsize=10)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > (cm.max() / 2) else "black"
            ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Diagram saved successfully to: {save_path}")

    if show:
        plt.show()

    return fig,p,thrs


def evaluate_model(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    ref_samples: np.ndarray,
    ref_symbols: np.ndarray,
    model_path: str = "ecg_hierarchical_v4.keras",
    tol_samples: int = 10,
    save_plot: Optional[str] = "evaluation_report.png",
    save_threshold: bool = True,
    show: bool = True
) -> Dict[str, Any]:
    """
    End-to-end evaluation entry point:
    1. Loads HierarchicalQRSClassifier.
    2. Runs inference on the full signal (with proper edge padding).
    3. Matches detected R-peaks to annotations within ±tol_samples.
    4. Computes ROC-AUC, PR-AUC, F1 sweep, and plots diagrams.
    5. Saves calibrated threshold to JSON.
    """
    from hierarchical_qrs_classifier import HierarchicalQRSClassifier

    print(f"Loading model from: {model_path} ...")
    clf = HierarchicalQRSClassifier(model_path=model_path)

    print("Running model inference...")
    res = clf.predict(raw_signal=signal, r_peaks=r_peaks, fs=fs)
    probs_all = res["abnormal_probability"]

    print("Matching detected peaks to annotations...")
    y_true, matched = match_labels(
        r_peaks=r_peaks,
        ref_samples=ref_samples,
        ref_symbols=ref_symbols,
        fs=fs,
        tol_samples=tol_samples
    )

    n_matched = int(np.sum(matched))
    n_unmatched = int(len(r_peaks) - n_matched)
    print(f"Total Detected Peaks: {len(r_peaks)} | Matched: {n_matched} | Unmatched: {n_unmatched}")

    if n_matched == 0:
        raise ValueError("No detected peak was matched within tolerance.")

    y_eval = y_true[matched]
    probs_eval = probs_all[matched]

    if len(np.unique(y_eval)) < 2:
        raise ValueError("Matched beats only contain a single class (need both Normal and Abnormal).")

    print("Computing metrics and threshold sweep...")
    metrics = compute_metrics_sweep(y_eval, probs_eval)
    metrics["n_detected"] = len(r_peaks)
    metrics["n_matched"] = n_matched
    metrics["n_unmatched"] = n_unmatched

    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLD EVALUATION SUMMARY")
    print("=" * 60)
    print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"PR-AUC (AP):    {metrics['pr_auc']:.4f}")
    print(f"Optimal Thr:    {metrics['best_thr']:.4f} (Max F1)")
    print(f"Max F1-Score:   {metrics['best_f1']:.4f}")
    print(f"Sensitivity:    {metrics['best_recall']:.4f}")
    print(f"Specificity:    {metrics['best_specificity']:.4f}")
    print(f"Precision:      {metrics['best_precision']:.4f}")
    print(f"Confusion:      TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
    print("=" * 60)

    # Save threshold
    if save_threshold:
        thr_file = os.path.splitext(model_path)[0] + "_threshold.json"
        with open(thr_file, "w") as f:
            json.dump({
                "threshold": metrics["best_thr"],
                "f1": metrics["best_f1"],
                "sensitivity": metrics["best_recall"],
                "specificity": metrics["best_specificity"],
                "roc_auc": metrics["roc_auc"],
                "method": "max_f1"
            }, f, indent=2)
        print(f"Saved optimal threshold configuration to: {thr_file}")

    # Plot diagrams
    k=plot_evaluation_diagrams(
        y_true=y_eval,
        y_probs=probs_eval,
        metrics=metrics,
        title=f"ECG Arrhythmia Evaluation Report (fs={fs:g} Hz)",
        save_path=save_plot,
        show=True
    )
    # print(k[2][k[1]],type(k),"qskhdfgqksdhf")
    return metrics,k[2][k[1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HierarchicalQRSClassifier thresholds and generate diagrams.")
    parser.add_argument("--record", default=r"C:\Users\Artin\Desktop\Dataset\mitdb\200", help="WFDB record path")
    parser.add_argument("--model_path", default="ecg_hierarchical_v4.keras", help="Path to trained .keras model")
    parser.add_argument("--save_plot", default="evaluation_report.png", help="Path to save output diagrams")
    parser.add_argument("--tol_samples", type=int, default=10, help="Matching tolerance in samples")
    args = parser.parse_args()

    import wfdb
    from functions import r_detection

    print(f"Loading record: {args.record} ...")
    rec = wfdb.rdrecord(args.record)
    ann = wfdb.rdann(args.record, "atr")
    sig = rec.p_signal[:, 0]
    fs = float(rec.fs)

    print("Running detector...")
    detected_peaks = np.sort(np.asarray(r_detection(sig=sig, fs=fs), dtype=np.int64))

    evaluate_model(
        signal=sig,
        r_peaks=detected_peaks,
        fs=fs,
        ref_samples=ann.sample,
        ref_symbols=ann.symbol,
        model_path=args.model_path,
        tol_samples=args.tol_samples,
        save_plot=args.save_plot,
        save_threshold=True,
        show=False
    )
