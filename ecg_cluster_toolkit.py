"""
ECG beat clustering toolkit
===========================
Four clustering methods that can be used alone or chained:

    "hdbscan"      density clustering on (PCA-reduced) beat shapes
    "autoencoder"  train a small autoencoder, then HDBSCAN on the latent codes
    "corr"         sequential: join the cluster whose median has Pearson corr > thr
    "rms"          sequential: join the cluster whose median has
                   (energy of error) / min(energy of the two beats) < thr

Chaining = each step re-clusters INSIDE every cluster produced by the previous step.
  e.g. steps=[("hdbscan", {"min_cluster_size": 5}), ("corr", {"thr": 0.9})]

Label -1 means noise (only HDBSCAN / autoencoder produce it).

Main entry points
-----------------
run_clustering(sig, fs, rpeaks, out_dir, steps)      -> clusters, saves plots + clusters.npz
refine_from_npz(npz_path, sig, out_dir, steps)       -> loads a saved .npz, refines further
compare_combinations(sig, fs, rpeaks, out_root, combos)  -> runs many combos, prints a table
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Beat extraction
# ----------------------------------------------------------------------
def extract_beats(sig, rpeaks, fs, pre=0.250, post=0.300, remove_mean=True):
    """Cut a window [R-pre, R+post) around every R-peak.
    Returns beats (n, L) and keep_idx (positions in rpeaks that were kept;
    beats too close to the signal edges are dropped)."""
    sig = np.asarray(sig, float)
    rpeaks = np.asarray(rpeaks, int)
    npre, npost = int(round(pre * fs)), int(round(post * fs))
    beats, keep_idx = [], []
    for i, r in enumerate(rpeaks):
        s, e = r - npre, r + npost
        if s < 0 or e > len(sig):
            continue
        b = sig[s:e].copy()
        if remove_mean:
            b -= b.mean()
        beats.append(b)
        keep_idx.append(i)
    L = npre + npost
    beats = np.stack(beats) if beats else np.empty((0, L))
    return beats, np.array(keep_idx, int)


# ----------------------------------------------------------------------
# Sequential methods (RMS / correlation)
# ----------------------------------------------------------------------
def _score_rms(b, M):
    """Energy of error / min energy of the two beats. Lower = more similar."""
    Eb = b @ b
    Ee = ((M - b) ** 2).sum(1)
    Emin = np.minimum(Eb, (M ** 2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Emin > 0, Ee / Emin, np.inf)


def _score_corr(b, M):
    """1 - Pearson correlation. Lower = more similar."""
    bc = b - b.mean()
    Mc = M - M.mean(1, keepdims=True)
    den = np.linalg.norm(bc) * np.linalg.norm(Mc, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(den > 0, (Mc @ bc) / den, -1.0)
    return 1.0 - c


def _sequential(beats, score_fn, max_score, pick="best", median_cap=500):
    """New beat is compared to the MEDIAN of every existing cluster.
    Joins if score < max_score (best one, or first one if pick='first'),
    otherwise starts a new cluster. Cluster median is updated after each join.
    median_cap: median is computed from the last `median_cap` members (speed)."""
    n, L = beats.shape
    labels = np.empty(n, int)
    if n == 0:
        return labels
    M = np.empty((16, L))
    members, K = [], 0
    for i, b in enumerate(beats):
        best = -1
        if K:
            s = score_fn(b, M[:K])
            ok = np.where(s < max_score)[0]
            if len(ok):
                best = ok[0] if pick == "first" else ok[np.argmin(s[ok])]
        if best == -1:
            if K == len(M):
                M = np.vstack([M, np.empty_like(M)])
            M[K] = b
            members.append([i])
            labels[i] = K
            K += 1
        else:
            members[best].append(i)
            M[best] = np.median(beats[members[best][-median_cap:]], axis=0)
            labels[i] = best
    return labels


def cluster_rms(beats, thr=0.05, pick="best", median_cap=500):
    """Beat joins a cluster if Ee / min(E_beat, E_median) < thr."""
    return _sequential(beats, _score_rms, thr, pick, median_cap)


def cluster_corr(beats, thr=0.90, pick="best", median_cap=500):
    """Beat joins a cluster if Pearson corr with the cluster median > thr."""
    return _sequential(beats, _score_corr, 1.0 - thr, pick, median_cap)


# ----------------------------------------------------------------------
# HDBSCAN / autoencoder
# ----------------------------------------------------------------------
def _hdbscan(X, min_cluster_size, min_samples=None):
    try:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size,
                       min_samples=min_samples).fit_predict(X)
    except ImportError:
        import hdbscan  # pip install hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples).fit_predict(X)


def cluster_hdbscan(beats, min_cluster_size=5, min_samples=None,
                    n_components=0.95, normalize=False):
    """HDBSCAN on PCA-reduced beats.
    n_components: float in (0,1) = keep as many PCs as needed to explain that
                  fraction of the variance (default 0.95); int = fixed number of PCs;
                  None = no PCA (raw samples).
    normalize=True z-scores each beat first (amplitude-independent, shape only)."""
    n = len(beats)
    if n <= max(min_cluster_size, 2):
        return np.full(n, -1)
    X = np.asarray(beats, float)
    if normalize:
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    if n_components and n_components < min(X.shape):
        from sklearn.decomposition import PCA
        X = PCA(n_components=n_components, svd_solver="full",
                random_state=0).fit_transform(X)
    return _hdbscan(X, min_cluster_size, min_samples)


def cluster_autoencoder(beats, min_cluster_size=5, latent_dim=8, epochs=60,
                        lr=1e-3, batch=256, seed=0, min_samples=None):
    """Train an MLP autoencoder on the beats, then HDBSCAN on the latent codes."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("autoencoder needs PyTorch: pip install torch") from e

    n, L = beats.shape
    if n <= max(min_cluster_size, 2):
        return np.full(n, -1)

    torch.manual_seed(seed)
    X = np.asarray(beats, np.float32)
    X = X / (X.std() + 1e-12)
    Xt = torch.from_numpy(X)

    enc = nn.Sequential(nn.Linear(L, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, latent_dim))
    dec = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 64),
                        nn.ReLU(), nn.Linear(64, L))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    lossf = nn.MSELoss()

    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            xb = Xt[perm[s:s + batch]]
            opt.zero_grad()
            lossf(dec(enc(xb)), xb).backward()
            opt.step()

    with torch.no_grad():
        Z = enc(Xt).numpy()
    return _hdbscan(Z, min_cluster_size, min_samples)


METHODS = {
    "hdbscan": cluster_hdbscan,
    "autoencoder": cluster_autoencoder,
    "corr": cluster_corr,
    "rms": cluster_rms,
}


# ----------------------------------------------------------------------
# Chaining
# ----------------------------------------------------------------------
def apply_steps(beats, steps, labels=None, recluster_noise=False):
    """Apply steps one after another; each step runs inside every existing cluster.
    steps: list of "name" or ("name", {params}).
    recluster_noise: if True the noise group (-1) is also passed to the next step."""
    n = len(beats)
    labels = np.zeros(n, int) if labels is None else np.asarray(labels).copy()
    for step in steps:
        name, params = (step, {}) if isinstance(step, str) else step
        fn = METHODS[name]
        new = np.full(n, -1)
        nxt = 0
        for g in np.unique(labels):
            if g == -1 and not recluster_noise:
                continue
            idx = np.where(labels == g)[0]
            sub = fn(beats[idx], **params)
            for u in np.unique(sub[sub >= 0]):
                new[idx[sub == u]] = nxt
                nxt += 1
        labels = new
    return labels


# ----------------------------------------------------------------------
# Saving (plots + npz)
# ----------------------------------------------------------------------
def _plot_cluster(cb, med, t, title, path, max_beats=300, seed=0):
    if len(cb) > max_beats:
        cb = cb[np.random.default_rng(seed).choice(len(cb), max_beats, replace=False)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, cb.T, color="gray", alpha=0.3, lw=0.8)
    if med is not None:
        ax.plot(t, med, "r", lw=2, label="median")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(out_dir, beats, rpeaks, labels, fs, pre, post, steps):
    """Writes out_dir/plots/cluster_XXX.png (+ noise.png, all_medians.png)
    and out_dir/clusters.npz."""
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for f in glob.glob(os.path.join(plot_dir, "*.png")):
        os.remove(f)

    L = beats.shape[1]
    t = (np.arange(L) / fs - pre) * 1000.0
    ids = [k for k in np.unique(labels) if k >= 0]
    medians = {}
    for k in ids:
        cb = beats[labels == k]
        medians[k] = np.median(cb, axis=0)
        _plot_cluster(cb, medians[k], t, f"Cluster {k} (n={len(cb)})",
                      os.path.join(plot_dir, f"cluster_{k:03d}.png"))
    if np.any(labels == -1):
        nb = beats[labels == -1]
        _plot_cluster(nb, None, t, f"Noise (n={len(nb)})",
                      os.path.join(plot_dir, "noise.png"))
    if ids:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in ids:
            ax.plot(t, medians[k], label=f"{k} (n={np.sum(labels == k)})")
        if len(ids) <= 15:
            ax.legend(fontsize=7)
        ax.set_xlabel("Time from R-peak (ms)")
        ax.set_title("Cluster medians")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "all_medians.png"), dpi=150)
        plt.close(fig)

    per_cluster = {f"rpeaks_cluster_{k}": rpeaks[labels == k] for k in ids}
    if np.any(labels == -1):
        per_cluster["rpeaks_noise"] = rpeaks[labels == -1]
    np.savez(os.path.join(out_dir, "clusters.npz"),
             rpeaks=rpeaks, labels=labels, fs=fs, pre=pre, post=post,
             steps=np.array(str(steps)), **per_cluster)


# ----------------------------------------------------------------------
# Main entry points
# ----------------------------------------------------------------------
def run_clustering(sig, fs, rpeaks, out_dir, steps, pre=0.250, post=0.300,
                   remove_mean=True, recluster_noise=False):
    """Cluster, then save plots + clusters.npz into out_dir.
    Returns dict: labels, rpeaks (kept beats), keep_idx, beats."""
    os.makedirs(out_dir, exist_ok=True)
    beats, keep_idx = extract_beats(sig, rpeaks, fs, pre, post, remove_mean)
    kept = np.asarray(rpeaks, int)[keep_idx]
    labels = apply_steps(beats, steps, recluster_noise=recluster_noise)
    save_results(out_dir, beats, kept, labels, fs, pre, post, steps)
    return {"labels": labels, "rpeaks": kept, "keep_idx": keep_idx, "beats": beats}


def refine_from_npz(npz_path, sig, out_dir, steps, remove_mean=True,
                    recluster_noise=False, only_clusters=None):
    """Load a clusters.npz (rpeaks, labels, fs, pre, post), apply more steps
    inside each saved cluster, and save the result to out_dir.
    only_clusters: optional list of labels to refine; the others are kept as they are."""
    d = np.load(npz_path, allow_pickle=True)
    rpeaks = d["rpeaks"]
    fs, pre, post = float(d["fs"]), float(d["pre"]), float(d["post"])
    labels = d["labels"] if "labels" in d.files else None

    beats, keep_idx = extract_beats(sig, rpeaks, fs, pre, post, remove_mean)
    kept = rpeaks[keep_idx]
    labels = None if labels is None else labels[keep_idx]

    if only_clusters is None or labels is None:
        new = apply_steps(beats, steps, labels, recluster_noise)
    else:
        sel = np.isin(labels, only_clusters)
        sub = apply_steps(beats[sel], steps, labels[sel], recluster_noise)
        new = labels.copy()
        offset = labels.max() + 1
        new[sel] = np.where(sub >= 0, sub + offset, -1)
        u = np.unique(new[new >= 0])           # make labels consecutive
        remap = {old: i for i, old in enumerate(u)}
        new = np.array([remap.get(x, -1) for x in new])

    os.makedirs(out_dir, exist_ok=True)
    save_results(out_dir, beats, kept, new, fs, pre, post, steps)
    return {"labels": new, "rpeaks": kept, "keep_idx": keep_idx, "beats": beats}


# ----------------------------------------------------------------------
# Compare combinations
# ----------------------------------------------------------------------
def _quality(beats, labels):
    """Mean RMS-ratio and mean correlation of every clustered beat vs its cluster median."""
    ratios, corrs = [], []
    for k in np.unique(labels[labels >= 0]):
        cb = beats[labels == k]
        m = np.median(cb, axis=0)
        ratios.append(np.array([_score_rms(b, m[None])[0] for b in cb]))
        corrs.append(np.array([1 - _score_corr(b, m[None])[0] for b in cb]))
    if not ratios:
        return np.nan, np.nan
    return float(np.mean(np.concatenate(ratios))), float(np.mean(np.concatenate(corrs)))


def compare_combinations(sig, fs, rpeaks, out_root, combos, pre=0.250, post=0.300,
                         true_labels=None, **kw):
    """combos: {"name": steps}. Each combo is saved in out_root/<name>/.
    true_labels (optional, aligned with rpeaks) adds Adjusted Rand Index to the table.
    Prints and returns a list of dict rows."""
    rows = []
    for name, steps in combos.items():
        res = run_clustering(sig, fs, rpeaks, os.path.join(out_root, name), steps,
                             pre, post, **kw)
        lab, beats = res["labels"], res["beats"]
        ids = np.unique(lab[lab >= 0])
        sizes = np.array([np.sum(lab == k) for k in ids]) if len(ids) else np.array([0])
        ratio, corr = _quality(beats, lab)
        row = {"combo": name, "clusters": len(ids), "noise_%": 100 * np.mean(lab == -1),
               "top5_%": 100 * np.sort(sizes)[::-1][:5].sum() / max(len(lab), 1),
               "mean_rms_ratio": ratio, "mean_corr": corr}
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score
            row["ARI"] = adjusted_rand_score(np.asarray(true_labels)[res["keep_idx"]], lab)
        rows.append(row)

    keys = list(rows[0].keys())
    print("  ".join(f"{k:>14}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:>14.3f}" if isinstance(r[k], float) else f"{r[k]:>14}"
                        for k in keys))
    return rows


# ----------------------------------------------------------------------
# Custom pipeline: HDBSCAN -> corr -> rms, with "exceptions" folders
# ----------------------------------------------------------------------
def run_pipeline(sig, fs, rpeaks, out_dir, pre=0.250, post=0.300, remove_mean=True,
                 hdb_min_cluster=3, hdb_variance=0.999,
                 hdb_exc_frac=0.001, process_frac=0.10,
                 corr_thr=0.90, corr_exc_frac=0.01, rms_thr=0.5):
    """
    1. HDBSCAN (min_cluster_size=hdb_min_cluster, PCA variance=hdb_variance).
    2. HDBSCAN clusters with n < hdb_exc_frac * len(rpeaks)  -> exceptions
       (HDBSCAN noise goes there too as noise.png).
    3. HDBSCAN clusters with n <= process_frac * len(rpeaks) are final as they are.
       Clusters with n >  process_frac * len(rpeaks) are processed further:
    4.   corr (thr=corr_thr) inside the HDBSCAN cluster.
         Corr clusters with n < corr_exc_frac * (size of that HDBSCAN cluster)
         -> exceptions
    5.   rms (thr=rms_thr) inside every remaining corr cluster -> final clusters.

    Valid clusters   -> out_dir/valid/hdbXXX/
    Exceptions       -> out_dir/exceptions/ (HDBSCAN-level + noise) and
                        out_dir/exceptions/hdbXXX/ (corr-level)
    Every cluster gets <name>.png (all beats + red median) and <name>.npz (its R-peaks). Also saved: out_dir/all_clusters.npz and out_dir/report.csv.
    Returns the report (list of dicts).
    """
    import csv
    os.makedirs(out_dir, exist_ok=True)
    rpeaks = np.asarray(rpeaks, int)
    N = len(rpeaks)
    beats, keep_idx = extract_beats(sig, rpeaks, fs, pre, post, remove_mean)
    rp = rpeaks[keep_idx]
    nb = len(beats)
    t = (np.arange(beats.shape[1]) / fs - pre) * 1000.0

    labels = np.full(nb, -1)                      # final cluster id, -1 = exception
    group = np.full(nb, "", dtype=object)         # name of the cluster each beat is in
    report, counter = [], [0]

    def record(final, name, folder, idx, median=True):
        os.makedirs(folder, exist_ok=True)
        png = os.path.join(folder, f"{name}.png")
        npz = os.path.join(folder, f"{name}.npz")
        med = np.median(beats[idx], axis=0) if median else None
        _plot_cluster(beats[idx], med, t, f"{name} (n={len(idx)})", png)
        np.savez(npz, rpeaks=rp[idx], fs=fs, pre=pre, post=post)
        group[idx] = name
        if final:
            labels[idx] = counter[0]
            counter[0] += 1
        report.append({"kind": "final" if final else "exception", "name": name,
                       "n_beats": len(idx), "png": png, "rpeaks_npz": npz})

    hl = cluster_hdbscan(beats, min_cluster_size=hdb_min_cluster,
                         n_components=hdb_variance)
    exc_root = os.path.join(out_dir, "exceptions")
    valid_root = os.path.join(out_dir, "valid")

    noise = np.where(hl == -1)[0]
    if len(noise):
        record(False, "noise", exc_root, noise)

    for k in np.unique(hl[hl >= 0]):
        idx = np.where(hl == k)[0]
        n = len(idx)
        hn = f"hdb{k:03d}"
        hfolder = os.path.join(valid_root, hn)          # valid clusters of this HDBSCAN cluster
        hexc = os.path.join(exc_root, hn)               # its exceptions

        if n < hdb_exc_frac * N:                  # too small -> exception
            record(False, hn, exc_root, idx)
            continue
        if n <= process_frac * N:                 # medium -> final, not processed
            record(True, hn, hfolder, idx)
            continue

        cl = cluster_corr(beats[idx], thr=corr_thr)
        for c in np.unique(cl):
            cidx = idx[cl == c]
            cn = f"{hn}_corr{c:03d}"
            if len(cidx) < corr_exc_frac * n:     # tiny corr cluster -> exception
                record(False, cn, hexc, cidx)
                continue
            rl = cluster_rms(beats[cidx], thr=rms_thr)
            for r in np.unique(rl):
                record(True, f"{cn}_rms{r:03d}", hfolder, cidx[rl == r])

    np.savez(os.path.join(out_dir, "all_clusters.npz"), rpeaks=rp, labels=labels,
             group=group.astype(str), fs=fs, pre=pre, post=post)
    with open(os.path.join(out_dir, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "name", "n_beats", "png", "rpeaks_npz"])
        w.writeheader()
        w.writerows(report)

    fin = [r for r in report if r["kind"] == "final"]
    exc = [r for r in report if r["kind"] == "exception"]
    print(f"{'kind':<10} {'cluster':<28} {'beats':>7}  png")
    for r in fin + exc:
        print(f"{r['kind']:<10} {r['name']:<28} {r['n_beats']:>7}  {r['png']}")
    print(f"\nHDBSCAN found {len(np.unique(hl[hl >= 0]))} clusters, noise = {len(noise)} beats")
    print(f"Non-exception clusters: {len(fin)}  ({sum(r['n_beats'] for r in fin)} beats)")
    print(f"Exception clusters:     {len(exc)}  ({sum(r['n_beats'] for r in exc)} beats)")
    print(f"Total beats accounted:  {sum(r['n_beats'] for r in report)} / {nb}")
    return report
