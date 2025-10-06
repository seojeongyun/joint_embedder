import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Dict, Any
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import normalize


def plot_tsne_with_centroids(
    feats,                   # torch.Tensor[N,D] ?? np.ndarray[N,D]
    labels,                  # torch.Tensor[N]   ?? np.ndarray[N]
    title: str = "t-SNE (D?2D) with centroids",
    perplexity: int = 30,
    vocab: Optional[Dict[Any, Any]] = None,
    save_root: str = "./embedding_result_img",
    exp_name: str = "default",
    eval_space: str = "auto",          # "auto" | "orig" | "2d"
    tsne_max_samples: int = 50000,      # 2D ???? ???? ?? ?? ?
    tsne_min_samples_factor: int = 4,  # n >= 1 + factor*perplexity
    intra_per_class_cap: int = 300,    # intra-class ?? ?? ? ???? ?? ??
    orig_max_samples: int = 8000,      # ??D ??? ?? ? ?? ??
    random_state: int = 0,
    thr_intra_2d: float = 25.0,
    thr_sil_2d: float = 0.40,
    thr_inter_orig: float = 0.50,
    thr_sil_orig: float = 0.25,
):
    """1) ????  2) ??? ??  3) ??? ?? ??? PNG/metrics ??."""
    # --- ?? ?? (torch/np ?? ??) ---
    try:
        import torch
        from torch import Tensor as TorchTensor
        is_torch = isinstance(feats, TorchTensor)
    except Exception:
        is_torch = False

    X = feats.detach().cpu().numpy() if is_torch else np.asarray(feats)
    y = labels.detach().cpu().numpy() if is_torch else np.asarray(labels)

    assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "feats:[N,D], labels:[N]"
    N, D = X.shape
    classes = np.unique(y)
    C = len(classes)

    # --- 2D ???(????): t-SNE or PCA fallback + ???? ---
    if N > tsne_max_samples:
        used_idx = np.linspace(0, N - 1, num=tsne_max_samples, dtype=int)
        X_use = X[used_idx]
        y_use = y[used_idx]
    else:
        used_idx = np.arange(N, dtype=int)
        X_use = X
        y_use = y

    max_perp = max(5, min(perplexity, (len(X_use) - 1) // tsne_min_samples_factor))
    if max_perp < 5:
        # ?? ?? ? PCA ??
        from sklearn.decomposition import PCA
        X2 = PCA(n_components=2, random_state=random_state).fit_transform(X_use)
        method2d = "PCA(2D)"
    else:
        X2 = TSNE(n_components=2, init="pca", learning_rate="auto",
                  perplexity=max_perp, random_state=random_state).fit_transform(X_use)
        method2d = f"t-SNE(2D, perp={max_perp})"

    # --- ?? ?? ?? ---
    metrics = {
        "num_samples": int(N),
        "num_classes": int(C),
        "feat_dim": int(D),
        "method_2d": method2d,
        "tsne_used": int(len(X_use)),
    }

    # 2D: intra-class ?? ???(O(n^2)????? ???? ??), silhouette
    def _intra_2d(x2, yy, cap=intra_per_class_cap):
        vals = []
        for c in np.unique(yy):
            pts = x2[yy == c]
            if len(pts) > cap:
                idx = np.linspace(0, len(pts) - 1, num=cap, dtype=int)
                pts = pts[idx]
            if len(pts) > 1:
                Dm = pairwise_distances(pts)
                vals.append(np.mean(Dm))
        return float(np.mean(vals)) if vals else float("nan")

    metrics["intra_class_distance_2d"] = _intra_2d(X2, y_use, intra_per_class_cap)
    if C > 1 and len(X2) >= 10 and len(X2) > C:
        try:
            metrics["silhouette_score_2d"] = float(silhouette_score(X2, y_use))
        except Exception:
            metrics["silhouette_score_2d"] = float("nan")
    else:
        metrics["silhouette_score_2d"] = float("nan")

    # ?? D: inter-class cosine(??? ?? ??), silhouette (???+????)
    if N > orig_max_samples:
        idx_o = np.linspace(0, N - 1, num=orig_max_samples, dtype=int)
        Xo, yo = X[idx_o], y[idx_o]
    else:
        Xo, yo = X, y

    feats_norm = normalize(Xo, axis=1)
    # inter-class cosine
    class_means = []
    for c in np.unique(yo):
        sel = feats_norm[yo == c]
        if len(sel) == 0:
            continue
        class_means.append(sel.mean(axis=0))
    if len(class_means) >= 2:
        M = np.stack(class_means)
        cs = M @ M.T
        off = np.sum(cs) - np.trace(cs)
        pairs = len(M) * (len(M) - 1)
        metrics["inter_class_similarity_orig"] = float(off / pairs)
    else:
        metrics["inter_class_similarity_orig"] = float("nan")

    if C > 1 and len(Xo) >= 10 and len(Xo) > C:
        try:
            metrics["silhouette_score_orig"] = float(silhouette_score(feats_norm, yo))
            # ?? ? ??? ??:
            # metrics["silhouette_score_orig_cos"] = float(silhouette_score(feats_norm, yo, metric="cosine"))
        except Exception:
            metrics["silhouette_score_orig"] = float("nan")
    else:
        metrics["silhouette_score_orig"] = float("nan")

    # --- ?? ?? ?? (??) ---
    # ??D ???? ??(??/??? ?? ?? & NaN ??)?? ??D ??.
    if eval_space == "orig":
        pref = "orig"
    elif eval_space == "2d":
        pref = "2d"
    else:
        pref = "orig" if not np.isnan(metrics["silhouette_score_orig"]) else "2d"
    metrics["preferred_eval_space"] = pref

    # --- ??? ?? ---
    pass_2d = (
        (not np.isnan(metrics["intra_class_distance_2d"]) and metrics["intra_class_distance_2d"] < thr_intra_2d) and
        (not np.isnan(metrics["silhouette_score_2d"])       and metrics["silhouette_score_2d"] >= thr_sil_2d)
    )
    pass_orig = (
        (not np.isnan(metrics["inter_class_similarity_orig"]) and metrics["inter_class_similarity_orig"] < thr_inter_orig) and
        (not np.isnan(metrics["silhouette_score_orig"])        and metrics["silhouette_score_orig"] >= thr_sil_orig)
    )
    passed = pass_orig if pref == "orig" else pass_2d
    metrics["passed"] = bool(passed)

    # --- ??? ?? ??? ?? ---
    if passed:
        out_dir = os.path.join(save_root, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        # ?? ??(??)
        label2name = None
        if vocab is not None:
            try:
                # {"name": id}? ?? ??
                if all(isinstance(k, str) and isinstance(v, (int, np.integer)) for k, v in vocab.items()):
                    label2name = {v: k for k, v in vocab.items()}
                else:
                    label2name = vocab
            except Exception:
                label2name = vocab

        # 2D scatter + centroids
        classes2 = np.unique(y_use)
        centroids = np.stack([X2[y_use == c].mean(axis=0) for c in classes2], axis=0) if len(classes2) > 0 else None

        plt.figure(figsize=(8, 8))
        for i, c in enumerate(classes2):
            mask = (y_use == c)
            name = label2name.get(int(c), str(int(c))) if label2name else str(int(c))
            plt.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.8, label=name)
            if centroids is not None:
                plt.scatter(centroids[i, 0], centroids[i, 1], s=130, marker='X',
                            edgecolors='black', linewidths=1.0)
        plt.title(f"{title}\n{method2d} | N={len(X_use)} | C={C}")
        plt.xlabel("Dim-1"); plt.ylabel("Dim-2")
        if len(classes2) <= 30:
            plt.legend(markerscale=1.5, fontsize=8, ncol=2)
        plt.tight_layout()

        png_path = os.path.join(out_dir, f"{exp_name}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()

        with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

    return metrics