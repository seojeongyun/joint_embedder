import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from typing import Optional, Dict, Any

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import normalize

from sklearn.metrics import pairwise_distances


def dunn_index(X, y, metric="cosine"):
    """
    Compute the Dunn Index for cluster separation quality.

    Dunn Index = (minimum inter-cluster distance) / (maximum intra-cluster diameter)
      - Inter-cluster distance: the smallest distance between any two points belonging to different clusters
      - Intra-cluster diameter: the largest distance between any two points within the same cluster
    A higher Dunn Index indicates better separation between clusters and tighter within-cluster compactness.

    Parameters:
        X (np.ndarray): Feature matrix of shape [N, D]
        y (np.ndarray): Cluster labels of shape [N]
        metric (str): Distance metric ("cosine" or "euclidean")

    Returns:
        float: The computed Dunn Index value (higher is better)
    """
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")

    # Compute full pairwise distance matrix (symmetric, diagonal = 0)
    D = pairwise_distances(X, metric=metric)

    # Collect indices for each cluster
    idx_list = [np.where(y == c)[0] for c in classes]

    # Compute intra-cluster diameters (maximum pairwise distance within each cluster)
    diameters = []
    for idx in idx_list:
        if len(idx) <= 1:
            diameters.append(0.0)
        else:
            sub = D[np.ix_(idx, idx)]
            mask = ~np.eye(len(idx), dtype=bool)  # exclude diagonal
            m = np.max(sub[mask])
            diameters.append(float(m))
    max_diameter = max(diameters) if len(diameters) > 0 else 0.0

    # Compute minimum inter-cluster distance across all cluster pairs
    inter_min = np.inf
    for i in range(len(idx_list)):
        for j in range(i + 1, len(idx_list)):
            sub = D[np.ix_(idx_list[i], idx_list[j])]
            inter_min = min(inter_min, float(np.min(sub)))

    # Avoid division by zero (if all clusters have zero diameter)
    if max_diameter == 0.0:
        return float("inf")
    return inter_min / (max_diameter + 1e-12)

def plot_tsne_with_centroids(config, feats, labels, vocab, file_name=None, visualization=None):
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

    Xo, yo = X, y

    # If metric == 'cosine', L2 normalization is recommended
    Xo_eval = normalize(Xo, axis=1) if config.VIS.PLOT_METRIC_METHOD == "cosine" else Xo

    # --- inter-class cosine similarity (centroid-based, mean of off-diagonal entries) ---
    class_means = []
    for c in np.unique(yo):
        sel = Xo_eval[yo == c]
        if len(sel) == 0:
            continue
        class_means.append(sel.mean(axis=0)) # centroid point, length=20, class_mean[0]:512dim

    metrics = {
        "num_samples": int(N),
        "num_classes": int(C),
        "feat_dim": int(D),
        f"metric_{D}d": config.VIS.PLOT_METRIC_METHOD,
    }

    if len(class_means) >= 2:
        M = np.stack(class_means)  # [C, D]
        cs = M @ M.T if config.VIS.PLOT_METRIC_METHOD == "cosine" else M @ M.T / (
                np.linalg.norm(M, axis=1, keepdims=True) * np.linalg.norm(M, axis=1, keepdims=True).T + 1e-12
        )  # for safety; equivalent to cosine when metric='cosine'
        off = np.sum(cs) - np.trace(cs)
        pairs = len(M) * (len(M) - 1)
        metrics["inter_class_similarity_orig"] = float(off / pairs)
    else:
        metrics["inter_class_similarity_orig"] = float("nan")

    # --- 512D silhouette: per-sample, per-cluster, and overall ---
    if C > 1 and len(Xo_eval) >= 10 and len(Xo_eval) > C:
        try:
            s_vals = silhouette_samples(Xo_eval, yo, metric=config.VIS.PLOT_METRIC_METHOD)  # silhouette value for each sample s(i)
            sil_overall = float(np.mean(s_vals))  # overall mean silhouette score
            # compute mean silhouette for each cluster
            sil_per_class = {
                int(c): float(np.mean(s_vals[yo == c])) for c in np.unique(yo) if np.sum(yo == c) > 1
            }
            metrics["silhouette_score_orig"] = sil_overall
            metrics["silhouette_score_per_class"] = sil_per_class
        except Exception:
            metrics["silhouette_score_orig"] = float("nan")
            metrics["silhouette_score_per_class"] = {}
            s_vals = None
    else:
        metrics["silhouette_score_orig"] = float("nan")
        metrics["silhouette_score_per_class"] = {}
        s_vals = None

    # --- Dunn Index (512D) ---
    try:
        metrics["dunn_index_orig"] = float(dunn_index(Xo_eval, yo, metric=config.VIS.PLOT_METRIC_METHOD))
    except Exception:
        metrics["dunn_index_orig"] = float("nan")

    # ---------- (B) 2D t-SNE: visualization only ----------
    if visualization:
        if file_name is None:
            file_name = config.FILE_NAME
        else:
            file_name = file_name

        X2 = TSNE(
            n_components=2, init="pca", learning_rate="auto",
            perplexity=config.VIS.TSNE_PERPLEXITY, random_state=config.VIS.TSNE_RANDOM_SEED,
            n_iter=config.VIS.TSNE_N_ITER
        ).fit_transform(X)  # use the entire X for visualization (can be sampled if needed)

    # ---------- (C) Visualization ----------
        out_dir = os.path.join(config.VIS.PLOT_SAVE_ROOT, file_name)
        os.makedirs(out_dir, exist_ok=True)

        # build label2name mapping once (use joint names if provided)
        if vocab is not None:
            try:
                # case: {"Right_Elbow": 0, "Left_Knee": 1, ...}  -> invert
                if all(isinstance(k, str) and isinstance(v, (int, np.integer)) for k, v in vocab.items()):
                    label2name = {int(v): str(k) for k, v in vocab.items()}
                else:
                    # case: {0: "Right_Elbow", 1: "Left_Knee", ...}
                    label2name = {int(k): str(v) for k, v in vocab.items()}
            except Exception:
                # fallback: just cast to string where possible
                label2name = {int(k): str(v) for k, v in vocab.items()} if isinstance(vocab, dict) else None
        else:
            # fallback to numeric labels as strings
            label2name = {int(c): str(int(c)) for c in classes}

        # (C-1) 512D silhouette plot (left panel)
        #  - x-axis: silhouette coefficient s(i)
        #  - y-axis: samples grouped by cluster (sorted within each cluster)
        #  - overall average shown as a red dashed line
        if s_vals is not None:  # silhouette score / t-sne
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ax_sil, ax_tsne = ax[0], ax[1]

            y_lower = 10
            colors = plt.cm.get_cmap("tab20", C)

            for i, c in enumerate(np.unique(yo)):
                s_c = s_vals[yo == c]
                if len(s_c) == 0:
                    continue
                s_c = np.sort(s_c)
                size_c = len(s_c)
                y_upper = y_lower + size_c

                ax_sil.fill_betweenx(
                    y=np.arange(y_lower, y_upper),
                    x1=0, x2=s_c,
                    alpha=0.7, color=colors(i)
                )
                joint_name = label2name.get(int(c), str(int(c)))
                ax_sil.text(-0.08, y_lower + 0.5 * size_c, joint_name, fontsize=8)

                y_lower = y_upper + 10  # space between clusters

            ax_sil.axvline(x=metrics["silhouette_score_orig"], color="red", linestyle="--", linewidth=2)

            # -------------------- ADD: draw Dunn Index (blue, dash-dot) --------------------
            dunn_val = metrics.get("dunn_index_orig", float("nan"))
            if np.isfinite(dunn_val):
                # clamp to axis if Dunn > 1.0 (silhouette axis ends at 1.0)
                x_dunn_draw = min(dunn_val, 0.99)
                ax_sil.axvline(x=x_dunn_draw, color="blue", linestyle="-.", linewidth=2)

                # annotate actual value near the top of the panel
                y_top = ax_sil.get_ylim()[1] if ax_sil.get_ylim() else (y_lower + 10)
                note = f"Dunn = {dunn_val:.3f}" + (" (capped)" if dunn_val > 1.0 else "")
                ax_sil.text(x_dunn_draw, y_top - 5, note, color="blue", fontsize=8,
                            ha="right" if dunn_val > 1.0 else "left", va="top")
            # -------------------------------------------------------------------------------

            ax_sil.set_title(f"Silhouette plot per cluster ({D}D)")
            ax_sil.set_xlabel("Silhouette coefficient")
            ax_sil.set_ylabel("Cluster")
            ax_sil.set_xlim([-0.1, 1.0])
            ax_sil.set_yticks([])

            # (C-2) 2D t-SNE scatter (right panel)
            classes2 = np.unique(y)
            centroids = np.stack([X2[y == c].mean(axis=0) for c in classes2], axis=0) if len(classes2) > 0 else None

            for i, c in enumerate(classes2):
                mask = (y == c)
                name = label2name.get(int(c), str(int(c)))
                ax_tsne.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.8, label=name)
                if centroids is not None:
                    ax_tsne.scatter(centroids[i, 0], centroids[i, 1], s=130, marker='X',
                                    edgecolors='black', linewidths=1.0)

            ax_tsne.set_title(
                f"t-SNE ({D}D to 2D) with centroids\nperp={config.VIS.TSNE_PERPLEXITY} | N={len(X)} | C={C}")
            ax_tsne.set_xlabel("Dim-1");
            ax_tsne.set_ylabel("Dim-2")
            if len(classes2) <= 30:
                ax_tsne.legend(markerscale=1.5, fontsize=8, ncol=2)
            plt.tight_layout()

            png_path = os.path.join(out_dir, f"{file_name}.png")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close()

        else:   # only t-sne
            # t-SNE only (re-use label2name here as well)
            plt.figure(figsize=(8, 8))
            classes2 = np.unique(y)
            centroids = np.stack([X2[y == c].mean(axis=0) for c in classes2], axis=0) if len(classes2) > 0 else None
            for i, c in enumerate(classes2):
                mask = (y == c)
                name = label2name.get(int(c), str(int(c)))
                plt.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.8, label=name)
                if centroids is not None:
                    plt.scatter(centroids[i, 0], centroids[i, 1], s=130, marker='X',
                                edgecolors='black', linewidths=1.0)
            plt.title(f"t-SNE (512D to 2D) with centroids\nperp={config.VIS.TSNE_PERPLEXITY} | N={len(X)} | C={C}")
            plt.xlabel("Dim-1");
            plt.ylabel("Dim-2")
            if len(classes2) <= 30:
                plt.legend(markerscale=1.5, fontsize=8, ncol=2)
            plt.tight_layout()

            out_dir = os.path.join(config.VIS.PLOT_SAVE_ROOT, file_name)
            os.makedirs(out_dir, exist_ok=True)
            png_path = os.path.join(out_dir, f"{file_name}.png")
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close()

        # Save metrics to file
        with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.dump(dict(config), f, allow_unicode=True, default_flow_style=False)

        print(f"SAVE Done ! in {out_dir}")

    return metrics