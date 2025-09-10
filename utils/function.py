import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import normalize


def plot_tsne_with_centroids(feats: torch.Tensor, labels: torch.Tensor, title="t-SNE (512?2D) with centroids", perplexity=30, vocab=None, train_config=None):
    """
    t-SNE ??? + ??? ?? + ?? ??
    Args:
        feats: [N, D] float tensor (e.g., [batch, 512])
        labels: [N] long tensor (e.g., [batch])
    Returns:
        metrics: dict = {
            "intra_class_distance": float,
            "inter_class_similarity": float,
            "silhouette_score": float
        }
    """
    X = feats.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    # 1) t-SNE (2D ??)
    X2 = TSNE(n_components=2, init="pca", learning_rate="auto",
              perplexity=perplexity, random_state=0).fit_transform(X)  # [N,2]

    # 2) ???? centroid ??
    classes = np.unique(y)
    centroids = np.stack([X2[y == c].mean(axis=0) for c in classes], axis=0)  # [C, 2]

    # 3) ???
    palette = sns.color_palette("hls", len(classes))
    plt.figure(figsize=(8, 8))
    for i, c in enumerate(classes):
        mask = (y == c)
        for k, v in vocab.items():
            if c == v:
                joint_name = k
        plt.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.8, color=palette[i], label=f"{joint_name}")
        plt.scatter(centroids[i, 0], centroids[i, 1], s=130, marker='X', edgecolors='black', linewidths=1.0, color=palette[i])
    plt.title(title)
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")
    plt.legend(markerscale=1.5, fontsize=8, ncol=2)
    plt.tight_layout()
    save_path = f"/home/jysuh/PycharmProjects/coord_embedding/embedding_result_img/tsne_{train_config}.png"
    # plt.show()

    # 4) ?? ?? ?? ??
    metrics = {}

    # (1) Intra-class ?? (? ??? ? ?? ??)
    intra_dists = []
    for c in classes:
        mask = (y == c)
        class_pts = X2[mask]
        if len(class_pts) > 1:
            dists = pairwise_distances(class_pts)
            intra_dists.append(np.mean(dists))
    metrics["intra_class_distance"] = float(np.mean(intra_dists))

    # (2) Inter-class cosine similarity (512D ??)
    feats_norm = normalize(X, axis=1)
    class_means = np.stack([feats_norm[y == c].mean(axis=0) for c in classes])
    cosine_sim = class_means @ class_means.T
    inter_cosine_sim = np.sum(cosine_sim) - np.trace(cosine_sim)  # off-diagonal ?
    num_pairs = len(classes) * (len(classes) - 1)
    metrics["inter_class_similarity"] = float(inter_cosine_sim / num_pairs)

    # (3) Silhouette score (?? ?? ??)
    if len(np.unique(y)) > 1 and len(X2) > len(np.unique(y)):
        metrics["silhouette_score"] = float(silhouette_score(X2, y))
    else:
        metrics["silhouette_score"] = -1  # ?? ??

    if metrics['intra_class_distance'] < 25 and metrics['inter_class_similarity'] < 0.5 and metrics['silhouette_score'] > 0.4:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    return metrics