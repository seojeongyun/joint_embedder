import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import yaml

from typing import Optional, Dict, Any
from matplotlib.patches import Rectangle

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import normalize
import matplotlib.colors as mcolors
from sklearn.metrics import pairwise_distances


def _stratified_sample(X, y, max_samples_per_class=300, random_state=614):
    """
    Select at most ``max_samples_per_class`` samples from every class.

    The fixed random seed makes validation metrics and plots reproducible across
    repeated evaluations of the same checkpoint.
    """
    rng = np.random.default_rng(random_state)
    selected_indices = []

    for class_id in np.unique(y):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) > max_samples_per_class:
            class_indices = rng.choice(
                class_indices,
                size=max_samples_per_class,
                replace=False,
            )
        selected_indices.append(class_indices)

    if not selected_indices:
        return X, y

    selected_indices = np.concatenate(selected_indices)
    return X[selected_indices], y[selected_indices]


def _build_label2name(vocab, classes):
    """Create a label-id to display-name mapping."""
    if vocab is not None:
        try:
            if all(
                isinstance(k, str) and isinstance(v, (int, np.integer))
                for k, v in vocab.items()
            ):
                label2name = {int(v): str(k) for k, v in vocab.items()}
            else:
                label2name = {int(k): str(v) for k, v in vocab.items()}
        except Exception:
            label2name = {}
    else:
        label2name = {}

    label2name.setdefault(0, "PAD")
    label2name.setdefault(1, "SEP")
    for class_id in classes:
        label2name.setdefault(int(class_id), str(int(class_id)))

    return label2name


def _joint_centroid_cosine_analysis(
    X,
    y,
    label2name,
    num_special_tokens=2,
):
    """
    Compute a joint-only centroid cosine matrix.

    Each sample is L2-normalized first, samples belonging to the same joint are
    averaged, and each resulting centroid is normalized again.
    """
    joint_classes = np.asarray(
        [
            class_id
            for class_id in np.unique(y)
            if int(class_id) >= num_special_tokens
        ]
    )

    if len(joint_classes) < 2:
        return None

    normalized_samples = normalize(np.asarray(X), axis=1)
    joint_centroids = []

    for class_id in joint_classes:
        class_samples = normalized_samples[y == class_id]
        if len(class_samples) == 0:
            continue
        joint_centroids.append(class_samples.mean(axis=0))

    joint_centroids = normalize(np.stack(joint_centroids), axis=1)
    cosine_matrix = np.clip(
        joint_centroids @ joint_centroids.T,
        -1.0,
        1.0,
    )

    joint_names = [
        label2name.get(int(class_id), str(int(class_id)))
        for class_id in joint_classes
    ]

    pair_results = []
    for i in range(len(joint_classes)):
        for j in range(i + 1, len(joint_classes)):
            pair_results.append(
                {
                    "label_a": int(joint_classes[i]),
                    "joint_a": joint_names[i],
                    "label_b": int(joint_classes[j]),
                    "joint_b": joint_names[j],
                    "cosine": float(cosine_matrix[i, j]),
                }
            )

    pair_results.sort(key=lambda item: item["cosine"], reverse=True)
    pair_values = np.asarray(
        [item["cosine"] for item in pair_results],
        dtype=np.float64,
    )

    name_to_index = {
        joint_name: index
        for index, joint_name in enumerate(joint_names)
    }
    left_right_cosine = {}
    for left_name, left_index in name_to_index.items():
        if not left_name.startswith("Left "):
            continue
        right_name = "Right " + left_name[len("Left "):]
        if right_name not in name_to_index:
            continue
        right_index = name_to_index[right_name]
        left_right_cosine[f"{left_name} <-> {right_name}"] = float(
            cosine_matrix[left_index, right_index]
        )

    return {
        "classes": joint_classes,
        "names": joint_names,
        "centroids": joint_centroids,
        "cosine_matrix": cosine_matrix,
        "mean_off_diagonal": float(pair_values.mean()),
        "max_off_diagonal": float(pair_values.max()),
        "min_off_diagonal": float(pair_values.min()),
        "most_similar_pair": pair_results[0],
        "top5_similar_pairs": pair_results[:5],
        "left_right_cosine": left_right_cosine,
    }


def _save_joint_centroid_cosine_outputs(
    analysis,
    img_dir,
    metrics_dir,
    file_name,
):
    """Save the 20x20 cosine matrix as a CSV file and an annotated heatmap."""
    cosine_matrix = analysis["cosine_matrix"]
    joint_names = analysis["names"]

    csv_path = os.path.join(
        metrics_dir,
        f"{file_name}_joint_centroid_cosine_matrix.csv",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["joint"] + joint_names)
        for joint_name, row in zip(joint_names, cosine_matrix):
            writer.writerow(
                [joint_name] + [f"{float(value):.8f}" for value in row]
            )

    fig_size = max(12, len(joint_names) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(
        cosine_matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ticks = np.arange(len(joint_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(joint_names, rotation=55, ha="right", fontsize=11)
    ax.set_yticklabels(joint_names, fontsize=11)
    ax.set_xlabel("Joint centroid")
    ax.set_ylabel("Joint centroid")
    ax.set_title(
        "Joint Centroid Cosine Similarity\n"
        f"mean(off-diagonal)={analysis['mean_off_diagonal']:.4f} | "
        f"max={analysis['max_off_diagonal']:.4f}"
    )

    for row in range(len(joint_names)):
        for column in range(len(joint_names)):
            value = cosine_matrix[row, column]
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    most_similar = analysis["most_similar_pair"]
    label_to_index = {
        int(class_id): index
        for index, class_id in enumerate(analysis["classes"])
    }
    first_index = label_to_index[most_similar["label_a"]]
    second_index = label_to_index[most_similar["label_b"]]
    for row, column in (
        (first_index, second_index),
        (second_index, first_index),
    ):
        ax.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="lime",
                linewidth=2.5,
            )
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine similarity")
    fig.tight_layout()

    heatmap_path = os.path.join(
        img_dir,
        f"{file_name}_joint_centroid_cosine_heatmap.png",
    )
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "joint_centroid_cosine_csv": csv_path,
        "joint_centroid_cosine_heatmap": heatmap_path,
    }


def _save_centroid_sphere_outputs(
    X,
    y,
    label2name,
    img_dir,
    metrics_dir,
    file_name,
    colors,
    random_state=614,
    radius_percentile=95.0,
):
    """
    Draw a class-centroid sphere summary in a common angular-distance scale.

    The class-centroid positions are obtained by applying metric MDS to the
    centroid-to-centroid angular-distance matrix. Each sphere radius is the
    selected percentile of sample-to-own-centroid angular distances. Therefore,
    centre separation and sphere radius use the same unit (radians).

    This is a cluster summary rather than a projection of individual samples.
    """
    classes = np.unique(y)
    if len(classes) < 2:
        return {}

    normalized_samples = normalize(np.asarray(X), axis=1)
    centroids = []
    radii = []
    class_counts = []

    for class_id in classes:
        class_samples = normalized_samples[y == class_id]
        centroid = normalize(
            class_samples.mean(axis=0, keepdims=True),
            axis=1,
        )[0]
        sample_cosine = np.clip(
            class_samples @ centroid,
            -1.0,
            1.0,
        )
        sample_angles = np.arccos(sample_cosine)

        centroids.append(centroid)
        radii.append(
            float(np.percentile(sample_angles, radius_percentile))
        )
        class_counts.append(int(len(class_samples)))

    centroids = np.stack(centroids)
    radii = np.asarray(radii, dtype=np.float64)

    centroid_cosine = np.clip(
        centroids @ centroids.T,
        -1.0,
        1.0,
    )
    centroid_angular_distance = np.arccos(centroid_cosine)
    np.fill_diagonal(centroid_angular_distance, 0.0)

    mds = MDS(
        n_components=3,
        metric=True,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=8,
        max_iter=1000,
        eps=1e-9,
        n_jobs=-1,
        normalized_stress=False,
    )
    centroid_positions = mds.fit_transform(
        centroid_angular_distance
    )

    csv_path = os.path.join(
        metrics_dir,
        f"{file_name}_centroid_spheres.csv",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "label",
                "class_name",
                "num_samples",
                f"angular_radius_p{radius_percentile:g}_radian",
                f"angular_radius_p{radius_percentile:g}_degree",
                "mds_x",
                "mds_y",
                "mds_z",
            ]
        )
        for class_id, count, radius, position in zip(
            classes,
            class_counts,
            radii,
            centroid_positions,
        ):
            writer.writerow(
                [
                    int(class_id),
                    label2name.get(int(class_id), str(int(class_id))),
                    count,
                    f"{radius:.10f}",
                    f"{np.degrees(radius):.10f}",
                    f"{position[0]:.10f}",
                    f"{position[1]:.10f}",
                    f"{position[2]:.10f}",
                ]
            )

    sphere_u = np.linspace(0.0, 2.0 * np.pi, 28)
    sphere_v = np.linspace(0.0, np.pi, 18)
    nonzero_centroid_distances = centroid_angular_distance[
        centroid_angular_distance > 1e-9
    ]
    typical_centroid_distance = (
        float(np.median(nonzero_centroid_distances))
        if len(nonzero_centroid_distances) > 0
        else 1.0
    )
    zero_radius_threshold = typical_centroid_distance * 1e-5

    image_path = os.path.join(
        img_dir,
        f"{file_name}_centroid_spheres.png",
    )
    spread_x2_image_path = os.path.join(
        img_dir,
        f"{file_name}_centroid_spheres_spread_x2.png",
    )

    # Save both the physically consistent view and a qualitative view in which
    # only centroid positions are spread by x2. Sphere radii remain the measured
    # p95 angular radii in both images.
    for position_scale, output_path in (
        (1.0, image_path),
        (2.0, spread_x2_image_path),
    ):
        display_positions = centroid_positions * position_scale

        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        for index, (class_id, position, radius) in enumerate(
            zip(classes, display_positions, radii)
        ):
            color = colors[index % len(colors)]
            name = label2name.get(
                int(class_id),
                str(int(class_id)),
            )

            # Centroids are always shown. A zero-radius class (typically
            # repeated PAD/SEP vectors) is shown as a point, not an artificial
            # sphere.
            ax.scatter(
                position[0],
                position[1],
                position[2],
                s=150,
                marker="X",
                color=color,
                edgecolors="black",
                linewidths=1.0,
                label=name,
                depthshade=False,
            )

            if radius <= zero_radius_threshold:
                continue

            sphere_x = (
                position[0]
                + radius * np.outer(
                    np.cos(sphere_u),
                    np.sin(sphere_v),
                )
            )
            sphere_y = (
                position[1]
                + radius * np.outer(
                    np.sin(sphere_u),
                    np.sin(sphere_v),
                )
            )
            sphere_z = (
                position[2]
                + radius * np.outer(
                    np.ones_like(sphere_u),
                    np.cos(sphere_v),
                )
            )
            ax.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color=color,
                alpha=0.16,
                linewidth=0.0,
                antialiased=True,
                shade=True,
            )
            ax.plot_wireframe(
                sphere_x,
                sphere_y,
                sphere_z,
                color=color,
                alpha=0.22,
                linewidth=0.35,
                rstride=3,
                cstride=3,
            )

        plot_lower = np.min(
            display_positions - radii[:, None],
            axis=0,
        )
        plot_upper = np.max(
            display_positions + radii[:, None],
            axis=0,
        )
        plot_center = (plot_lower + plot_upper) / 2.0
        half_extent = max(
            float(np.max(plot_upper - plot_lower)) / 2.0,
            1e-6,
        ) * 1.08

        ax.set_xlim(
            plot_center[0] - half_extent,
            plot_center[0] + half_extent,
        )
        ax.set_ylim(
            plot_center[1] - half_extent,
            plot_center[1] + half_extent,
        )
        ax.set_zlim(
            plot_center[2] - half_extent,
            plot_center[2] + half_extent,
        )
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.1)
        except TypeError:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=22, azim=-58)

        axis_suffix = (
            "angular-distance geometry"
            if position_scale == 1.0
            else "centroid display spread x2"
        )
        ax.set_xlabel(
            f"MDS Dim-1 ({axis_suffix})",
            fontsize=9,
            labelpad=2,
        )
        ax.set_ylabel(
            f"MDS Dim-2 ({axis_suffix})",
            fontsize=9,
            labelpad=2,
        )
        ax.set_zlabel(
            f"MDS Dim-3 ({axis_suffix})",
            fontsize=9,
            labelpad=4,
        )

        if position_scale == 1.0:
            scale_description = (
                "centre distance: centroid angular distance"
            )
        else:
            scale_description = (
                "centroid display positions: x2.0 (visual aid)"
            )
        ax.set_title(
            "Class Centroid Sphere Summary\n"
            f"{scale_description} | radius: actual within-class "
            f"p{radius_percentile:g} angular distance"
        )
        ax.legend(
            markerscale=1.15,
            fontsize=8,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(
            left=0.01,
            right=0.78,
            bottom=0.17,
            top=0.94,
        )
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.25,
        )
        plt.close(fig)

    return {
        "centroid_sphere_image": image_path,
        "centroid_sphere_spread_x2_image": spread_x2_image_path,
        "centroid_sphere_spread_x2_position_scale": 2.0,
        "centroid_sphere_csv": csv_path,
        "centroid_sphere_radius_percentile": float(radius_percentile),
        "centroid_sphere_mean_angular_radius_radian": float(
            radii.mean()
        ),
        "centroid_sphere_max_angular_radius_radian": float(
            radii.max()
        ),
        "centroid_sphere_mds_stress": float(mds.stress_),
    }


def _plot_tsne_classes(
    ax,
    tsne_coordinates,
    labels,
    classes,
    label2name,
    colors,
    plot_dim,
    num_special_tokens=2,
):
    """Draw 2D or 3D t-SNE points and visual centroids."""
    visual_centroids = np.stack(
        [
            tsne_coordinates[labels == class_id].mean(axis=0)
            for class_id in classes
        ],
        axis=0,
    ) if len(classes) > 0 else None

    for index, class_id in enumerate(classes):
        mask = labels == class_id
        name = label2name.get(int(class_id), str(int(class_id)))
        is_special = int(class_id) < num_special_tokens
        point_size = 50 if is_special else 12
        point_alpha = 0.5 if is_special else 0.8
        color = colors[index % len(colors)]

        coordinates = [
            tsne_coordinates[mask, dimension]
            for dimension in range(plot_dim)
        ]
        ax.scatter(
            *coordinates,
            s=point_size,
            alpha=point_alpha,
            label=name,
            c=color,
        )

        if not is_special and visual_centroids is not None:
            centroid_coordinates = [
                visual_centroids[index, dimension]
                for dimension in range(plot_dim)
            ]
            ax.scatter(
                *centroid_coordinates,
                s=130,
                marker="X",
                color=color,
                edgecolors="black",
                linewidths=1.0,
            )

    return visual_centroids


def _apply_3d_percentile_zoom(
    ax,
    coordinates,
    lower_percentile=1.0,
    upper_percentile=99.0,
    padding_ratio=0.04,
):
    """Zoom a 3D axis to the central percentile range of t-SNE coordinates."""
    lower = np.percentile(
        coordinates,
        lower_percentile,
        axis=0,
    )
    upper = np.percentile(
        coordinates,
        upper_percentile,
        axis=0,
    )

    span = np.maximum(upper - lower, 1e-6)
    padding = span * padding_ratio

    ax.set_xlim(
        lower[0] - padding[0],
        upper[0] + padding[0],
    )
    ax.set_ylim(
        lower[1] - padding[1],
        upper[1] + padding[1],
    )
    ax.set_zlim(
        lower[2] - padding[2],
        upper[2] + padding[2],
    )

    # A cubic display box makes each projected axis use the available panel
    # area. Using ``span`` here preserves the raw t-SNE aspect ratio, but can
    # make a short axis look unnecessarily compressed in a static 3D image.
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.15)
    except TypeError:
        # ``zoom`` is not available in older Matplotlib versions.
        ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.view_init(elev=22, azim=-58)
    ax.margins(x=0.0, y=0.0, z=0.0)


def _style_tsne_legend(ax, plot_dim, num_classes):
    """Place the legend without hiding the point cloud."""
    if num_classes > 30:
        return

    if plot_dim == 3:
        ax.legend(
            markerscale=1.35,
            fontsize=8,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
    else:
        ax.legend(markerscale=1.5, fontsize=8, ncol=2)


def dunn_index(X, y, metric="cosine", distance_matrix=None):
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

    # Reuse a precomputed matrix when silhouette and Dunn are evaluated on the
    # same validation subset. This avoids the most expensive computation being
    # performed twice.
    if distance_matrix is None:
        D = pairwise_distances(X, metric=metric, n_jobs=-1)
    else:
        D = np.asarray(distance_matrix)
        expected_shape = (len(X), len(X))
        if D.shape != expected_shape:
            raise ValueError(
                f"distance_matrix must have shape {expected_shape}, got {D.shape}"
            )

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

def plot_tsne_with_centroids(config, feats, labels, vocab, save_root=None, file_name=None, visualization=None):
    if save_root is None:
        SAVE_ROOT = config.SAVE_ROOT
    else:
        SAVE_ROOT = save_root
    try:
        import torch
        from torch import Tensor as TorchTensor
        is_torch = isinstance(feats, TorchTensor)
    except Exception:
        is_torch = False

    X = feats.detach().cpu().numpy() if is_torch else np.asarray(feats)
    y = labels.detach().cpu().numpy() if is_torch else np.asarray(labels)

    assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "feats:[N,D], labels:[N]"
    original_N, D = X.shape

    # Silhouette, exact Dunn, and t-SNE all use the same class-balanced subset.
    sample_seed = int(getattr(config.VIS, "TSNE_RANDOM_SEED", 614))
    plot_dim = int(getattr(config.VIS, "PLOT_DIM", 2))
    if plot_dim not in (2, 3):
        raise ValueError(
            "config.VIS.PLOT_DIM must be either 2 or 3"
        )

    samples_per_class = int(
        getattr(config.VIS, "SAMPLES_PER_CLASS", 300)
    )
    if samples_per_class <= 0:
        raise ValueError(
            "config.VIS.SAMPLES_PER_CLASS must be greater than 0"
        )

    X, y = _stratified_sample(
        X,
        y,
        max_samples_per_class=samples_per_class,
        random_state=sample_seed,
    )

    N = len(X)
    classes = np.unique(y)
    C = len(classes)
    label2name = _build_label2name(vocab, classes)

    Xo, yo = X, y

    # If metric == 'cosine', L2 normalization is recommended
    Xo_eval = normalize(Xo, axis=1) if config.VIS.PLOT_METRIC_METHOD == "cosine" else Xo

    # --- inter-class cosine similarity (centroid-based, mean of off-diagonal entries) ---
    class_means = []
    for c in np.unique(yo):
        sel = Xo_eval[yo == c]
        if len(sel) == 0:
            continue
        class_means.append(sel.mean(axis=0))

    metrics = {
        "num_samples": int(N),
        "num_samples_before_sampling": int(original_N),
        "max_samples_per_class": samples_per_class,
        "num_classes": int(C),
        "feat_dim": int(D),
        "tsne_plot_dim": plot_dim,
        f"metric_{D}d": config.VIS.PLOT_METRIC_METHOD,
    }

    if len(class_means) >= 2:
        M = normalize(np.stack(class_means), axis=1)
        cs = M @ M.T
        off = np.sum(cs) - np.trace(cs)
        pairs = len(M) * (len(M) - 1)
        metrics["inter_class_similarity_orig"] = float(off / pairs)
    else:
        metrics["inter_class_similarity_orig"] = float("nan")

    num_special_tokens = int(
        getattr(getattr(config, "DATASET", {}), "NUM_TOKEN", 2)
    )
    joint_cosine_analysis = _joint_centroid_cosine_analysis(
        Xo,
        yo,
        label2name,
        num_special_tokens=num_special_tokens,
    )
    if joint_cosine_analysis is not None:
        metrics["joint_centroid_cosine_matrix_shape"] = list(
            joint_cosine_analysis["cosine_matrix"].shape
        )
        metrics["joint_centroid_cosine_mean_off_diagonal"] = (
            joint_cosine_analysis["mean_off_diagonal"]
        )
        metrics["joint_centroid_cosine_max_off_diagonal"] = (
            joint_cosine_analysis["max_off_diagonal"]
        )
        metrics["joint_centroid_cosine_min_off_diagonal"] = (
            joint_cosine_analysis["min_off_diagonal"]
        )
        metrics["joint_centroid_most_similar_pair"] = (
            joint_cosine_analysis["most_similar_pair"]
        )
        metrics["joint_centroid_top5_similar_pairs"] = (
            joint_cosine_analysis["top5_similar_pairs"]
        )
        metrics["left_right_joint_centroid_cosine"] = (
            joint_cosine_analysis["left_right_cosine"]
        )

    # Compute the O(N^2) distance matrix only once, then share it between
    # silhouette and the exact (non-robust) Dunn Index.
    distance_matrix = None
    if C > 1 and len(Xo_eval) >= 10 and len(Xo_eval) > C:
        try:
            distance_matrix = pairwise_distances(
                Xo_eval,
                metric=config.VIS.PLOT_METRIC_METHOD,
                n_jobs=-1,
            )
        except Exception:
            distance_matrix = None

    # --- original-dimensional silhouette: per-sample, per-cluster, and overall ---
    if C > 1 and len(Xo_eval) >= 10 and len(Xo_eval) > C:
        try:
            if distance_matrix is None:
                raise RuntimeError("Failed to compute pairwise distance matrix")
            s_vals = silhouette_samples(
                distance_matrix,
                yo,
                metric="precomputed",
            )
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
        metrics["dunn_index_orig"] = float(
            dunn_index(
                Xo_eval,
                yo,
                metric=config.VIS.PLOT_METRIC_METHOD,
                distance_matrix=distance_matrix,
            )
        )
    except Exception:
        metrics["dunn_index_orig"] = float("nan")

    # ---------- (B) 2D/3D t-SNE: visualization only ----------
    if visualization:
        if file_name is None:
            file_name = config.FILE_NAME
        else:
            file_name = file_name

        # PCA reduces the 768-dimensional input before t-SNE. The visualization
        # still contains the same class-balanced samples used by the metrics.
        tsne_input = Xo_eval
        pca_components = min(50, tsne_input.shape[0], tsne_input.shape[1])
        if pca_components >= 2 and tsne_input.shape[1] > pca_components:
            tsne_input = PCA(
                n_components=pca_components,
                random_state=sample_seed,
            ).fit_transform(tsne_input)

        requested_perplexity = float(config.VIS.TSNE_PERPLEXITY)
        class_counts = [
            int(np.sum(y == class_id))
            for class_id in np.unique(y)
        ]
        min_class_count = min(class_counts)

        # Perplexity represents the effective number of neighbours considered
        # by t-SNE. When it is larger than the number of samples in the smallest
        # class (for example, 10 samples/class with perplexity 30), cross-class
        # neighbourhoods dominate and visually tight original-space clusters
        # can be drawn as loose clouds. Keep it below the smallest class size.
        perplexity = min(
            requested_perplexity,
            max(1.0, float(min_class_count - 1)),
            max(1.0, float(len(tsne_input) - 1)),
        )
        metrics["tsne_requested_perplexity"] = requested_perplexity
        metrics["tsne_effective_perplexity"] = perplexity
        metrics["tsne_min_class_count"] = min_class_count

        X_tsne = TSNE(
            n_components=plot_dim, init="pca", learning_rate="auto",
            perplexity=perplexity, random_state=sample_seed,
            n_iter=config.VIS.TSNE_N_ITER, method="barnes_hut",
            n_jobs=-1,
        ).fit_transform(tsne_input)

    # ---------- (C) Visualization ----------
        result_root = os.path.join(SAVE_ROOT, file_name)
        img_dir = os.path.join(result_root, "img")
        metrics_dir = os.path.join(result_root, "metrics")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray',
            'olive', 'cyan', 'magenta', 'gold', 'navy', 'lime', 'teal', 'coral',
            'darkred', 'darkblue', 'darkgreen', 'indigo', 'khaki', 'maroon'
        ]

        # (C-1) 512D silhouette plot (left panel)
        #  - x-axis: silhouette coefficient s(i)
        #  - y-axis: samples grouped by cluster (sorted within each cluster)
        #  - overall average shown as a red dashed line
        if s_vals is not None:  # silhouette score / t-sne
            if plot_dim == 2:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                ax_sil, ax_tsne = axes[0], axes[1]
                silhouette_fig = fig
            else:
                # A 3D axes loses too much usable area when it shares a
                # horizontal figure with the silhouette panel. Save the
                # silhouette separately and let the main PNG be a full-size
                # 3D cluster visualization.
                silhouette_fig, ax_sil = plt.subplots(figsize=(10, 8))
                fig = plt.figure(figsize=(12, 9))
                ax_tsne = fig.add_subplot(1, 1, 1, projection="3d")

            y_lower = 10

            # colors = plt.cm.get_cmap("tab20", C)
            # colors = plt.cm.get_cmap("hsv", C)
            #
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
                    alpha=0.7, color=colors[i]
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

            if plot_dim == 3:
                silhouette_fig.tight_layout()
                silhouette_path = os.path.join(
                    img_dir,
                    f"{file_name}_silhouette.png",
                )
                silhouette_fig.savefig(
                    silhouette_path,
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                plt.close(silhouette_fig)
                metrics["silhouette_plot"] = silhouette_path

            # (C-2) 2D/3D t-SNE scatter (right panel)
            classes_tsne = np.unique(y)
            _plot_tsne_classes(
                ax_tsne,
                X_tsne,
                y,
                classes_tsne,
                label2name,
                colors,
                plot_dim,
                num_special_tokens=num_special_tokens,
            )

            ax_tsne.set_title(
                f"t-SNE ({D}D to {plot_dim}D) with centroids\n"
                f"perp={perplexity:g} | sampled N={len(X)}/{original_N} | C={C}"
                + (" | central 98% zoom" if plot_dim == 3 else ""))
            ax_tsne.set_xlabel("Dim-1")
            ax_tsne.set_ylabel("Dim-2")
            if plot_dim == 3:
                ax_tsne.set_zlabel("Dim-3")
                _apply_3d_percentile_zoom(
                    ax_tsne,
                    X_tsne,
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                )
            _style_tsne_legend(
                ax_tsne,
                plot_dim,
                len(classes_tsne),
            )
            if plot_dim == 3:
                fig.subplots_adjust(
                    left=0.01,
                    right=0.78,
                    bottom=0.02,
                    top=0.90,
                )
            else:
                fig.tight_layout()

            png_path = os.path.join(img_dir, f"{file_name}.png")
            fig.savefig(
                png_path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )
            plt.close(fig)

        else:   # only t-sne
            # t-SNE only (re-use label2name here as well)
            fig = plt.figure(figsize=(9, 8))
            if plot_dim == 2:
                ax_tsne = fig.add_subplot(1, 1, 1)
            else:
                ax_tsne = fig.add_subplot(1, 1, 1, projection="3d")

            classes_tsne = np.unique(y)
            _plot_tsne_classes(
                ax_tsne,
                X_tsne,
                y,
                classes_tsne,
                label2name,
                colors,
                plot_dim,
                num_special_tokens=num_special_tokens,
            )
            ax_tsne.set_title(
                f"t-SNE ({D}D to {plot_dim}D) with centroids\n"
                f"perp={perplexity:g} | sampled N={len(X)}/{original_N} | C={C}"
                + (" | central 98% zoom" if plot_dim == 3 else "")
            )
            ax_tsne.set_xlabel("Dim-1")
            ax_tsne.set_ylabel("Dim-2")
            if plot_dim == 3:
                ax_tsne.set_zlabel("Dim-3")
                _apply_3d_percentile_zoom(
                    ax_tsne,
                    X_tsne,
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                )
            _style_tsne_legend(
                ax_tsne,
                plot_dim,
                len(classes_tsne),
            )
            if plot_dim == 3:
                fig.subplots_adjust(
                    left=0.01,
                    right=0.78,
                    bottom=0.02,
                    top=0.90,
                )
            else:
                fig.tight_layout()

            png_path = os.path.join(img_dir, f"{file_name}.png")
            fig.savefig(
                png_path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )
            plt.close(fig)

        if joint_cosine_analysis is not None:
            cosine_output_paths = _save_joint_centroid_cosine_outputs(
                joint_cosine_analysis,
                img_dir,
                metrics_dir,
                file_name,
            )
            metrics.update(cosine_output_paths)

        try:
            centroid_sphere_outputs = _save_centroid_sphere_outputs(
                Xo,
                yo,
                label2name,
                img_dir,
                metrics_dir,
                file_name,
                colors,
                random_state=sample_seed,
                radius_percentile=95.0,
            )
            metrics.update(centroid_sphere_outputs)
        except Exception as error:
            # A secondary visualization must not invalidate the checkpoint
            # evaluation or prevent the other validation outputs being saved.
            metrics["centroid_sphere_error"] = repr(error)

        # Save metrics to file
        with open(os.path.join(metrics_dir, "metrics.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        print(f"SAVE Done ! in {result_root}")

    return metrics
