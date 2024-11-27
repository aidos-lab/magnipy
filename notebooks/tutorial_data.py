"""File for creating example datasets for user tutorials."""

from importlib import reload
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_swiss_roll, make_blobs
from sklearn.preprocessing import MinMaxScaler
from magnipy.utils.datasets import (
    sample_points_gaussian,
    sample_points_gaussian_2,
    sample_points_square,
    hawkes_process,
)
from magnipy.utils.plots import plot_points
from magnipy import Magnipy

# from matplotlib import mpl_toolkits
# from mpl_toolkits.mplot3d import Axes3D


def get_Xs():
    # Sample 200 points randomly from the square [0, 2]
    X1 = sample_points_square(200, 2)

    # Sample from hawkes process
    np.random.seed(1)
    X2 = hawkes_process(91, 0.6)
    X2 = (X2 * 2)[:200, :]

    # Sample 100 points each from Gaussians at (0.5, 0.5) and (1.5, 1.5)
    mean2 = [[0.5, 0.5], [1.5, 1.5]]
    cov2 = np.eye(2) * 0.02
    X3 = np.concatenate([sample_points_gaussian(mean, cov2, 100) for mean in mean2])

    # Sample 200 points from a Gaussian centered at (0, 0.5)
    mean1 = [0.5, 0.5]
    cov1 = np.eye(2) * 0.02
    X4 = sample_points_gaussian(mean1, cov1, 200)
    return X1, X2, X3, X4


def plot_spaces(X1, X2, X3, X4):
    fig2, ax2 = plt.subplots(2, 2, figsize=(8, 8))

    axs = [ax2[0, 0], ax2[1, 0], ax2[0, 1], ax2[1, 1]]

    datasets = [X1, X2, X3, X4]

    colors = ["C0", "C1", "C2", "C3"]
    texts = ["X1", "X2", "X3", "X4"]
    names = [
        "X1 random pattern",
        "X2 clustered pattern",
        "X3 two Gaussians",
        "X4 one Gaussian",
    ]

    for i, ax in enumerate(axs):
        plot_points(ax, datasets[i], color=colors[i], label=texts[i])

    for i, ax in enumerate(axs):
        ax.text(
            0.04,
            0.79,
            texts[i],
            transform=ax.transAxes,
            color=colors[i],
            fontsize=60,
            bbox=dict(facecolor="white", alpha=0.9),
        )
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    plt.show()


def normalize(data):
    scaler = MinMaxScaler()
    # Normalize the data
    normalized_data = scaler.fit_transform(data)
    return normalized_data


def get_random(n=1000):
    np.random.seed(0)
    rando = np.random.uniform(0, 10, size=(n, 3)), 3
    rando_data = rando[0]
    rando_df = pd.DataFrame(normalize(rando_data), columns=["x", "y", "z"])
    return rando_data, rando_df


def get_clusters(n=1000):
    # Clusters/blobs
    np.random.seed(54)
    blobs = make_blobs(n, centers=5, n_features=3)[0], 3
    blobs_data = blobs[0]
    blobs_df = pd.DataFrame(normalize(blobs_data), columns=["x", "y", "z"])
    return blobs_data, blobs_df


def get_swiss_roll(n=1000):
    # Swiss roll
    sr = make_swiss_roll(n)[0], 2
    sr_data = sr[0]
    sr_df = pd.DataFrame(normalize(sr_data), columns=["x", "y", "z"])
    return sr_data, sr_df


def plot_df(df, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["x"], df["y"], df["z"], c=df["z"], cmap="viridis", marker="o", s=5)
    plt.title(title)
    plt.show()


def plot_dfs(dfs, titles):
    n = len(dfs)

    # Create a figure with 1 row and n columns of 3D subplots
    fig, axes = plt.subplots(1, n, figsize=(18, 6), subplot_kw={"projection": "3d"})

    for idx in range(0, n):
        df = dfs[idx]
        title = titles[idx]
        color = "C" + str(idx)
        axes[idx].scatter(df["x"], df["y"], df["z"], color=color)
        axes[idx].set_title(title)

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()


def normalize_viridis(matrix):
    scaler = MinMaxScaler(feature_range=(0, 15))
    # Normalize the data
    normalized_data = scaler.fit_transform(matrix)
    return normalized_data


def plot_matrices(matrices, titles):
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(12, 4))

    for idx in range(0, n):
        # normalizing
        # matrix = normalize_viridis(matrices[idx])
        # print(np.max(matrix))
        matrix = matrices[idx]
        title = titles[idx]
        axes[idx].imshow(matrix, cmap="viridis", interpolation="nearest")
        axes[idx].set_title(title)

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()


def plot_matrix_heatmaps(matrices, distance=True):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
    if distance:
        fig.suptitle("Distance Matrices")
        label = "Euclidean Distance"
    else:
        fig.suptitle("Similarity Matrices")
        label = "Similarity"

    # Find the global min and max across all datasets
    vmin = min(matrices[0].min(), matrices[1].min(), matrices[2].min())
    vmax = max(matrices[0].max(), matrices[1].max(), matrices[2].max())

    rando_heatmap = axs[0].imshow(matrices[0], cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("Random")
    axs[0].set_ylabel("Index of Datapoint")
    blob_heatmap = axs[1].imshow(matrices[1], cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("Blobs / Clusters")
    swiss_heatmap = axs[2].imshow(matrices[2], cmap="viridis", vmin=vmin, vmax=vmax)
    axs[2].set_title("Swiss Roll")

    fig.colorbar(
        rando_heatmap,
        ax=axs,
        orientation="horizontal",
        location="bottom",
        label=label,
    )
    plt.show()


def plot_weights(dfs, weights, titles):
    # scaling colorbar
    vmin = min(weights[0][:, 0].min(), weights[1][:, 0].min(), weights[2][:, 0].min())
    vmax = max(
        weights[0][:, -1].max(), weights[1][:, -1].max(), weights[2][:, -1].max()
    )

    n = len(dfs)
    fig, axes = plt.subplots(
        n, 3, figsize=(18, 16), subplot_kw={"projection": "3d"}, constrained_layout=True
    )

    # Assuming 30 t values
    t_idxs = [1, 14, 29]

    for idx in range(0, n):
        df = dfs[idx]
        title = titles[idx]
        weight_vals = weights[idx]
        for t_idx in range(0, 3):
            t = t_idxs[t_idx]
            weights_at_t = weight_vals[:, t]
            plot = axes[idx, t_idx].scatter(
                df["x"],
                df["y"],
                df["z"],
                c=weights_at_t,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if t_idx == 0:
                axes[idx, t_idx].text(
                    0,
                    0,
                    1.7,
                    f"{title}",
                    fontsize=14,
                )
            if idx == 0:
                if t_idx == 0:
                    axes[idx, t_idx].set_title(
                        "$t=min({{t values}} > 0)$ \nSmallest Nonzero Distance Scale"
                    )
                elif t_idx == 1:
                    axes[idx, t_idx].set_title(
                        "$t=1/2 * t_{{conv}}$ \nHalf the Convergence Scale"
                    )
                else:
                    axes[idx, t_idx].set_title("$t=t_{{conv}}$ \nConvergence Scale")

    # Adjust layout and show the figure
    cbar = fig.colorbar(
        plot, ax=axes, aspect=50, shrink=0.8, orientation="horizontal", location="top"
    )
    cbar.set_label("Magnitude Weights", fontsize=20)
    plt.show()
