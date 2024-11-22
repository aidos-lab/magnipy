"""File for creating example datasets for user tutorials."""

from importlib import reload
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_swiss_roll, make_blobs
from magnipy.utils.datasets import (
    sample_points_gaussian,
    sample_points_gaussian_2,
    sample_points_square,
    hawkes_process,
)
from magnipy.utils.plots import plot_points


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

    colors = ["#ee3377", "#ee7733", "#009988", "#0077BB"]
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
