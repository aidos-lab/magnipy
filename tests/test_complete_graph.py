from magnipy import Magnipy
import numpy as np
from math import e
import pytest

methods = [
    "cholesky",
    "scipy",
    "scipy_sym",
    "naive",
    "pinv",
    "conjugate_gradient_iteration",
    "cg",
    "spread",
    "spread_torch",
    "naive_torch",
    "cholesky_torch",
    "pinv_torch",
    # "krylov",
]
tss = [[1], np.linspace(0.01, 1, 100), None]


def complete_graph_distance_matrix(n):
    """Generate the distance matrix for a complete graph with n vertices."""
    # Create an n x n matrix filled with 1s
    D = np.ones((n, n), dtype=int)
    # Set the diagonal to 0 (distance from a vertex to itself)
    np.fill_diagonal(D, 0)
    return D


def test_graph_function():
    ## K3,2 has a singularity at t=log(sqrt(2))
    for n in [5, 10, 50, 100]:
        for ts in tss:
            for method in methods:
                Mag = Magnipy(
                    X=complete_graph_distance_matrix(n),
                    metric="precomputed",
                    ts=ts,
                    method=method,
                    n_ts=100,
                )

                mag, ts = Mag.get_magnitude()

                analytic = []
                for t in ts:
                    q = np.exp(-t)

                    # if method in ["spread", "spread_torch"]:
                    #    analytic.append(
                    #        n / (1+(n-1)*q)
                    #    )
                    # else:
                    analytic.append(n / (1 + (n - 1) * q))

                analytic = np.array(analytic)

                np.testing.assert_array_almost_equal(
                    mag,
                    analytic,
                    decimal=4,
                    err_msg="Function complete graph test failed for method: "
                    + method
                    + " and ts: "
                    + str(ts),
                )
