from magnipy import Magnipy
import numpy as np
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
]  # , "krylov"]

tss = [[1], np.linspace(0.01, 1, 100), None]


def tes_ts():
    ts = np.linspace(0, 1, 100)
    Mag = Magnipy(X=np.array([[0], [1]]), ts=ts)

    assert np.assert_array_equal(Mag.get_scales(), ts)


def test_fun():
    for ts in tss:
        # ts = np.linspace(0, 1, 100)
        for method in methods:
            Mag = Magnipy(
                X=np.array([[0], [1]]), ts=ts, method=method, n_ts=100
            )

            mag, ts = Mag.get_magnitude()

            analytic = np.array([2 / (1 + np.exp(-t)) for t in ts])

            np.testing.assert_array_almost_equal(
                mag,
                analytic,
                decimal=4,
                err_msg="Function test failed for method: "
                + method
                + " and ts: "
                + str(ts),
            )


def test_weights():
    # ts = np.linspace(0, 1, 100)
    for ts in tss:
        for method in methods:
            Mag = Magnipy(X=np.array([[0], [1]]), ts=ts, n_ts=100)

            mag, ts = Mag.get_magnitude_weights()

            weights = np.zeros((2, len(ts)))
            for i in range(0, len(ts)):
                if ts[i] == 0:
                    w = 1 / 2
                else:
                    w = 1 / (1 + np.exp(-ts[i]))
                weights[:, i] = [w, w]

            np.testing.assert_array_almost_equal(
                mag,
                weights,
                decimal=4,
                err_msg="Weight test failed for method: "
                + method
                + " and ts: "
                + str(ts),
            )
