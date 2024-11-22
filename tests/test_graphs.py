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
    "spread"
    #"krylov",
]
tss = [[1], np.linspace(0.01, 1, 100), None]


def test_graph_function():
    ## K3,2 has a singularity at t=log(sqrt(2))
    for ts in tss:
        for method in methods:
            Mag = Magnipy(
                X=np.array(
                    [
                        [0.0, 1.0, 2.0, 2.0, 3.0],
                        [1.0, 0.0, 1.0, 1.0, 2.0],
                        [2.0, 1.0, 0.0, 1.0, 2.0],
                        [2.0, 1.0, 1.0, 0.0, 1.0],
                        [3.0, 2.0, 2.0, 1.0, 0.0],
                    ]
                ),
                metric="precomputed",
                ts=ts,
                method=method,
                n_ts=100,
            )

            mag, ts = Mag.get_magnitude()

            analytic = []
            for t in ts:
                q = np.exp(-t)

                if method == "spread":
                    analytic.append(
                        2/(1+2*q**2+q+q**3) + 
                        2/(1+3*q+q**2) + 
                        1/(1 + 2*q**2 + 2* q)
                    )
                else:
                    analytic.append(
                        (5 + 5 * q - 4 * q**2) / ((1 + q) * (1 + 2 * q))
                    )

            analytic = np.array(analytic)

            assert np.allclose(mag, analytic), (
                "Function graph test failed for method: "
                + method
                + " and ts: "
                + str(ts)
            )
