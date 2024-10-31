from magnipy import Magnipy
import numpy as np
from math import e
import pytest

methods = ["cholesky", "scipy", "scipy_sym", "inv", "pinv", "conjugate_gradient_iteration", "cg", "krylov"]
tss= [[1], np.linspace(0.01, 1, 100), None]

def test_graph_function():
    for ts in tss:
        for method in methods:
            Mag = Magnipy(X=np.array(
                [[0., 1., 2., 2., 3.],
                [1., 0., 1., 1., 2.],
                [2., 1., 0., 1., 2.],
                [2., 1., 1., 0., 1.],
                [3., 2., 2., 1., 0.]]),metric="precomputed", ts=ts, method=method, n_ts=100)

            mag, ts = Mag.get_magnitude()

            analytic = []
            for t in ts:
                q = np.exp(-t)
                analytic.append((5+5*q-4*q**2)/((1+q)*(1+2*q)))
            analytic = np.array(analytic)

            assert np.allclose(mag, analytic), "Function graph test failed for method: "+method + " and ts: "+ str(ts)