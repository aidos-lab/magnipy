from magnipy import Magnipy
import numpy as np
from math import e

def test_graph_mag():
    Mag = Magnipy(X=np.array(
        [[0., 1., 2., 2., 3.],
        [1., 0., 1., 1., 2.],
        [2., 1., 0., 1., 2.],
        [2., 1., 1., 0., 1.],
        [3., 2., 2., 1., 0.]]), metric="precomputed", ts=[1])

    q=np.exp(-1)
    analytic = (5+5*q-4*q**2)/((1+q)*(1+2*q))

    assert np.isclose(Mag.get_magnitude()[0][0], analytic)

def test_graph_function():
    ts = np.linspace(0.01, 1, 100)
    Mag = Magnipy(X=np.array(
        [[0., 1., 2., 2., 3.],
        [1., 0., 1., 1., 2.],
        [2., 1., 0., 1., 2.],
        [2., 1., 1., 0., 1.],
        [3., 2., 2., 1., 0.]]),metric="precomputed", ts=ts)

    analytic = []
    for t in ts:
        q = np.exp(-t)
        analytic.append((5+5*q-4*q**2)/((1+q)*(1+2*q)))
    analytic = np.array(analytic)

    assert np.allclose(Mag.get_magnitude()[0], analytic)