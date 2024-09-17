from magnipy import Magnipy
import numpy as np

def test_mag():
    Mag = Magnipy(X=np.array([[0],[1]]), ts=[1])

    analytic = 2/(1+np.exp(-1))

    assert Mag.get_magnitude()[0][0] == analytic

def tes_ts():
    ts = np.linspace(0, 1, 100)
    Mag = Magnipy(X=np.array([[0],[1]]), ts=ts)

    assert np.assert_array_equal(Mag.get_scales(), ts)

def test_fun():
    ts = np.linspace(0, 1, 100)
    Mag = Magnipy(X=np.array([[0],[1]]), ts=ts)

    analytic = np.array([2/(1+np.exp(-t)) for t in ts])

    np.testing.assert_array_almost_equal(Mag.get_magnitude()[0], analytic)

def test_weights():
    ts = np.linspace(0, 1, 100)
    Mag = Magnipy(X=np.array([[0],[1]]), ts=ts)

    weights = np.zeros((2, 100))
    weights[:,0] = [0.5, 0.5]
    for i in range(1,100):
        w = 1/(1+np.exp(-ts[i]))
        weights[:,i] = [w, w]

    np.testing.assert_array_almost_equal(Mag.get_magnitude_weights()[0], weights)
