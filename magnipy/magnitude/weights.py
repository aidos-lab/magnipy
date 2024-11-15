import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import solve_triangular, solve
from scipy.sparse.linalg import cg
import numexpr as ne


def weights_cholesky(Z):
    """
    Compute the magnitude weight vector from a similarity matrix using Cholesky inversion.

    Parameters
    ----------
    Z : array_like, shape (`n_obs`, `n_obs`)
        The similarity matrix.

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, )
        The magnitue weight vector.
    """
    c, lower = cho_factor(Z)
    x = solve_triangular(c, np.ones(Z.shape[0]), trans=1)
    w = solve_triangular(c, x.T, trans=0)
    return w


def weights_naive(Z):
    """
    Compute the magnitude weight vector from a similarity matrix by inverting
    the whole similarity matrix using numpy.inv.

    Parameters
    ----------
    Z : array_like, shape (`n_obs`, `n_obs`)
        The similarity matrix.

    Returns
    -------
    w : array_like, shape (`n_ts`, )
        The magnitue weight vector.
    """
    M = np.linalg.inv(Z)
    return M.sum(axis=1)


def weights_pinv(Z):
    """
    Compute the magnitude weight vector from a similarity matrix by inverting
    the whole similarity matrix using pseudo-inversion with numpy.pinv.

    Parameters
    ----------
    Z : array_like, shape (`n_obs`, `n_obs`)
        The similarity matrix.

    Returns
    -------
    w : array_like, shape (`n_ts`, )
        The magnitue weight vector.
    """
    M = np.linalg.pinv(Z)
    return M.sum(axis=1)


def weights_scipy(Z):
    """
    Compute the magnitude weight vector from a similarity matrix by solving for
    the row sums with scipy.solve assuming the similarity matrix is
    positive definite.

    Parameters
    ----------
    Z : array_like, shape (`n_obs`, `n_obs`)
        The similarity matrix.

    Returns
    -------
    w : array_like, shape (`n_ts`, )
        The magnitue weight vector.
    """
    w = solve(Z, np.ones(Z.shape[0]), assume_a="pos")
    return w


def weights_scipy_sym(Z):
    """
    Compute the magnitude weight vector from a similarity matrix by solving for
    the row sums with scipy.solve assuming the similarity matrix is
    positive definite.

    Parameters
    ----------
    Z : array_like, shape (`n_obs`, `n_obs`)
        The similarity matrix.

    Returns
    -------
    w : array_like, shape (`n_ts`, )
        The magnitue weight vector.
    """
    w = solve(Z, np.ones(Z.shape[0]), assume_a="sym")
    return w


def weights_cg(Z):
    """
    Compute the magnitude weight vector from a similarity matrix
    using conjugate gradient iteration at one scale.

    Parameters
    ----------
    Z : array_like, shape (`n_obs`, `n_obs`)
        The similarity matrix.

    Returns
    -------
    w : array_like, shape (`n_ts`, )
        The magnitue weight vector.
    """
    ones = np.ones(Z.shape[0])
    w, _ = cg(Z, ones, atol=1e-3)
    return w


# def weights_from_similarities_krylov(Z, ts, positive_definite = True):
#    """""
#    Compute magnitude weights from a similarity matrix across a fixed choice of scales
#    using pre-conditioned conjugate gradient iteration as implemented by Shilan (2021). #
#
#    Parameters
#    ----------
#    D : array_like, shape (`n_obs`, `n_obs`)
#        A matrix of distances.
#    ts : array-like, shape (`n_ts`, )
#        A vector of scaling parameters at which to evaluate magnitude.
#
#    Returns
#    -------
#    weights : array_like, shape (`n_obs`, `n_ts`)
#        A matrix with the magnitude weights (whose ij-th entry is the magnitude weight
#        of the ith observation evaluated at the jth scaling parameter).#
#
#    References
#    ----------
#    .. [1] from the PhD thesis of Salim, Shilan (2021)
#    """""
#    n=Z.shape[0]
#    weights = np.zeros(shape=(n, len(ts)))
#    w = np.ones(n)/n
#    for i in range(len(ts)):
#        linear_system = LinearSystem(Z**(ts[i]), np.ones(n), self_adjoint = True, positive_definite = positive_definite)
#        w = Cg(linear_system,  x0 = w).xk
#        weights[:,i]=w.squeeze()
#    return weights


def weights_from_similarities_cg(Z, ts):
    """
    Compute magnitude weights from a distance matrix across a fixed choice of scales
    using pre-conditioned conjugate gradient iteration.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.

    Returns
    -------
    weights : array_like, shape (`n_obs`, `n_ts`)
        A matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    """
    n = Z.shape[0]
    weights = np.zeros(shape=(n, len(ts)))
    w = np.ones(n) / n
    for i in range(len(ts)):
        # associated similarity matrix
        # Z = np.exp(-ts[i]*D)
        w, _ = cg(Z ** (ts[i]), np.ones(n), w)
        weights[:, i] = w.squeeze()
    return weights


def weights_spread(Z):
    """
    Compute the spread weight vector from a similarity matrix.

    Parameters
    ----------
    Z : array_like, shape (`n_obs`, `n_obs`)
        The similarity matrix.

    Returns
    -------
    w : array_like, shape (`n_ts`, )
        The spread weight vector.

    References
    ----------
    .. [1] Willerton S. Spread: a measure of the size of metric spaces. International Journal of
        Computational Geometry & Applications. 2015;25(03):207-25.
    """
    return 1 / np.sum(Z, axis=0)


def spread_weights(Z, ts):
    """
    Compute the spread weights from a distance matrix across a fixed choice of scales.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    mag_fn : function
        A function that computes the magnitude weight vector from a similarity matrix.

    Returns
    -------
    weights : array_like, shape (`n_obs`, `n_ts`)
        A matrix with the magnitude weights (whose ij-th entry is the spread weight
        of the ith observation evaluated at the jth scaling parameter).

    References
    ----------
    .. [1] Willerton S. Spread: a measure of the size of metric spaces. International Journal of
        Computational Geometry & Applications. 2015;25(03):207-25.
    """
    n = Z.shape[0]
    weights = np.ones(shape=(n, len(ts))) / n

    for i, t in enumerate(ts):
        # Z = np.exp(-t * D)
        weights[:, i] = weights_spread(Z**t)
    return weights


def magnitude_weights(
    Z, ts, mag_fn, one_point_property=True, perturb_singularities=True
):
    """
    Compute the magnitude weights from a distance matrix across a fixed choice of scales.
    Whenever the similarity matrix is not invertible, a small amount of constant noise is added
    the similarity matrix as implemented by Bunch et al. (2020) and the inversion is attempted again.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    mag_fn : function
        A function that computes the magnitude weight vector from a similarity matrix.

    Returns
    -------
    weights : array_like, shape (`n_obs`, `n_ts`)
        A matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).

    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024.
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations.
        arXiv preprint arXiv:2311.16054.
    .. [2] Bunch, E., Dickinson, D., Kline, J. and Fung, G., 2020.
        Practical applications of metric space magnitude and weighting vectors.
        arXiv preprint arXiv:2006.14063.
    .. [3] Leinster, T., 2013. The magnitude of metric spaces. Documenta Mathematica, 18, pp.857-905.
    """
    n = Z.shape[0]
    weights = np.ones(shape=(n, len(ts))) / n

    for i, t in enumerate(ts):
        # see above loop
        if t == 0:
            if one_point_property:
                weights[:, i] = np.ones(n) / n
            else:
                weights[:, i] = np.full((n, n), np.nan)
                # raise Exception("We cannot compute magnitude at t=0 unless we assume the one point property!")
        else:
            # if checksingularity():
            #     print(warning)
            try:
                weights[:, i] = mag_fn(Z**t)
            except Exception as e:
                if perturb_singularities:
                    print(f"Exception: {e} for t: {t} perturbing matrix")
                    Z_new = Z**t + 0.01 * np.identity(
                        n=n
                    )  # perturb similarity mtx to invert
                    weights[:, i] = mag_fn(Z_new)
                else:
                    raise Exception(f"Exception: {e} for t: {t}")
    return weights  # np.array(


def positive_weights_only(weights):
    """
    Ensure that the magnitude weights are positive.

    Parameters
    ----------
    weights : array_like, shape (`n_obs`, `n_ts`)
        A matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).

    Returns
    -------
    weights : array_like, shape (`n_obs`, `n_ts`)
        A matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    """
    return np.maximum(weights, 0)


def magnitude_from_weights(weights):
    """
    Compute the magnitude function from the magnitude weights.

    Parameters
    ----------
    weights : array_like, shape (`n_obs`, `n_ts`)
        A matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, )
        A vector with the values of the magnitude function.
    """
    return weights.sum(axis=0)


def similarity_matrix(D):
    # n = D.shape[0]
    Z = np.zeros(D.shape)
    ne.evaluate("exp(-D)", out=Z)
    return Z
