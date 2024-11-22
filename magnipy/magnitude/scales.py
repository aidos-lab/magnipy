import numpy as np


def get_scales(t_conv, n_ts=10, log_scale=False, one_point_property=True):
    """
    Choose a fixed number of scale parameters
    between zero and the approximated convergence scale
    either evenly-spaced or sampled on a logarithmic scale.

    Parameters
    ----------
    t_conv : float
        The scaling parameter at which the magnitude function reaches the target value i.e.
        the upper bound of the evaluation interval.
    n_ts : int
        The number of evaluation scales that should be sampled.
    log_scale : bool
        If True sample evaluation scales on a logarithmic scale instead of evenly.

    Returns
    -------
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters.
    """
    if one_point_property:
        if log_scale:
            ts_log = np.geomspace(
                t_conv / n_ts, t_conv, n_ts - 1
            )  # np.log(t_conv)
            ts = [0] + [i for i in ts_log]
            ts = np.array(ts)
        else:
            ts = np.linspace(0, t_conv, n_ts)
    else:
        if log_scale:
            ts = np.geomspace(t_conv / n_ts, t_conv, n_ts)
        else:
            ts = np.linspace(t_conv / n_ts, t_conv, n_ts)
    return ts


def scale_when_scattered(D, n=None):
    """
    Compute the scale after which a scaled space is guaranteed to be scattered.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.

    Returns
    -------
    t_scatterd : float
        The scaling parameter at which the space is scattered.

    References
    ----------
    .. [1] Leinster, T., 2013. The magnitude of metric spaces. Documenta Mathematica, 18, pp.857-905.
    """
    if n is None:
        n = D.shape[0]
    return np.log(n - 1) / np.min(D[np.nonzero(D)])


def scale_when_almost_scattered(D, n=None, q=None):
    """
    Compute the scale after which a scaled space is almost scattered.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    q : float
        The quantile to compute. Must be between 0 and 1.
    Returns
    -------
    t_scatterd : float
        The scaling parameter at which the space is almost scattered.
    """
    if n is None:
        n = D.shape[0]
    if q is None:
        q = 1 / n
    return np.log(n - 1) / np.quantile(D[np.nonzero(D)], q=q)


def cut_ts(ts, t_cut):
    """
    Cut off a magnitude function at a specified cut-off scale.
    """
    index_cut = np.searchsorted(ts, t_cut)
    ts_new = np.concatenate((ts[:index_cut], [t_cut]))
    return ts_new
