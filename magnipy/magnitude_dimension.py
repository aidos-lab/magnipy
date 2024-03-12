import numpy as np

def magitude_dimension_profile(mag, ts, return_log_scale=False):
    """
    Compute the magnitude dimension profile from a pre-computed magntude function
    by approximating the slope of the log-log plot via the slope of the secant
    across the evaluated scales.

    Parameters
    ----------
    mag : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    return_log_scale : bool
        If True output the evaluation scales of the magnitude dimension profile on a logarithmic scale. 
        If False output them on the original scale of ts.
  
    Returns
    -------
    magnitude_dim_profile : array_like, shape (`n_ts`, )
        A vector with the values of the magnitude dimension profile.
    ts_new : array-like, shape (`n_ts`, )
        The scales at which the magnitude dimension profile has been approximated.
    
    References
    ----------
    .. [1] Andreeva, R., Limbeck, K., Rieck, B. and Sarkar, R., 2023. 
        Metric Space Magnitude and Generalisation in Neural Networks. 
        Topological, Algebraic and Geometric Learning Workshop ICML 2023 (pp. 242-253).
    """
    one_point_property=(ts[0]==0)
    if one_point_property:
        log_magnitude = np.log(mag[1:])
        log_ts = np.log(ts[1:])
        ts = ts[1:]
    else:
        log_magnitude = np.log(mag)
        log_ts = np.log(ts)
    slopes = np.diff(log_magnitude)/np.diff(log_ts)
    ts_new_log = log_ts[:-1]+np.diff(log_ts)/2

    if return_log_scale:
        return slopes, ts_new_log
    else:
        ts_new = np.exp(ts_new_log)
        if one_point_property:
            slopes = np.insert(slopes,0,0)
            ts_new = np.insert(ts_new,0,0)
        return slopes, ts_new

def magnitude_dimension(mag_dim_profile):
    """
    Estimate the intrinsic dimensionality i.e. the magnitude dimension of a space 
    by estimating the maximum value of its magnitude dimension profile.

    Parameters
    ----------
    magnitude_dim_profile : array_like, shape (`n_ts`, )
        A vector with the values of the magnitude dimension profile.

    Returns
    -------
    mag_dim : float
        The estimated magnitude dimension.

    References
    ----------
    .. [1] Meckes, M.W., 2015. Magnitude, diversity, capacities, and dimensions of metric spaces. Potential Analysis, 42, pp.549-572.
    """
    return np.max(mag_dim_profile)