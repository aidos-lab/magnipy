import numpy as np
from magnitude import compute_t_conv, get_scales, magnitude_from_distances, compute_magnitude_until_convergence
from distances import get_dist

def compute_magnitude_dimension_profile(X, ts=None, h=None, target_value=None, n_ts=10, log_scale = True, method="cholesky",
                                       metric="Lp", p=2, normalise_by_diameter=False, one_point_property=False, 
                        n_neighbors=12, exact=False, return_log_scale=False):
    D = get_dist(X, p=p, metric=metric, normalise_by_diameter=normalise_by_diameter, n_neighbors=n_neighbors)
    if exact:
        slopes, ts = magnitude_dimension_profile_exact(D, ts=ts,h=h, n_ts=n_ts, method=method, target_value=target_value,
                                                            log_scale = log_scale, one_point_property=one_point_property, 
                                                            return_log_scale=return_log_scale)
    else:
        magnitude, ts_mag = compute_magnitude_until_convergence(D, ts=ts, n_ts=n_ts, method=method, 
                                                        log_scale = log_scale, get_weights=False)
        slopes, ts = magitude_dimension_profile(magnitude, ts_mag, return_log_scale=return_log_scale, 
                                                one_point_property=one_point_property)
    return slopes, ts

def magnitude_dimension_profile_exact(D, ts=None, h=None, target_value=None, n_ts=10, log_scale = True,
                                      return_log_scale=False, one_point_property=True, method="cholesky"):
    if D.shape[0] != D.shape[1]:
        raise Exception("D must be symmetric.")
    if ts is None:
        t_conv = compute_t_conv(D, target_value=target_value, method=method)
        if one_point_property & ~return_log_scale:
            n_tsss = n_ts-1
        else:
            n_tsss = n_ts
        ts = get_scales(t_conv, n_tsss, log_scale = log_scale, one_point_property = False)#[1:] ## remove the first scale for now
        h = np.min(np.diff(np.log(ts)))/5
        log_ts= np.log(ts)
        #h = np.log((ts[1]-ts[0]))/100
    else:
        if ts[0] == 0:
            ts = ts[1:]
        log_ts = np.log(ts)
        if h is None:
            h = np.min(np.diff(np.log(ts)))
        elif h >= np.min(np.diff(np.log(ts))):
            raise Exception("h must be smaller than the minimum difference between the log-scales.") 
        
    lower_ts = np.exp(log_ts - h)
    upper_ts = np.exp(log_ts + h)
    lower = np.log(magnitude_from_distances(D, lower_ts, method=method, get_weights=False))
    upper = np.log(magnitude_from_distances(D, upper_ts, method=method, get_weights=False))
    slopes = (upper-lower)/(2*h)
    if return_log_scale:
        return slopes, log_ts
    else:
        if one_point_property:
            ts = np.insert(ts,0,0)
            slopes = np.insert(slopes,0,0)
        return slopes, ts

def magitude_dimension_profile(mag, ts, return_log_scale=False, one_point_property=True):
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
    #one_point_property=(ts[0]==0)
    if ts[0]==0:
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