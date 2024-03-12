import numpy as np
from scipy.integrate import simpson, trapz
from function_utils import diff_of_functions, cut_until_scale
from matplotlib import pyplot as plt

def area_under_curve(magnitude, ts, integration="trapz", absolute_area=True, scale=True):
    """
    Compute the area under a magnitude function as a summary of magnitude 
    i.e. diversity across multiple scales.

    Parameters
    ----------
    magnitude : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    integration : str
        Use "trapz" or "simpson" integration the approximate the integral.
    absolute_area : bool
        If True take the absolute difference.
    scale : bool
        If True divide the area between the functions by the maximum evaluation scale.

    Returns
    -------
    magnitude_area : float
        The area under the magnitude function.
    
    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024. 
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations. 
        arXiv preprint arXiv:2311.16054.
    """
    if absolute_area:
        magnitude = np.abs(magnitude)

    if integration=="simpson":
        area=simpson(y=magnitude,  x=ts)
    else:
        area=trapz(y=magnitude,  x=ts)
    if scale: 
        area = area / ts[-1]
    return area

def mag_diff(magnitude, ts, D, magnitude2, ts2, D2, method="cholesky", t_cut=None, exact=False, 
            integration="trapz", absolute_area=True, scale=True, plot=False):
    """
    Compute the difference between two magnitude functions via the area 
    between these functions as a summary of the difference in magnitude 
    i.e. the difference in diversity across multiple scales.

    Parameters
    ----------
    magnitude : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    magnitude2 : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts2 : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D2 : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    method : str
        The method used to compute magnitude.
    t_cut : None or float
        The evaluation scale until which to estimate the integral.
        If None evaluate across all pre-defined scales.
    exact : bool
        If True and both D and D2 are not None re-compute the two magnitude functions 
        across the union of their evaluation scales. Else use interpolation.
    integration : str
        Use "trapz" or "simpson" integration the approximate the integral.
    absolute_area : bool
        If True take the absolute difference.
    scale : bool
        If True divide the area between the functions by the maximum evaluation scale.
    plot : bool
        If True plot the difference between the magnitude functions.

    Returns
    -------
    magnitude_diff : float
        The difference between the two magnitude functions.
    
    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024. 
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations. 
        arXiv preprint arXiv:2311.16054.
    """
    
    diff_of_interpolated_vectors, ts_list = diff_of_functions(magnitude, ts, D, magnitude2, ts2, D2, exact=exact, method=method, t_cut=t_cut)

    area = area_under_curve(ts=ts_list, magnitude=diff_of_interpolated_vectors, integration=integration, absolute_area=absolute_area, scale=scale)
    if plot:
        plt.plot(ts_list, diff_of_interpolated_vectors, label="difference between magnitude functions")
        plt.xlabel("t")
        plt.ylabel("difference between magnitude functions")
        plt.title(f"MagDiff {round(area,2)}")
    return area

def mag_area(magnitude, ts, D=None, t_cut=None, integration="trapz", #normalise_by_cardinality=False, 
            absolute_area=True, scale=True, plot=False):
    """
    Compute the area under a magnitude function as a summary of magnitude 
    i.e. diversity across multiple scales.

    Parameters
    ----------
    magnitude : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    t_cut : float or None
        The evaluation scale until which to estimate the integral.
        If None evaluate across all pre-defined scales.
    integration : str
        Use "trapz" or "simpson" integration the approximate the integral.
    absolute_area : bool
        If True take the absolute difference.
    scale : bool
        If True divide the area between the functions by the maximum evaluation scale.
    plot : bool
        If True plot the difference between the magnitude functions.

    Returns
    -------
    magnitude_area : float
        The area under the magnitude function.
    
    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024. 
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations. 
        arXiv preprint arXiv:2311.16054.
    """
    if t_cut is not None:
        magnitude, ts = cut_until_scale(ts, magnitude, t_cut, D=D, method="cholesky")
    area = area_under_curve(ts=ts, magnitude=magnitude, integration=integration, absolute_area=absolute_area, scale=scale)
    
    #if normalise_by_cardinality:
    #    area = area / D.shape[0]

    if plot:
        plt.plot(ts, magnitude, label="magnitude function")
        plt.xlabel("t")
        plt.ylabel("magnitude")
        plt.title(f"MagArea {round(area,2)}")
    return area