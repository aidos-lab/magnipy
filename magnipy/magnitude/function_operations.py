import numpy as np
from scipy.integrate import simpson, trapz
from scipy.interpolate import interp1d
from magnipy.magnitude.compute import compute_magnitude_from_distances
from matplotlib import pyplot as plt


def cut_until_scale(
    ts,
    magnitude,
    t_cut,
    D=None,
    method="cholesky",
    kind="linear",
    magnitude_from_distances=compute_magnitude_from_distances,
):
    """
    Cut off a magnitude function at a specified cut-off scale.

    Parameters
    ----------
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    mag : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    t_cut : float
        The cut-off scale.
    D : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    method : str
        The method used to compute magnitude.
    kind : str
        How to interpolate the function. Only used if D is None.

    Returns
    -------
    magnitude_new : array_like, shape (`n_ts_new`, )
        The first magnitude function interpolated.
    ts_new : array_like, shape (`n_ts_new`, )
        The new evaluation scales cut off at the cut-off scale.
    """
    x_values = ts
    y_values = magnitude

    sorted_indices = np.argsort(x_values)
    x_sorted = x_values[sorted_indices]
    y_sorted = y_values[sorted_indices]

    # Find the index where x_cut fits in the sorted array
    index_cut = np.searchsorted(x_sorted, t_cut)

    if D is None:
        # Perform linear interpolation to find f(x_cut)
        f_x_cut = interp1d(
            x_sorted, y_sorted, kind=kind, fill_value="extrapolate"
        )(t_cut)
    else:
        f_x_cut = magnitude_from_distances(D, [t_cut], method)[0]

    # Create new vectors up to and including t_cut
    ts = np.concatenate((x_sorted[:index_cut], [t_cut]))
    magnitude = np.concatenate((y_sorted[:index_cut], [f_x_cut]))
    return magnitude, ts


def interpolate_functions(mag, ts, mag2, ts2, kind="linear"):
    """
    Interpolate two magnitude functions across the same scales.

    Parameters
    ----------
    mag : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    mag2 : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts2 : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D2 : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    method : str
        The method used to compute magnitude.
    kind : str
        How to interpolate the functions.

    Returns
    -------
    magnitude_new : array_like, shape (`n_ts_new`, )
        The first magnitude function interpolated.
    magnitude_new2 : array_like, shape (`n_ts_new`, )
        The second magnitude function interpolated.
    ts_combined : array_like, shape (`n_ts_new`, )
        The union of the evaluation scales of both functions.
    """
    xs = np.union1d(ts, ts2)
    xs_list = np.sort(xs)
    common_length = xs_list.shape[0]

    # Initialize an array to store the sum of interpolated vectors
    # sum_of_interpolated_vectors = np.zeros(common_length)

    inter1 = interp1d(
        ts, mag, kind=kind, fill_value=(1, np.max(mag)), bounds_error=False
    )  # kind='quadratic'
    inter2 = interp1d(
        ts2, mag2, kind=kind, fill_value=(1, np.max(mag2)), bounds_error=False
    )
    interpolated = inter1(xs_list)
    interpolated2 = inter2(xs_list)
    return interpolated, interpolated2, xs_list


def get_reevaluated_function(
    mag,
    ts,
    ts2,
    D,
    method="cholesky",
    magnitude_from_distances=compute_magnitude_from_distances,
):
    """
    Re-evaluate a magnitude function across more scales.

    Parameters
    ----------
    mag : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which magnitude has been computed.
    D : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    ts2 : array-like, shape (`n_ts`, )
        A vector of new scaling parameters at which to evaluate magnitude.
    method : str
        The method used to compute magnitude.

    Returns
    -------
    magnitude_new : array_like, shape (`n_ts_new`, )
        The magnitude function re-evaluated across more scales.
    ts_combined : array_like, shape (`n_ts_new`, )
        The union of the evaluation scales.
    """
    ts_diff = np.setdiff1d(ts2, ts)  # t in ts2 but not in ts
    mag_new = magnitude_from_distances(D, ts_diff, method=method)
    new_ts = np.concatenate((ts_diff, ts))
    new_mags = np.concatenate((mag_new, mag))
    ind = new_ts.argsort()
    new_ts = new_ts[ind]
    new_mags = new_mags[ind]
    return new_mags, new_ts


def reevaluate_functions(
    mag,
    ts,
    D,
    mag2,
    ts2,
    D2,
    method="cholesky",
    magnitude_from_distances=compute_magnitude_from_distances,
    magnitude_from_distances2=compute_magnitude_from_distances,
):
    """
    Re-evaluate two magnitude functions across the same scales.

    Parameters
    ----------
    mag : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    mag2 : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts2 : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D2 : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    method : str
        The method used to compute magnitude.

    Returns
    -------
    magnitude_new : array_like, shape (`n_ts_new`, )
        The first magnitude function re-evaluated.
    magnitude_new2 : array_like, shape (`n_ts_new`, )
        The second magnitude function re-evaluated.
    ts_combined : array_like, shape (`n_ts_new`, )
        The union of the evaluation scales of both functions.
    """
    new_mags, new_ts = get_reevaluated_function(
        mag,
        ts,
        ts2,
        D,
        method=method,
        magnitude_from_distances=magnitude_from_distances,
    )
    new_mags2, new_ts2 = get_reevaluated_function(
        mag2,
        ts2,
        ts,
        D2,
        method=method,
        magnitude_from_distances=magnitude_from_distances2,
    )
    return new_mags, new_mags2, new_ts


def combine_functions(
    mag,
    ts,
    D,
    mag2,
    ts2,
    D2,
    method="cholesky",
    exact=False,
    t_cut=None,
    addition=False,
    magnitude_from_distances=compute_magnitude_from_distances,
    magnitude_from_distances2=compute_magnitude_from_distances,
):
    """
    Add or subtract two magnitude functions.

    Parameters
    ----------
    mag : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    mag2 : array_like, shape (`n_ts`, )
        A vector of the values of the magnitude function evaluated at the scales ts.
    ts2 : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    D2 : None or array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    method : str
        The method used to compute magnitude.
    exact : bool
        If True and both D and D2 are not None re-compute the two magnitude functions
        across the union of their evaluation scales. Else use interpolation.
    t_cut : None or float
        The evaluation scale until which to estimate the integral.
        If None evaluate across all pre-defined scales.
    addition : bool
        If True add the functions. Else subtract the second from the first function.

    Returns
    -------
    magnitude_combined : array_like, shape (`n_ts_new`, )
        The sum or the difference the two magnitude functions.
    ts_combined : array_like, shape (`n_ts_new`, )
        The union of the evaluation scales of both functions.
    """
    if t_cut is not None:
        mag, ts = cut_until_scale(
            ts,
            mag,
            t_cut,
            D=D,
            method=method,
            magnitude_from_distances=magnitude_from_distances,
        )
        mag2, ts2 = cut_until_scale(
            ts2,
            mag2,
            t_cut,
            D=D2,
            method=method,
            magnitude_from_distances=magnitude_from_distances2,
        )

    if ts is ts2:
        interpolated = mag
        interpolated2 = mag2
        xs_list = ts
    else:
        if (exact) | (D is None) | (D2 is None):
            try:
                interpolated, interpolated2, xs_list = interpolate_functions(
                    mag, ts, mag2, ts2, kind="quadratic"
                )
            except Exception as e:
                # print(e)
                interpolated, interpolated2, xs_list = interpolate_functions(
                    mag, ts, mag2, ts2, kind="linear"
                )
            # interpolated, interpolated2, xs_list = reevaluate_functions(mag, ts, D, mag2, ts2, D2, method=method)
        else:
            interpolated, interpolated2, xs_list = reevaluate_functions(
                mag,
                ts,
                D,
                mag2,
                ts2,
                D2,
                method=method,
                magnitude_from_distances=magnitude_from_distances,
                magnitude_from_distances2=magnitude_from_distances2,
            )

    if addition:
        sum_of_interpolated_vectors = interpolated + interpolated2
    else:
        sum_of_interpolated_vectors = interpolated - interpolated2
    return sum_of_interpolated_vectors, xs_list


def diff_of_functions(
    mag,
    ts,
    D,
    mag2,
    ts2,
    D2,
    exact=False,
    method="cholesky",
    t_cut=None,
    magnitude_from_distances=compute_magnitude_from_distances,
    magnitude_from_distances2=compute_magnitude_from_distances,
):
    """
    Subtract two magnitude functions.
    """
    return combine_functions(
        mag,
        ts,
        D,
        mag2,
        ts2,
        D2,
        exact=exact,
        method=method,
        t_cut=t_cut,
        addition=False,
        magnitude_from_distances=magnitude_from_distances,
        magnitude_from_distances2=magnitude_from_distances2,
    )


def sum_of_functions(
    mag,
    ts,
    D,
    mag2,
    ts2,
    D2,
    exact=False,
    method="cholesky",
    t_cut=None,
    magnitude_from_distances=compute_magnitude_from_distances,
    magnitude_from_distances2=compute_magnitude_from_distances,
):
    """
    Add two magnitude functions.
    """
    return combine_functions(
        mag,
        ts,
        D,
        mag2,
        ts2,
        D2,
        exact=exact,
        method=method,
        t_cut=t_cut,
        addition=True,
        magnitude_from_distances=magnitude_from_distances,
        magnitude_from_distances2=magnitude_from_distances2,
    )


def area_under_curve(
    magnitude, ts, integration="trapz", absolute_area=True, scale=True
):
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

    if integration == "simpson":
        area = simpson(y=magnitude, x=ts)
    else:
        area = trapz(y=magnitude, x=ts)
    if scale:
        area = area / ts[-1]
    return area


def mag_diff(
    magnitude,
    ts,
    D,
    magnitude2,
    ts2,
    D2,
    method="cholesky",
    t_cut=None,
    exact=False,
    integration="trapz",
    absolute_area=True,
    scale=True,
    plot=False,
    name="",
):
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

    diff_of_interpolated_vectors, ts_list = diff_of_functions(
        magnitude,
        ts,
        D,
        magnitude2,
        ts2,
        D2,
        exact=exact,
        method=method,
        t_cut=t_cut,
    )

    area = area_under_curve(
        ts=ts_list,
        magnitude=diff_of_interpolated_vectors,
        integration=integration,
        absolute_area=absolute_area,
        scale=scale,
    )
    if plot:
        plt.plot(
            ts_list,
            diff_of_interpolated_vectors,
            label="difference between magnitude functions " + name,
        )
        plt.xlabel("t")
        plt.ylabel("difference between magnitude functions")
        plt.title(f"MagDiff {round(area,2)}")
    return area


def mag_area(
    magnitude,
    ts,
    D=None,
    t_cut=None,
    integration="trapz",  # normalise_by_cardinality=False,
    absolute_area=True,
    scale=True,
    plot=False,
    name="",
):
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
        magnitude, ts = cut_until_scale(
            ts=ts, magnitude=magnitude, t_cut=t_cut, D=D, method="cholesky"
        )
    area = area_under_curve(
        ts=ts,
        magnitude=magnitude,
        integration=integration,
        absolute_area=absolute_area,
        scale=scale,
    )

    # if normalise_by_cardinality:
    #    area = area / D.shape[0]

    if plot:
        plt.plot(ts, magnitude, label="magnitude function " + name)
        plt.xlabel("t")
        plt.ylabel("magnitude")
        plt.title(f"MagArea {round(area,2)}")
    return area
