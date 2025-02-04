import numpy as np
from magnipy.magnitude.distances import get_dist
import numexpr as ne
from magnipy.magnitude.weights import *
from magnipy.magnitude.scales import get_scales
from magnipy.magnitude.convergence import guess_convergence_scale


def compute_magnitude_from_distances(
    D,
    ts=np.arange(0.01, 5, 0.01),
    method="cholesky",
    get_weights=False,
    one_point_property=True,
    perturb_singularities=True,
    positive_magnitude=False,
    input_distances=True,
    **parameters
):
    """
    Compute the magnitude function of magnitude weights from a distance matrix
    across a fixed choice of scales.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    method : str
        The method used to compute magnitude.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).

    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024.
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations.
        arXiv preprint arXiv:2311.16054.
    """
    # TODO only check if not checked before
    if D.shape[0] != D.shape[1]:
        raise Exception("D must be symmetric.")
    if D.shape[0] == 1:
        weights = np.ones(shape=(1, len(ts)))

    if input_distances:
        Z = similarity_matrix(D)
    else:
        Z = D

    if method == "spread":
        weights = spread_weights(Z, ts)
    # elif method=="krylov":
    #    weights = weights_from_similarities_krylov(Z, ts)
    elif method == "cg":
        weights = weights_from_similarities_cg(Z, ts)
    else:
        if method == "scipy":
            mag_fn = weights_scipy
        elif method == "scipy_sym":
            mag_fn = weights_scipy_sym
        elif method == "cholesky":
            mag_fn = weights_cholesky
        elif method == "conjugate_gradient_iteration":
            mag_fn = weights_cg
        elif method == "pinv":
            mag_fn = weights_pinv
        elif method == "naive":
            mag_fn = weights_naive
        elif method == "spread_torch":
            mag_fn = weights_spread_torch
            Z = torch.tensor(Z)
        elif method == "lstq_torch":
            mag_fn = weights_lstq_torch
            Z = torch.tensor(Z)
        elif method == "naive_torch":
            mag_fn = weights_naive_torch
            Z = torch.tensor(Z)
        elif method == "pinv_torch":
            mag_fn = weights_pinv_torch
            Z = torch.tensor(Z)
        elif method == "cholesky_torch":
            mag_fn = weights_cholesky_torch
            Z = torch.tensor(Z)
        elif method == "solve_torch":
            Z = torch.tensor(Z)
            mag_fn = weights_solve_torch
        elif method == "iterative_normalization":
            from magnipy.magnitude.approximation import add_and_normalize
            Z = torch.tensor(Z, dtype=torch.float)
            def mag_fn(Z):
                return add_and_normalize(Z, h=parameters.get("h", 100))
            #mag_fn = weights_iterative_normalization
        elif method == "sgd":
            from magnipy.magnitude.approximation import magnitude_by_SGD
            Z = torch.tensor(Z, dtype=torch.float)
            def mag_fn(Z):
                return magnitude_by_SGD(Z, h=parameters.get("h", 100), lr=parameters.get("lr", 1e-10), device=parameters.get("device", "cpu")).detach().numpy()
        elif method == "batch_sgd":
            from magnipy.magnitude.approximation import magnitude_by_batch_SGD
            Z = torch.tensor(Z, dtype=torch.float)
            def mag_fn(Z):
                return magnitude_by_batch_SGD(Z, h=parameters.get("h", 100), batch_size = parameters.get("batch_size", 1),
                lr=parameters.get("lr", 1e-10), device=parameters.get("device", "cpu")).detach().numpy()
            #mag_fn = magnitude_by_batch_SGD
        #elif method == "greedy_maximization":
        #    from magnipy.magnitude.approximation import greedy_maximization
        #    Z = torch.tensor(Z, dtype=torch.float)
        #    def mag_fn(Z):
        #        return greedy_maximization(Z, tolerance_parameter=parameters.get("lr", 1e-10), device=parameters.get("device", "cpu"))
        #elif method == "discrete_center":
        #    from magnipy.magnitude.approximation import discrete_center_hierarchy
        #    Z = torch.tensor(Z, dtype=torch.float)
        #    def mag_fn(Z):
        #        return discrete_center_hierarchy(Z, device=parameters.get("device", "cpu"))[1]
        else:
            raise Exception(
                "The computation method must be one of 'cholesky', 'scipy', 'scipy_sym', 'pinv', 'conjugate_gradient_iteration', 'cg', 'naive', 'spread'."
            )

        weights = magnitude_weights(
            Z,
            ts,
            mag_fn,
            one_point_property=one_point_property,
            perturb_singularities=perturb_singularities,
        )

    if positive_magnitude:
        weights = positive_weights_only(weights)

    if get_weights:
        return weights
    else:
        return magnitude_from_weights(weights)


def compute_magnitude_until_convergence(
    D,
    ts=None,
    target_value=None,
    n_ts=10,
    log_scale=False,
    method="cholesky",
    get_weights=False,
    one_point_property=True,
    perturb_singularities=True,
    positive_magnitude=False,
    input_distances=True,
):
    """
    Compute the magnitude function of magnitude weights from a distance matrix
    either across a fixed choice of scales
    or until the magnitude function has reached a certain target value.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    ts : None or array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
        Alternativally, if ts is None, the evaluation scales will be choosen automatically.
    target_value : float
        The value of margnitude that should be reached. Only used if ts is None.
    n_ts : int
        The number of evaluation scales that should be sampled. Only used if ts is None.
    log_scale : bool
        If True sample evaluation scales on a logarithmic scale instead of evenly. Only used if ts is None.
    method : str
        The method used to compute magnitude.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    ts : array_like, shape (`n_ts`, )
        The scales at which magnitude has been evaluated.

    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024.
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations.
        arXiv preprint arXiv:2311.16054.
    """
    if D.shape[0] != D.shape[1]:
        raise Exception("D must be symmetric.")

    if input_distances:
        Z = similarity_matrix(D)
    else:
        Z = D

    if ts is None:
        t_conv = compute_t_conv(
            Z,
            target_value=target_value,
            method=method,
            input_distances=False,
            positive_magnitude=positive_magnitude,
        )
        ts = get_scales(
            t_conv,
            n_ts,
            log_scale=log_scale,
            one_point_property=one_point_property,
        )
        # print(f"Evaluate magnitude at {self._n_ts} scales between 0 and the approximate convergence scale {self._t_conv}")
    return (
        compute_magnitude_from_distances(
            Z,
            ts,
            method=method,
            get_weights=get_weights,
            one_point_property=one_point_property,
            perturb_singularities=perturb_singularities,
            positive_magnitude=positive_magnitude,
            input_distances=False,
        ),
        ts,
    )


def compute_magnitude(
    X,
    ts=None,
    target_value=None,
    n_ts=10,
    log_scale=False,
    method="cholesky",
    get_weights=False,
    metric="Lp",
    p=2,
    normalise_by_diameter=False,
    n_neighbors=12,
    one_point_property=True,
    perturb_singularities=True,
    positive_magnitude=False,
):
    """
    Compute the magnitude function of magnitude weights given a dataset
    either across a fixed choice of scales
    or until the magnitude function has reached a certain target value.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    ts : None or array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
        Alternativally, if ts is None, the evaluation scales will be choosen automatically.
    target_value : float
        The value of margnitude that should be reached. Only used if ts is None.
    n_ts : int
        The number of evaluation scales that should be sampled. Only used if ts is None.
    log_scale : bool
        If True sample evaluation scales on a logarithmic scale instead of evenly. Only used if ts is None.
    method : str
        The method used to compute magnitude.
    get_weights : bool
        If True output the magnitude weights. If False output the magnitude function.
    metric: str
        The distance metric to use. The distance function can be
        'Lp', 'isomap',
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule'.
    p: float
        Parameter for the Minkowski metric.
    normalise_by_diameter: bool
        If True normalise all distances (and hence also the scaling parameters) by the diameter.
    n_neighbors : int
        The number of nearest neighbours used to compute geodesic distances.
        Only used if the metric is "isomap".

    Returns
    -------
    magnitude : array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        Either a vector with the values of the magnitude function
        or a matrix with the magnitude weights (whose ij-th entry is the magnitude weight
        of the ith observation evaluated at the jth scaling parameter).
    ts : array_like, shape (`n_ts`, )
        The scales at which magnitude has been evaluated.

    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024.
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations.
        arXiv preprint arXiv:2311.16054.
    """
    D = get_dist(
        X,
        p=p,
        metric=metric,
        normalise_by_diameter=normalise_by_diameter,
        n_neighbors=n_neighbors,
    )
    Z = similarity_matrix(D)
    magnitude, ts = compute_magnitude_until_convergence(
        Z,
        ts=ts,
        n_ts=n_ts,
        method=method,
        target_value=target_value,
        log_scale=log_scale,
        get_weights=get_weights,
        one_point_property=one_point_property,
        perturb_singularities=perturb_singularities,
        positive_magnitude=positive_magnitude,
        input_distances=False,
    )
    # compute_magnitude_from_distances(D, ts=ts, method=method, get_weights=get_weights)
    return magnitude, ts


def compute_t_conv(
    D,
    target_value,
    method="cholesky",
    positive_magnitude=False,
    input_distances=True,
):
    """
    Compute the scale at which the magnitude function has reached a certain target value
    using numeric root-finding.
    The target value is typically set to a high proportion of the cardinality.
    This pocedure assumes the magnitude function is typically non-decreasing.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    target_value : float
        The value of margnitude that should be reached.
        This value needs to be larger than 1 and smaller than the cardinality of the space.
    method : str
        The method used to compute the magnitude function.

    Returns
    -------
    t_conv : float
        The scaling parameter at which the magnitude function reaches the target value.

    References
    ----------
    .. [1] Limbeck, K., Andreeva, R., Sarkar, R. and Rieck, B., 2024.
        Metric Space Magnitude for Evaluating the Diversity of Latent Representations.
        arXiv preprint arXiv:2311.16054.
    """
    if D.shape[0] == 1:
        raise Exception(
            "We cannot find the convergence scale for a one point space!"
        )

    def comp_mag(X, ts):
        return compute_magnitude_from_distances(
            X,
            ts,
            method=method,
            one_point_property=True,
            perturb_singularities=True,
            positive_magnitude=positive_magnitude,
            input_distances=False,
        )

    if target_value is None:
        target_value = 0.95 * D.shape[0]
    else:
        if target_value >= D.shape[0]:
            raise Exception(
                "The target value needs to be smaller than the cardinality!"
            )
        if 0 >= target_value:
            raise Exception("The target value needs to be larger than 0!")
        # TODO also check for duplicates

    if input_distances:
        Z = similarity_matrix(D)
    else:
        Z = D

    t_conv = guess_convergence_scale(
        D=Z, comp_mag=comp_mag, target_value=target_value, guess=10
    )
    return t_conv
