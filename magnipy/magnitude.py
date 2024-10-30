import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import solve_triangular, solve
from scipy.sparse.linalg import cg
#from krypy.linsys import LinearSystem , Cg
from scipy.optimize import toms748
from magnipy.distances import get_dist
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
    w = solve(Z, np.ones(Z.shape[0]), assume_a = "pos")
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
    w = solve(Z, np.ones(Z.shape[0]), assume_a = "sym")
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

#def weights_from_similarities_krylov(Z, ts, positive_definite = True):
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
    n=Z.shape[0]
    weights = np.zeros(shape=(n, len(ts)))
    w = np.ones(n)/n
    for i in range(len(ts)):
        # associated similarity matrix
        #Z = np.exp(-ts[i]*D)
        w, _ = cg(Z**(ts[i]), np.ones(n), w)
        weights[:,i]=w.squeeze()
    return weights

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
    """
    return 1/np.sum(Z, axis=0)

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
    .. [1] 
    """
    n=Z.shape[0]
    weights = np.ones(shape=(n, len(ts)))/n
    
    for i, t in enumerate(ts):
        #Z = np.exp(-t * D)
        weights[:,i] = (weights_spread(Z**t))
    return weights

def similarity_matrix(D):
    #n = D.shape[0]
    Z = np.zeros(D.shape)
    ne.evaluate("exp(-D)", out=Z)
    return Z

def magnitude_weights(Z, ts, mag_fn, one_point_property=True, perturb_singularities=True):
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
    """
    n=Z.shape[0]
    weights = np.ones(shape=(n, len(ts)))/n
    
    for i, t in enumerate(ts):
        # see above loop
        if t==0:
            if one_point_property:
                weights[:,i] = np.ones(n)/n
            else:
                weights[:, i] = np.full((n,n), np.nan)
                #raise Exception("We cannot compute magnitude at t=0 unless we assume the one point property!")
        else:
            # if checksingularity():
            #     print(warning)
            try:
                weights[:,i] = (mag_fn(Z**t))
            except Exception as e:
                if perturb_singularities:
                    print(f'Exception: {e} for t: {t} perturbing matrix')
                    Z_new = Z**t + 0.01 * np.identity(n=n)  # perturb similarity mtx to invert
                    weights[:,i] = (mag_fn(Z_new))
                else:
                    raise Exception(f'Exception: {e} for t: {t}')
    return weights # np.array(

def magnitude_from_distances(D, ts=np.arange(0.01, 5, 0.01), method="cholesky", get_weights=False,
                              one_point_property=True, perturb_singularities=True, positive_magnitude=False,
                              input_distances=True):
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
    if D.shape[0]==1:
        weights = np.ones(shape=(1,len(ts)))
    
    if input_distances:
        Z = similarity_matrix(D)
    else:
        Z = D

    if method=="spread":
        weights = spread_weights(Z, ts)
    #elif method=="krylov":
    #    weights = weights_from_similarities_krylov(Z, ts)
    elif method =="cg":
        weights = weights_from_similarities_cg(Z, ts)
    else:
        if method=="scipy":
            mag_fn = weights_scipy
        elif method == "scipy_sym":
            mag_fn = weights_scipy_sym
        elif method=="cholesky":
            mag_fn = weights_cholesky
        elif method =="conjugate_gradient_iteration":
            mag_fn = weights_cg
        elif method =="pinv":
            mag_fn = weights_pinv
        else:
            mag_fn = weights_naive
        weights = magnitude_weights(Z, ts, mag_fn, one_point_property=one_point_property, perturb_singularities=perturb_singularities)
    
    if positive_magnitude:
        weights = positive_weights_only(weights)

    if get_weights:
        return weights
    else:
        return magnitude_from_weights(weights)

def compute_magnitude_until_convergence(D, ts=None, target_value=None, n_ts=10, 
                                        log_scale = False, method="cholesky", get_weights=False, 
                                        one_point_property=True, perturb_singularities=True, 
                                                        positive_magnitude=False, input_distances=True):
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
        t_conv = compute_t_conv(Z, target_value=target_value, method=method, input_distances=False, positive_magnitude=positive_magnitude)
        ts = get_scales(t_conv, n_ts, log_scale = log_scale, one_point_property=one_point_property)
        #print(f"Evaluate magnitude at {self._n_ts} scales between 0 and the approximate convergence scale {self._t_conv}")
    return magnitude_from_distances(Z, ts, method=method, get_weights=get_weights, one_point_property=one_point_property,
                                     perturb_singularities=perturb_singularities, 
                                                        positive_magnitude=positive_magnitude, input_distances=False), ts

def compute_magnitude(X, ts=None, target_value=None, n_ts=10, log_scale = False, method="cholesky", 
                        get_weights=False, metric="Lp", p=2, normalise_by_diameter=False, 
                        n_neighbors=12, one_point_property=True, perturb_singularities=True, positive_magnitude=False):
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
    D = get_dist(X, p=p, metric=metric, normalise_by_diameter=normalise_by_diameter, n_neighbors=n_neighbors)
    Z = similarity_matrix(D)
    magnitude, ts = compute_magnitude_until_convergence(Z, ts=ts, n_ts=n_ts, method=method, target_value=target_value,
                                                        log_scale = log_scale, get_weights=get_weights, 
                                                        one_point_property=one_point_property, perturb_singularities=perturb_singularities, 
                                                        positive_magnitude=positive_magnitude, input_distances=False)
    #compute_magnitude_from_distances(D, ts=ts, method=method, get_weights=get_weights)
    return magnitude, ts


def mag_convergence(x0, x1, f=None, max_iterations=100):
    """
    Compute the scale at which a function approximately equals zero.
    
    Parameters
    ----------
    x0 : float
        A lower guess for the evaluation parameter.
    x1 : float
        A upper guess for the evaluation parameter.
    f : function
        A function whose root should be found.
    max_iterations : int
        The maximum number of iterations.
  
    Returns
    -------
    t_conv : float
        The value at which the function reaches zero.
    """
    return toms748(f, x0, x1, maxiter=max_iterations, rtol=1e-05)

def guess_convergence_scale(D, comp_mag, target_value, guess=10):
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
    comp_mag : function
        A function that computes the magnitude given a distance matrix and a vector of scales.
    guess :
        An initial guess for the scaling parameter.
  
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
    def f(x, W=D):
        mag = comp_mag(W, ts=[x])
        return mag[0] - target_value
    
    ### n/t =< Mag(t) =< t^n |A|
    ### 1 =< Mag(t) * t/n
    ### n/Mag(t) =< t #Meckes for Euclidean space
    lower_guess = 0
    f_guess = f(guess)
    while (f_guess<0):
        lower_guess = guess
        guess = guess*10
        f_guess = f(guess)
    t_conv = mag_convergence(lower_guess, guess, f, max_iterations=100)
    return t_conv

def compute_t_conv(D, target_value, method="cholesky", positive_magnitude=False, input_distances=True):
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
        raise Exception("We cannot find the convergence scale for a one point space!")
    def comp_mag(X, ts):
        return magnitude_from_distances(X, ts, method=method, one_point_property=True, perturb_singularities=True, 
                                        positive_magnitude=positive_magnitude, input_distances=False)
    if target_value is None:
        target_value=0.95*D.shape[0]
    else:
        if target_value >= D.shape[0]:
            raise Exception("The target value needs to be smaller than the cardinality!")
        if 0 >= target_value:
            raise Exception("The target value needs to be larger than 0!")
        # TODO also check for duplicates
    
    if input_distances:
        Z = similarity_matrix(D)
    else:
        Z = D

    t_conv = guess_convergence_scale(D=Z, comp_mag=comp_mag, target_value=target_value, guess=10)
    return t_conv

def get_scales(t_conv, n_ts=10, log_scale = False, one_point_property=True):
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
            ts_log = np.geomspace(t_conv/n_ts, t_conv, n_ts-1) #np.log(t_conv)
            ts=[0] + [i for i in ts_log]
            ts=np.array(ts)
        else:
            ts = np.linspace(0, t_conv, n_ts)
    else:
        if log_scale:
            ts = np.geomspace(t_conv/n_ts, t_conv, n_ts)
        else:
            ts = np.linspace(t_conv/n_ts, t_conv, n_ts)
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
        q=1/n
    return np.log(n - 1) / np.quantile(D[np.nonzero(D)], q=q)