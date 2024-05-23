from sklearn.manifold import Isomap
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import numpy as np

def distances_isomap(X, n_neighbors=12, p=2):
    """
    Compute geodesic distances as used by Isomap.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    n_neighbors : int
        The number of nearest neighbours used to compute geodesic distances.
    p: float
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
  
    Returns
    -------
    D : array-like, shape (`n_obs`, `n_obs`)
        A matrix of geodesic distances as computed by sklearn.manifold.Isomap.

    References
    ----------
    .. [1] Tenenbaum, J.B., Silva, V.D. and Langford, J.C., 2000. 
        A global geometric framework for nonlinear dimensionality reduction. Science, 290 (5500), pp.2319-2323.
    .. [2] Pedregosa et al., 2011. Scikit-learn: Machine Learning in Python. JMLR 12, pp.2825-2830.
    """
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2, p=p)
    isom = isomap.fit(X)
    return isom.dist_matrix_

def distances_scipy(X, X2, metric="cosine", p=2):
    """
    Compute the distance matrix using scipy.spatial.distance.cdist.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    metric: str
        The distance metric to use. The distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule'.
    p: float
        Parameter for the Minkowski metric.
  
    Returns
    -------
    D : ndarray, shape (`n_obs`, `n_obs`)
        A matrix of distances as computed by scipy.spatial.distance.cdist.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    """
    if metric == 'minkowski':
        dist = cdist(X, X2, metric=metric, p=p)
    else:
        dist = cdist(X, X2, metric=metric)
    return dist

def distances_lp(X, X2, p=2):
    """
    Compute the Lp distance matrix using scipy.spatial.distance_matrix.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    p: float
        Parameter for the Minkowski metric.
  
    Returns
    -------
    D : ndarray, shape (`n_obs`, `n_obs`)
        A matrix of Lp distances.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html#scipy.spatial.distance_matrix
    """
    dist = distance_matrix(X, X2, p=p)
    return dist

def normalise_distances_by_diameter(D):
    """
    Normalise all distances by the diameter of the space i.e. divide by the largest distance.

    Parameters
    ----------
    D : array_like, shape (`n_obs`, `n_obs`)
        A matrix of distances.
  
    Returns
    -------
    D_norm : ndarray, shape (`n_obs`, `n_obs`)
        A matrix of normalised distances.
    """
    diameter = np.max(D)
    #print("Diameter " + str(round(diameter, 2)))
    return D/diameter

def remove_duplicates(X):
    """
    Remove duplicate observations from a dataset.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
  
    Returns
    -------
    X_unique : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows/observations are unique.
    """
    X_unique, indices = np.unique(X, axis=0, return_index=True)
    n_new = X_unique.shape[0]
    n = X.shape[0]
    if n_new != n:
        print("Out of the "+ str(round(n)) + " observations in X, only "+ str(round(n_new)) + " are unique.")
    #n=X_unique.shape[0]
    #print(str(round(n)) + " distinct points in X")
    return X_unique#, indices, n

def get_dist(X, X2=None, metric="Lp", p=2, normalise_by_diameter=False, check_for_duplicates=True, n_neighbors=12):
    """
    Compute the distance matrix.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
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
        If True normalise all distances by the diameter.
    check_for_duplicates: bool
        If True remove all duplicate observations and compute distances only between unique points.
    n_neighbors : int
        The number of nearest neighbours used to compute geodesic distances. Only used if the metric is "isomap".

    Returns
    -------
    D : ndarray, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    """
    if check_for_duplicates:
        X = remove_duplicates(X)

    if X2 is None:
        X2 = X
    else:
        if check_for_duplicates:
            X2 = remove_duplicates(X2)
        #X2 = X
    #isinstance(X2, np.ndarray):
    #    X2 = remove_duplicates(X2)
    #if metric == "cosine":
    #    dist = distances_cosine(X)
    if metric == "Lp":
        dist = distances_lp(X, X2, p=p)
    elif metric == "isomap":
        dist = distances_isomap(X, n_neighbors=n_neighbors, p=p)
    else:
        dist = distances_scipy(X,X2, metric=metric, p=p)

    if normalise_by_diameter:
        dist = normalise_distances_by_diameter(dist)
    return dist