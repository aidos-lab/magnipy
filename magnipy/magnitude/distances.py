"Methods for computing distances from data."

from sklearn.manifold import Isomap
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import numpy as np
from scipy.sparse.csgraph import shortest_path
import torch
import warnings
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx

scipy_distance_metrics = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

#  ╭──────────────────────────────────────────────────────────╮
#  │ Feature distance computations                            │
#  ╰──────────────────────────────────────────────────────────╯


def distances_isomap(X, **kwargs):
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
    p = kwargs.get("p", 2)
    n_neighbors = kwargs.get("n_neighbors", 12)
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2, p=p)
    isom = isomap.fit(X)
    return isom.dist_matrix_


def distances_scipy(X, X2, metric="cosine", **kwargs):
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
    p = kwargs.get("p", 2)
    if metric == "minkowski":
        dist = cdist(X, X2, metric=metric, p=p)
    else:
        dist = cdist(X, X2, metric=metric)
    return dist


def distances_torch_cdist(X, X2, **kwargs):
    p = kwargs.get("p", 2)
    D = torch.cdist(X, X2, p=p)
    return D


def distances_lp(X, X2, **kwargs):
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
    p = kwargs.get("p", 2)
    dist = distance_matrix(X, X2, p=p)
    return dist


#  ╭──────────────────────────────────────────────────────────╮
#  │ Attributed graph metrics                                 │
#  ╰──────────────────────────────────────────────────────────╯


def to_nx_graph(x, adj):
    G = nx.from_numpy_array(adj)

    for i, feature in enumerate(x):
        G.nodes[i]["feature"] = feature
    return G


def distances_geodesic(
    X=None, X2=None, Adj=None, G=None, metric="euclidean", **kwargs
):
    """
    Compute a weighted / geodesic distance matrix from a graph.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    X2 : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    p: float
        Parameter for the Minkowski metric.
    Adj : array_like, shape (`n_obs`, `n_obs`)
        An adjacency matrix.

    Returns
    -------
    D : ndarray, shape (`n_obs`, `n_obs`)
        A matrix of distances as computed by scipy.spatial.distance.cdist.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    """
    ## todo add check if X, X2, Adj have matching dimensions
    p = kwargs.get("p", 2)

    if G is None and Adj is None:
        raise Exception("Either a graph G or adjacencies A must be provided.")

    if G is not None and Adj is not None:
        raise Exception("Provide either a graph G or adjacencies A, not both.")

    if G is not None:
        if X is None:
            if G.nodes[G.nodes[0]].get("feature") is not None:
                X = np.array([G.nodes[i]["feature"] for i in G.nodes])
        Adj = nx.adjacency_matrix(G).todense()

    if X is None:
        weighted_adjacency = Adj
    else:
        feature_distances = distances_scipy(X, X2, metric=metric, p=p)

        # Step 2: Combine feature distances with adjacency matrix
        # For example, you can multiply adjacency matrix by feature distances to create a weighted graph
        weighted_adjacency = Adj * feature_distances

    # Step 3: Compute geodesic distances using Dijkstra's algorithm on the weighted adjacency matrix
    geodesic_distances = shortest_path(weighted_adjacency, directed=False)
    return geodesic_distances


#  ╭──────────────────────────────────────────────────────────╮
#  │ Graph Structure Metrics                                  │
#  ╰──────────────────────────────────────────────────────────╯


def diffusion_distance(G=None, A=None, **kwargs):
    """Calculate diffusion distance between vertices of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    num_steps : int
        Number of steps for the diffusion operator that is used for
        the distance calculations.

    norm: bool (default True)
        Normalize the Laplacian.

    symmetric : bool (default True)
        Normalize the Laplacian _symmetrically_.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    t = kwargs.get("num_steps", 1)
    n_jobs = kwargs.get("n_jobs", None)
    symmetric = kwargs.get("symmetric", True)
    norm = kwargs.get("norm", True)

    if not norm and not symmetric:
        warnings.warn(
            "Assuming default Laplacian, which is symmetric.",
            UserWarning,
        )
        symmetric = True

    if G is None and A is None:
        raise Exception("Either a graph G or adjacencies A must be provided.")

    if G is not None and A is not None:
        raise Exception("Provide either a graph G or adjacencies A, not both.")

    if A is None:
        A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1))

    # Bail out early on if there are isolated nodes.
    if (np.diagonal(D) == 0).any():
        return np.nan

    # NEW VERSION
    L = _compute_laplacian(A, D, norm=norm, symmetric=symmetric)
    psi = _compute_psi(L, t, symmetric=symmetric)

    if np.iscomplexobj(psi):
        warnings.warn(
            "Input data contains complex numbers. The imaginary part will be discarded.",
            UserWarning,
        )
        psi = np.real(psi)
    return pairwise_distances(psi, metric="euclidean", n_jobs=n_jobs)


def heat_kernel_distance(G=None, A=None, **kwargs):
    """Calculate heat kernel distance between vertices of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    t : int or `None`
        Number of steps for the diffusion operator that is used for
        the potential distance. If set to `None`, a suitable $t$ is
        selected based on the von Neumann entropy.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    t = kwargs.get("num_steps", 1)
    n_jobs = kwargs.get("n_jobs", None)
    symmetric = kwargs.get("symmetric", True)
    norm = kwargs.get("norm", True)

    if G is None and A is None:
        raise Exception("Either a graph G or adjacencies A must be provided.")

    if G is not None and A is not None:
        raise Exception("Provide either a graph G or adjacencies A, not both.")

    if A is None:
        A = nx.adjacency_matrix(G).todense()
    D = np.diag(A.sum(axis=1))

    L = _compute_laplacian(A, D, norm=norm, symmetric=symmetric)

    eigenvalues, eigenvectors = _compute_spectrum(L, symmetric=symmetric)
    scales = np.exp(-t * eigenvalues)
    X = scales * eigenvectors

    if np.iscomplexobj(X):
        warnings.warn(
            "Input data contains complex numbers. The imaginary part will be discarded.",
            UserWarning,
        )
        X = np.real(X)

    return pairwise_distances(X, metric="euclidean", n_jobs=n_jobs)


def resistance_distance(G=None, A=None, weight=None, **kwargs):
    """
    Calculate resistance distance between vertices of a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    weight : str or None
        The edge attribute that holds the numerical value used as a
        weight. If set to `None`, the graph is treated as unweighted.

    kwargs
        Additional keyword arguments. Only required for compatibility
        reasons.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    if G is None and A is None:
        raise Exception("Either a graph G or adjacencies A must be provided.")

    if G is not None and A is not None:
        raise Exception("Provide either a graph G or adjacencies A, not both.")

    if A is not None:
        G = nx.from_numpy_array(A)

    try:
        distances = nx.resistance_distance(G, weight=weight)
        distances = nx.utils.dict_to_numpy_array(distances)
    except nx.NetworkXError:
        distances = [np.nan]

    return distances


def shortest_path_distance(G=None, A=None, weight=None, **kwargs):
    """Calculate shortest-path distance between vertices.

    Calculate shortest-path distance between vertices of a graph using
    the Floyd--Warshall algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph. All attributes of the graph will be ignored in
        the subsequent calculations.

    weight : str or None
        The edge attribute that holds the numerical value used as a
        weight. If set to `None`, the graph is treated as unweighted.

    kwargs
        Additional keyword arguments. Only required for compatibility
        reasons.

    Returns
    -------
    np.array
        Matrix of distance values
    """
    if G is None and A is None:
        raise Exception("Either a graph G or adjacencies A must be provided.")

    if G is not None and A is not None:
        raise Exception("Provide either a graph G or adjacencies A, not both.")

    if A is not None:
        G = nx.from_numpy_array(A)

    return nx.floyd_warshall_numpy(G, weight=weight)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Spectral Helper Functions                                │
#  ╰──────────────────────────────────────────────────────────╯


def _compute_laplacian(A, D, norm=True, symmetric=True):
    """
    Compute the Laplacian matrix of a graph.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the graph.
    D : numpy.ndarray
        Degree matrix of the graph.
    norm : bool, optional
        If True, compute the normalized Laplacian. Default is True.
    symmetric : bool, optional
        If True, compute the symmetric normalized Laplacian. Default is True.

    Returns
    -------
    L : numpy.ndarray
        The Laplacian matrix of the graph.

    Notes
    -----
    The Laplacian matrix is defined as L = D - A, where D is the degree matrix
    and A is the adjacency matrix.

    If `norm` is True and `symmetric` is True, the symmetric normalized Laplacian is computed as L = I - D^(-1/2) * A * D^(-1/2).

    If `norm` is True and `symmetric` is False, the random walk normalized Laplacian is computed as L = I - D^(-1) * A.
    """
    if norm and symmetric:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))
        L = np.eye(*A.shape) - D_inv_sqrt @ A @ D_inv_sqrt
    elif norm and not symmetric:
        L = np.eye(*A.shape) - np.diag(1.0 / D.diagonal()) @ A
    else:
        L = D - A
    return L


def _compute_psi(L, t, symmetric=True):
    """
    Compute Psi, a matrix made up of powered eigenvalues^t * eigenvectors of the Laplacian matrix.

    Parameters
    ----------
    L : array_like
        The Laplacian matrix.
    t : float
        The exponent to which the eigenvalues are raised.
    symmetric : bool, optional
        If True, assumes the Laplacian matrix is symmetric and uses `np.linalg.eigh`.
        If False, uses `np.linalg.eig`. Default is True.

    Returns
    -------
    psi : ndarray
        The Psi matrix computed from the eigenvalues and eigenvectors of the Laplacian matrix.

    Notes
    -----
    If `symmetric` is True, the function uses `np.linalg.eigh` which is more efficient for symmetric matrices.
    Otherwise, it uses `np.linalg.eig`.
    """
    eigenvalues, eigenvectors = _compute_spectrum(L, symmetric=symmetric)
    eigenvalues = np.power(eigenvalues, t)
    psi = eigenvalues * eigenvectors
    return psi


def _compute_spectrum(L, symmetric=True):
    """
    Compute the eigenvalues and eigenvectors of the Laplacian matrix.

    Parameters
    ----------
    L : array_like
        The Laplacian matrix.
    symmetric : bool, optional
        If True, assumes the Laplacian matrix is symmetric and uses `np.linalg.eigh`.
        If False, uses `np.linalg.eig`. Default is True.

    Returns
    -------
    eigenvalues : ndarray
        The eigenvalues of the Laplacian matrix.
    eigenvectors : ndarray
        The eigenvectors of the Laplacian matrix.

    Notes
    -----
    If `symmetric` is True, the function uses `np.linalg.eigh` which is more efficient for symmetric matrices.
    Otherwise, it uses `np.linalg.eig`.
    """
    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(L)
    return eigenvalues, eigenvectors


#  ╭──────────────────────────────────────────────────────────╮
#  │ Modify the distances                                     │
#  ╰──────────────────────────────────────────────────────────╯


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
    return D / diameter


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
        print(
            "Out of the "
            + str(round(n))
            + " observations in X, only "
            + str(round(n_new))
            + " are unique."
        )
    return X_unique


def compute_subgraphs_with_dist(G, dist_fn, subgraphs=None):
    """
    Compute the magnitude of a graph using a specified distance function.
    The magnitude is computed across a fixed choice of scales.
    This function computes the magnitude of each connected component
    of the graph separately and sums them up to obtain the total magnitude.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    ts : array-like, shape (`n_ts`, )
        A vector of scaling parameters at which to evaluate magnitude.
    dist_fn : function
        A function that takes a graph as input and returns a distance matrix.
    subgraphs : list of networkx.Graph, optional
        A list of subgraphs. If provided, the magnitude will be computed
        on each subgraph and summed up. If None, the connected components
        of the graph will be used as subgraphs.
    method : str
        The method used to compute magnitude. If 'cholesky' is chosen, the Cholesky decomposition
        will be used to compute magnitude. If 'spread' is chosen, the spread of a metric space will be computed.
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
    subgraphs : list of networkx.Graph
        The subgraphs on which the magnitude has been computed.
    Ds : list of np.array
        The distance matrices of the subgraphs.
    mags : list of array_like, shape (`n_ts`, ) or shape (`n_obs`, `n_ts`)
        The magnitudes of the subgraphs.
    """
    if subgraphs is None:
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    Ds = []

    for s in subgraphs:
        D = dist_fn(G=s)
        Ds.append(D)

    return subgraphs, Ds


#  ╭──────────────────────────────────────────────────────────╮
#  │ Choosing the right distance                              │
#  ╰──────────────────────────────────────────────────────────╯


def get_dist(
    X,
    X2=None,
    Adj=None,
    G=None,
    metric="euclidean",
    normalise_by_diameter=False,
    check_for_duplicates=True,
    mode="attributes",
    **kwargs,
):
    """
    Compute the distance matrix.

    Parameters
    ----------
    X : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    X2 : array_like, shape (`n_obs`, `n_vars`)
        A dataset whose rows are observations and columns are features.
    Adj : array_like, shape (`n_obs`, `n_obs`)
        An adjacency matrix.
    metric: str
        The distance metric to use. The distance function can be
        'Lp', 'isomap', "torch_cdist",
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule',
        "shortest_path_distance", "resistance_distance", "diffusion_distance", "heat_kernel_distance".
    p: float
        Parameter for the Minkowski metric.
    normalise_by_diameter: bool
        If True normalise all distances by the diameter.
    check_for_duplicates: bool
        If True remove all duplicate observations and compute distances only between unique points.
    mode: str
        The mode of distance computation. Can be either 'attributes', 'structure', or 'full'.

    Returns
    -------
    D : ndarray, shape (`n_obs`, `n_obs`)
        A matrix of distances.
    """

    if mode in ["attributes", "full"]:
        if (X is None) and (G is not None):
            if G.nodes[G.nodes[0]].get("feature") is not None:
                X = np.array([G.nodes[i]["feature"] for i in G.nodes])
            else:
                raise Exception(
                    "No attribute data provided to compute distances."
                )

    if check_for_duplicates and (X is not None):
        X = remove_duplicates(X)

    if X2 is None:
        X2 = X
    else:
        if check_for_duplicates and (X is not None):
            X2 = remove_duplicates(X2)

    if mode == "attributes":
        if (X is None) and (G is not None):
            if G.nodes[G.nodes[0]].get("feature") is not None:
                X = np.array([G.nodes[i]["feature"] for i in G.nodes])
                X2 = X

        if X is None:
            raise Exception("No data provided to compute distances.")

        if metric == "Lp":
            dist = distances_lp(X, X2, **kwargs)
        elif metric == "isomap":
            dist = distances_isomap(X, **kwargs)
        elif metric == "torch_cdist":
            dist = distances_torch_cdist(X, X2, **kwargs)
        elif metric in scipy_distance_metrics:
            dist = distances_scipy(X, X2, metric=metric, **kwargs)
        else:
            raise Exception(
                f"Metric {metric} not yet implemented for attributes mode."
            )
    else:
        if (Adj is not None) or (G is not None):
            if mode == "structure":
                if G is not None:
                    # G = to_nx_graph(X, Adj)
                    if G.number_of_edges() == 0:
                        n = G.number_of_nodes()
                        return np.zeros((n, n))
                    # if Adj is not None:

                if metric == "shortest_path_distance":
                    dist = shortest_path_distance(G=G, A=Adj, **kwargs)
                elif metric == "resistance_distance":
                    dist = resistance_distance(G=G, A=Adj, **kwargs)
                elif metric == "diffusion_distance":
                    dist = diffusion_distance(G=G, A=Adj, **kwargs)
                elif metric == "heat_kernel_distance":
                    dist = heat_kernel_distance(G=G, A=Adj, **kwargs)
                else:
                    raise Exception(
                        f"Metric {metric} not yet implemented for structure mode."
                    )
            elif mode == "full":
                if metric in scipy_distance_metrics:
                    # dist = distances_geodesic(X=X, X2=X2, Adj=Adj, G=G, metric=metric, **kwargs)
                    dist = distances_geodesic(
                        X, X2, Adj=Adj, G=G, metric=metric, **kwargs
                    )
                else:
                    raise Exception(
                        f"Metric {metric} not yet implemented for full mode."
                    )
        else:
            raise Exception("No graph provided to compute graph distances.")

    if normalise_by_diameter:
        dist = normalise_distances_by_diameter(dist)
    return dist
