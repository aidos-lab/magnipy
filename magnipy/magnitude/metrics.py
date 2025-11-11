"Methods for computing distance metrics on graphs. Adjusted from https://github.com/aidos-lab/rings."

import warnings

from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import networkx as nx

#  ╭──────────────────────────────────────────────────────────╮
#  │ Graph Metrics                                            │
#  ╰──────────────────────────────────────────────────────────╯


def diffusion_distance(G, **kwargs):
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


def heat_kernel_distance(G, **kwargs):
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


def resistance_distance(G, weight=None, **kwargs):
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
    try:
        distances = nx.resistance_distance(G, weight=weight)
        distances = nx.utils.dict_to_numpy_array(distances)
    except nx.NetworkXError:
        distances = [np.nan]

    return distances


def shortest_path_distance(G, weight=None, **kwargs):
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