import numpy as np
import scipy.stats as st


def sample_levy(alpha, seed=0, n_steps=500, dim=3):
    """
    Generate a Levy process using the Levy stable distribution.

    Parameters
    ----------
    alpha : float
        Stability parameter.
    seed : int
        Random seed.
    n_steps : int
        Number of steps.
    dim : int
        Dimension of the process.

    Returns
    -------
    levy_process : ndarray, shape (`n_steps`, `dim`)
        A Levy process.
    """
    beta = 0  # Skewness parameter
    np.random.seed(seed)
    levy_process = np.zeros((n_steps, dim))
    # Generate the Levy process
    for d in range(dim):
        levy_process[:, d] = np.cumsum(
            st.levy_stable.rvs(alpha, beta, size=n_steps) * np.sqrt(1 / n_steps)
        )
    return levy_process


def sample_sphere(n, d):
    """
    Sample n points on the d-dimensional sphere.
    """
    points = np.random.randn(n, d)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return points


def hawkes_process(lam, alpha):
    sigma = 0.02
    N = np.random.poisson(lam)
    X = np.zeros((10000, 2))
    # generate N centers for the clusters
    values = np.random.rand(N, 2)
    X[:N, :] = values
    total_so_far = N
    next = 0
    while next < total_so_far:
        next_X = X[next]  # select the next point
        N_children = np.random.poisson(alpha)
        new_X = np.tile(next_X, (N_children, 1)) + sigma * np.random.rand(
            N_children, 2
        )
        # update the next rows of X with the coordinates of the children
        X[total_so_far : total_so_far + N_children, :] = new_X
        total_so_far += N_children
        next += 1
    X_cut = X[:total_so_far]
    return X_cut


def sample_points_gaussian(mean, cov, n):
    points = np.random.multivariate_normal(mean, cov, n)
    points = np.clip(points, 0, 2)  # Clip values to [0, 1] range
    return points


def sample_points_gaussian_2(mean, cov, n):
    points = np.random.multivariate_normal(mean, cov, n)
    # points = np.clip(points, 0, 2)  # Clip values to [0, 1] range
    return points


def sample_points_square(n, range_max):
    points = np.random.rand(n, 2) * range_max
    return points
