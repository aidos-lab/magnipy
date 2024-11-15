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
