import numpy as np
import scipy.stats as st

def sample_levy(alpha, seed=0, n_steps = 500, dim=3):
  # Set the parameters of the Levy process
  # alpha Stability parameter
  beta = 0  # Skewness parameter
  np.random.seed(seed)
  levy_process = np.zeros((n_steps, dim))
  # Generate the Levy process
  for d in range(dim):
      levy_process[:, d] = np.cumsum(st.levy_stable.rvs(alpha, beta, size=n_steps)*np.sqrt(1/n_steps))
  return levy_process

def sample_sphere(n, d):
  points = np.random.randn(n, d)
  points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
  return points