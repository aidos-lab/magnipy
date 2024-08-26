import numpy as np

def one_spread(Z):
    # q = 1
    return np.shape(Z)[0] * np.prod((1/np.sum(Z, axis=0))**(1/np.shape(Z)[0]))

def q_spread(Z, q):
    # q not inf or 1
    vec = (1 / np.sum(Z, axis=0))**(1-q)
    vec2 = 1/(np.shape(Z)[0]**q) * np.sum(vec)
    return vec2**(1/(1-q))

def inf_spread(Z):
    # q = inf
    return np.min(np.shape(Z)[0]/np.sum(Z, axis=0))

# 0 the number of species
# 1 the exponential Shannon index
# 2 the Simpson index
# inf the reciprocal Berger-Parker diversity
# q the q-diversity