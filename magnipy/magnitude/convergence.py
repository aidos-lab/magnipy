from scipy.optimize import toms748

# from magnipy.magnitude.weights import similarity_matrix
# from magnipy.magnitude.compute import magnitude_from_distances


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
    while f_guess < 0:
        lower_guess = guess
        guess = guess * 10
        f_guess = f(guess)
    # print(f"Lower guess: {lower_guess}, Upper guess: {guess}")
    # print(f_guess)
    t_conv = mag_convergence(lower_guess, guess, f, max_iterations=100)
    return t_conv
