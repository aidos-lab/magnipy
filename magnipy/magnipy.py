from magnipy.magnitude.scales import (
    get_scales,
    scale_when_scattered,
    scale_when_almost_scattered,
    cut_ts,
)
from magnipy.magnitude.convergence import guess_convergence_scale
from magnipy.magnitude.weights import (
    magnitude_from_weights,
    similarity_matrix,
)
from magnipy.magnitude.compute import (
    compute_magnitude_until_convergence,
    compute_t_conv,
)
from magnipy.magnitude.dimension import (
    magitude_dimension_profile,
    magnitude_dimension,
    magnitude_dimension_profile_exact,
)
from magnipy.magnitude.distances import get_dist
from magnipy.magnitude.function_operations import (
    diff_of_functions,
    sum_of_functions,
    cut_until_scale,
    mag_area,
    mag_diff,
)
from magnipy.utils.plots import (
    plot_magnitude_function,
    plot_magnitude_dimension_profile,
)
import numpy as np
import copy


class Magnipy:
    def __init__(
        self,
        # Input data parameters
        X,
        # Parameters for the evaluation scales
        ts=None,
        n_ts=30,
        log_scale=False,
        return_log_scale=False,
        scale_finding="convergence",
        target_prop=0.95,
        # Parameters for the distance matrix
        metric="euclidean",
        p=2,
        Adj=None,
        n_neighbors=12,
        # Parameters for the computation of magnitude
        method="cholesky",
        one_point_property=True,
        perturb_singularities=True,
        positive_magnitude=False,
        # Other parameters
        recompute=False,
        name="",
    ):
        """
        Initialises a Magnipy object.

        Parameters
        ----------
        Input data parameters:
        X : array_like, shape (`n_obs`, `n_vars`)
            A dataset whose rows are observations and columns are features.

        Parameters for the evaluation scales:
        ts : array_like, shape (`n_ts`, )
            The scales at which to evaluate the magnitude functions. If None, the scales are computed automatically.
        n_ts : int
            The number of scales at which to evaluate the magnitude functions.
        log_scale : bool
            Whether to use a log-scale for the evaluation scales.
        return_log_scale : bool
            Whether to return the scales on log-scale when computing the magnitude dimension profile.
        scale_finding : str
            The method to use to find the scale at which to evaluate the magnitude functions. Either 'scattered' or 'convergence'.
        target_prop : float
            The proportion of points that are scattered OR the proportion of cardinality that the magnitude functon converges to.

        Parameters for the distance matrix:
        metric : str
            The distance metric to use. The distance function can be
            'Lp', 'isomap',
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule'.
        p : float
            Parameter for the Minkowski metric.
        Adj : array_like, shape (`n_obs`, `n_obs`)
            An adjacency matrix used to compute geodesic distances. If None, all points are adjacent.
        n_neighbors : int
            The number of nearest neighbours used to compute geodesic distances. Only used if the metric is "isomap".

        Parameters for the computation of magnitude:
        method : str
            The method to use to compute the magnitude functions.
            One of 'cholesky', 'scipy', 'scipy_sym', 'inv', 'pinv', 'conjugate_gradient_iteration', 'cg'.
        one_point_property : bool
            Whether to enforce the one-point property.
        perturb_singularities : bool
            Whether to perturb the simularity matrix whenever singularities in the magnitude function occure.
        positive_magnitude : bool
            Whether to compute positive magnitude, by taking only the sum of the positive weights.
        recompute : bool
            Whether to recompute the magnitude functions if they have already been computed.
        name : str
            The name of the Magnipy object.

        Returns
        -------
        Magnipy
            A Magnipy object.
        """

        self._Adj = Adj
        if metric != "precomputed":
            self._X = X

            def compute_distances(X, X2, Adj=None):
                return get_dist(
                    X,
                    X2=X2,
                    Adj=Adj,
                    p=p,
                    metric=metric,
                    normalise_by_diameter=False,
                    n_neighbors=n_neighbors,
                )

            self._get_dist = compute_distances
            self._D = self._get_dist(X, X2=None, Adj=self._Adj)
            self._n = self._D.shape[0]
            self._target_value = target_prop * self._D.shape[0]
            self._Z = similarity_matrix(self._D)
        else:
            self._X = None
            self._D = X
            self._n = self._D.shape[0]
            self._Z = similarity_matrix(self._D)
            self._target_value = target_prop * self._D.shape[0]

        self._proportion_scattered = target_prop
        if (scale_finding != "scattered") & (scale_finding != "convergence"):
            raise Exception(
                "The scale finding method must be either 'scattered' or 'convergence'."
            )

        if method not in [
            "cholesky",
            "scipy",
            "scipy_sym",
            "inv",
            "pinv",
            "conjugate_gradient_iteration",
            "cg",
            "spread",
            "naive",
        ]:
            raise Exception(
                "The computation method must be one of 'cholesky', 'scipy', 'scipy_sym', 'inv', 'pinv', 'conjugate_gradient_iteration', 'cg', 'naive', 'spread'."
            )

        def compute_mag(Z, ts, n_ts=n_ts, get_weights=False):
            return compute_magnitude_until_convergence(
                Z,
                ts=ts,
                n_ts=n_ts,
                method=method,
                log_scale=log_scale,
                get_weights=get_weights,
                one_point_property=one_point_property,
                perturb_singularities=perturb_singularities,
                positive_magnitude=positive_magnitude,
                input_distances=False,
            )

        self._compute_mag = compute_mag

        self._scale_finding = scale_finding
        self._ts = ts
        self._n_ts = n_ts
        self._log_scale = log_scale
        self._method = method
        self._metric = metric
        # self._p = p
        self._one_point_property = one_point_property
        self._perturb_singularities = perturb_singularities
        # self._n_neighbors = n_neighbors
        self._return_log_scale = return_log_scale
        self._recompute = recompute
        self._positive_magnitude = positive_magnitude
        self._name = name

        self._magnitude = None
        self._weights = None
        self._magnitude_dimension_profile = None
        self._ts_dim = None
        self._t_conv = None
        self._magnitude_dimension = None
        self._magnitude_area = None
        self._t_scattered = None
        self._t_almost_scattered = None

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Basic Informations                                       │
    #  ╰──────────────────────────────────────────────────────────╯

    def get_name(self):
        """
        Get the name of the Magnipy object.
        """
        return self._name

    def get_dist(self):
        """
        Compute the distance matrix.
        """
        if (self._D is None) | self._recompute:
            self._D = self._get_dist(self._X, X2=None, Adj=self._Adj)
        return self._D

    def get_similarity_matrix(self):
        """
        Compute the similarity matrix.
        """
        if (self._Z is None) | self._recompute:
            self._Z = similarity_matrix(self._D)
        return self._Z

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Find the Evaluation Scales                               │
    #  ╰──────────────────────────────────────────────────────────╯

    def get_t_conv(self):
        """
        Compute the scale at which the magnitude functions reach a certain value of magnitude.
        """
        if self._scale_finding == "convergence":
            if (self._t_conv is None) | self._recompute:

                def comp_mag(X, ts):
                    return self._compute_mag(X, ts)[0]

                self._t_conv = guess_convergence_scale(
                    D=self._Z,
                    comp_mag=comp_mag,
                    target_value=self._target_value,
                    guess=10,
                )
                # print(f"Convergence scale: {self.t_conv}")
                # self._t_conv = compute_t_conv(
                #    self._Z,
                #   target_value=self._target_value,
                #    method=self._method,
                #    positive_magnitude=self._positive_magnitude,
                #    input_distances=False,
                # )
            return self._t_conv
        elif self._scale_finding == "scattered":
            return self._scale_when_almost_scattered(q=None)

    def get_scales(self):
        """
        Compute the scales at which to evaluate the magnitude functions.
        """
        if (self._ts is None) | self._recompute:
            if self._scale_finding == "scattered":
                if self._proportion_scattered is None | self._recompute:
                    _ = self._scale_when_almost_scattered(
                        q=self._proportion_scattered
                    )
                self._ts = get_scales(
                    self._t_almost_scattered,
                    self._n_ts,
                    log_scale=self._log_scale,
                    one_point_property=self._one_point_property,
                )
            elif self._scale_finding == "convergence":
                if (self._t_conv is None) | self._recompute:
                    _ = self.get_t_conv()
                self._ts = get_scales(
                    self._t_conv,
                    self._n_ts,
                    log_scale=self._log_scale,
                    one_point_property=self._one_point_property,
                )
        return self._ts

    def change_scales(self, ts=None, t_cut=None):
        """
        Change the evaluation scales of the magnitude functions.

        Parameters
        ----------
        ts : array_like, shape (`n_ts_new`, )
            The new scales at which to evaluate the magnitude functions.
        t_cut : float
            The scale at which to cut the magnitude functions.
        """
        if ts is None:
            if t_cut is None:
                self._ts = None
                # raise Exception("A new evaluation interval or a cut-off scale need to be specified to change the evaluation scales!")
            else:
                self._ts = get_scales(
                    t_cut,
                    self._n_ts,
                    log_scale=self._log_scale,
                    one_point_property=self._one_point_property,
                )
        else:
            self._ts = ts
        self._magnitude = None
        self._magnitude_dimension_profile = None
        self._magnitude_dimension = None
        self._magnitude_area = None
        self._weights = None
        self._ts_dim = None

    def _scale_when_scattered(self):
        """
        Compute the scale at which the metric space is scattered.

        Returns
        -------
        t_scattered : float
            The scale at which the metric space is scattered.

        References
        ----------
        [1] Leinster T. The magnitude of metric spaces. Documenta Mathematica. 2013 Jan 1;18:857-905.
        """
        if (self._t_scattered is None) | self._recompute:
            self._t_scattered = scale_when_scattered(self._D)
        return self._t_scattered

    def _scale_when_almost_scattered(self, q=None):
        """
        Compute the scale at which the metric space is almost scattered.

        Parameters
        ----------
        q : float
            The proportion of points that are scattered.

        Returns
        -------
        t_almost_scattered : float
            The scale at which the metric space is almost scattered.
        """

        if (self._t_almost_scattered is None) | self._recompute:
            self._t_almost_scattered = scale_when_almost_scattered(
                self._D, n=self._n, q=q
            )
        return self._t_almost_scattered

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Compute Magnitude Weights and Functions                  │
    #  ╰──────────────────────────────────────────────────────────╯

    def get_magnitude_weights(self):
        """
        Compute the magnitude weights.

        Returns
        -------
        weights : array_like, shape (`n_obs`, `n_ts`)
            The weights of the magnitude function.
        """
        if (self._weights is None) | self._recompute:
            ts = self.get_scales()
            weights, ts = self._compute_mag(
                self._Z,
                ts=ts,
                # n_ts=self._n_ts,
                # method=self._method,
                # log_scale=self._log_scale,
                get_weights=True,
            )
            self._weights = weights
            self._ts = ts
            if self._ts is None:
                self._t_conv = ts[-1]
        return self._weights, self._ts

    def get_magnitude(self):
        """
        Compute the magnitude function.

        Returns
        -------
        magnitude : array_like, shape (`n_ts`, )
            The values of the magnitude function.
        ts : array_like, shape (`n_ts`, )
            The scales at which the magnitude function has been evaluated.
        """
        if (
            (self._magnitude is None) & (self._weights is None)
        ) | self._recompute:
            ts = self.get_scales()
            self._magnitude, ts = self._compute_mag(
                self._Z,
                ts=ts,
                # n_ts=self._n_ts,
                # method=self._method,
                # log_scale=self._log_scale,
                get_weights=False,
            )
            if self._ts is None:
                self._t_conv = ts[-1]
                self._ts = ts
        elif (self._magnitude is None) & (not (self._weights is None)):
            self._magnitude = magnitude_from_weights(self._weights)
        return self._magnitude, self._ts

    def _eval_at_scales(self, ts_new, get_weights=False):
        """
        Evaluate the magnitude functions at new scales.

        Parameters
        ----------
        ts_new : array_like, shape (`n_ts_new`, )
            The new scales at which to evaluate the magnitude functions.
        get_weights : bool
            Whether to compute the weights.

        Returns
        -------
        mag : array_like, shape (`n_ts_new`, )
            The values of the magnitude function evaluated at the new scales.
        ts : array_like, shape (`n_ts_new`, )
            The new scales at which the magnitude function has been evaluated.
        """
        mag, ts = self._compute_mag(
            self._Z,
            ts=ts_new,
            get_weights=get_weights,
        )
        return mag, ts

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Magnitude Dimension                                      │
    #  ╰──────────────────────────────────────────────────────────╯

    def get_magnitude_dimension_profile(self, exact=False, h=None):
        """
        Compute the magnitude dimension profile.

        Parameters
        ----------
        exact : bool
            Whether to compute the magnitude dimension profile exactly.
        h : float
            The stepsize to use for exact computations of the slope.
        """
        if (self._magnitude_dimension_profile is None) | self._recompute:
            if exact:
                (
                    self._magnitude_dimension_profile,
                    self._ts_dim,
                ) = magnitude_dimension_profile_exact(
                    self._D,
                    ts=self._ts,
                    h=h,
                    target_value=self._target_value,
                    n_ts=self._n_ts,
                    return_log_scale=self._return_log_scale,
                    one_point_property=self._one_point_property,
                    method=self._method,
                    log_scale=self._log_scale,
                )
            else:
                if self._magnitude is None:
                    _, _ = self.get_magnitude()
                (
                    self._magnitude_dimension_profile,
                    self._ts_dim,
                ) = magitude_dimension_profile(
                    mag=self._magnitude,
                    ts=self._ts,
                    return_log_scale=self._return_log_scale,
                    one_point_property=self._one_point_property,
                )
        return self._magnitude_dimension_profile, self._ts_dim

    def get_magnitude_dimension(self, exact=False):
        """
        Compute the magnitude dimension.

        Parameters
        ----------
        exact : bool
            Whether to compute the magnitude dimension exactly.

        Returns
        -------
        magnitude_dimension : float
            The magnitude dimension. We compute it as the maximum value of the
            magnitude dimension profile.
        """
        if self._magnitude_dimension_profile is None:
            _, _ = self.get_magnitude_dimension_profile(exact=exact)
        if (self._magnitude_dimension is None) | self._recompute:
            self._magnitude_dimension = magnitude_dimension(
                self._magnitude_dimension_profile
            )
        return self._magnitude_dimension

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Plots                                                    │
    #  ╰──────────────────────────────────────────────────────────╯

    def plot_magnitude_function(self):
        """
        Plot the magnitude function.
        """
        if (self._magnitude is None) | self._recompute:
            _, _ = self.get_magnitude()
        plot_magnitude_function(self._ts, self._magnitude, name=self._name)

    def plot_magnitude_dimension_profile(self):
        """
        Plot the magnitude dimension profile.
        """
        if (self._magnitude_dimension_profile is None) | self._recompute:
            _, _ = self.get_magnitude_dimension_profile()
        plot_magnitude_dimension_profile(
            ts=self._ts_dim,
            mag_dim=self._magnitude_dimension_profile,
            log_scale=self._return_log_scale,
            name=self._name,
        )

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Operations with Magnipy Objects                          │
    #  ╰──────────────────────────────────────────────────────────╯

    def include_points(self, X_new, Adj_new=None, update_ts=False):
        if self._X is None:
            self._X = X_new
            self._Adj = Adj_new
            self._D = self._get_dist(self._X, Adj=self._Adj)
            self._n = self._D.shape[0]
            self._Z = similarity_matrix(self._D)
        else:
            X = np.concatenate((self._X, X_new), axis=0)
            self._X = X
            self._Adj = Adj_new
            self._D = self._get_dist(
                self._X,
                Adj=self._Adj,
            )
            self._Z = similarity_matrix(self._D)
            self._n = self._D.shape[0]
        if update_ts:
            self._ts = None
            self._t_conv = None
            self._t_scattered = None
            self._t_almost_scattered = None
        self._magnitude = None
        self._weights = None
        self._magnitude_dimension_profile = None
        self._magnitude_dimension = None
        self._magnitude_area = None
        self._weights = None
        self._ts_dim = None

    def remove_points(self, ind_delete, update_ts=False):
        """
        Remove observations.

        Parameters
        ----------
        ind_delete :
            The indices of the observations to remove.
        update_ts : bool
            Whether to update the scales of evaluation.

        Returns
        -------
        Magnipy :
            A Magnipy object with the observations removed.
        """

        if (self._X is None) and (self._Adj is None):
            raise Exception("There are no points to remove!")
        else:
            if self._X is not None:
                X = np.delete(self._X, ind_delete, axis=0)
                self._X = X

            if (self._Adj is not None) | (self._metric == "isomap"):
                if self._Adj is not None:
                    self._Adj = np.delete(self._Adj, ind_delete, axis=0)
                    self._Adj = np.delete(self._Adj, ind_delete, axis=1)
                D = self._get_dist(
                    self._X,
                    Adj=self._Adj,
                )
            else:
                D = np.delete(self._D, ind_delete, axis=0)
                D = np.delete(D, ind_delete, axis=1)
            self._D = D
            self._Z = similarity_matrix(self._D)
            # self._D = get_dist(X, p=self._p, metric=self._metric, normalise_by_diameter=False,
            # n_neighbors=self._n_neighbors)
            self._n = self._D.shape[0]
        if update_ts:
            self._ts = None
            self._t_conv = None
            self._t_scattered = None
            self._t_almost_scattered = None
        self._magnitude = None
        self._weights = None
        self._magnitude_dimension_profile = None
        self._magnitude_dimension = None
        self._magnitude_area = None
        self._weights = None
        self._ts_dim = None

    def _cut_until_scale(self, t_cut):
        """
        Cut the magnitude functions at a given scale.

        Parameters
        ----------
        t_cut : float
            The scale at which to cut the magnitude functions.
        """
        if self._magnitude is not None:
            self._magnitude, self._ts = cut_until_scale(
                self._ts,
                self._magnitude,
                t_cut=t_cut,
                D=self._Z,
                method=self._method,
                magnitude_from_distances=self._compute_mag,
            )
        elif self._ts is not None:
            self._ts = cut_ts(self._ts, t_cut)
        self._magnitude_area = None
        self._magnitude_dimension = None
        if self._magnitude_dimension_profile is not None:
            self._magnitude_dimension_profile, self._ts_dim = cut_until_scale(
                self._ts_dim,
                self._magnitude_dimension_profile,
                t_cut=t_cut,
                D=None,
                method=self._method,
            )
        if self._weights is not None:
            self._weights = self._weights[: len()]

    def copy(self):
        """
        Return a copy of the Magnipy object.
        """
        return copy.deepcopy(self)

    def _subtract(self, other, t_cut=None, exact=True):
        """
        Subtract the magnitude functions of two Magnipy objects.

        Parameters
        ----------
        other : Magnipy
            The other Magnipy object.
        t_cut : float
            The scale at which to cut the magnitude functions.
        exact : bool
            Whether to compute the magnitude difference exactly.

        Returns
        -------
        Magnipy
            The difference of the magnitude functions
        """
        if self._metric != other._metric:
            raise Exception(
                "Magnitude functions need to share the same notion of distance in order to be subtracted across the same scales of t!!"
            )
        combined = Magnipy(None)
        combined._magnitude, combined._ts = diff_of_functions(
            self._magnitude,
            self._ts,
            self._Z,
            other._magnitude,
            other._ts,
            other._Z,
            method=self._method,
            exact=exact,
            t_cut=t_cut,
            magnitude_from_distances=self._compute_mag,
            magnitude_from_distances2=other._compute_mag,
        )
        combined._n_ts = len(combined._ts)
        return combined

    def _add(self, other, t_cut=None, exact=True):
        """
        Add the magnitude functions of two Magnipy objects.

        Parameters
        ----------
        other : Magnipy
            The other Magnipy object.
        t_cut : float
            The scale at which to cut the magnitude functions.
        exact : bool
            Whether to compute the magnitude sum exactly.

        Returns
        -------
        Magnipy
            The sum of the magnitude functions
        """
        if self._metric != other._metric:
            raise Exception(
                "Magnitude functions need to share the same notion of distance in order to be added across the same scales of t!!"
            )
        combined = Magnipy(None)
        combined._magnitude, combined._ts = sum_of_functions(
            self._magnitude,
            self._ts,
            self._Z,
            other._magnitude,
            other._ts,
            other._Z,
            method=self._method,
            exact=exact,
            t_cut=t_cut,
            magnitude_from_distances=self._compute_mag,
            magnitude_from_distances2=other._compute_mag,
        )
        combined._n_ts = len(combined._ts)
        return combined

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Diversity Summaries                                      │
    #  ╰──────────────────────────────────────────────────────────╯

    def MagArea(
        self,
        t_cut=None,
        integration="trapz",
        absolute_area=True,
        scale=False,
        plot=False,
    ):
        """
        Compute MagArea, the area under the magnitude function.

        Parameters
        ----------
        t_cut : float
            The scale at which to cut the magnitude function.
        integration : str
            The method of integration to use.
        absolute_area : bool
            Whether to compute the absolute area.
        scale : bool
            Whether to scale the magnitude functions to be on a domain [0,1] before computing the area.
        plot : bool
            Whether to plot the magnitude function.

        Returns
        -------
        mag_area : float
            The area under the magnitude function.
        """
        if self._magnitude is None:
            _, _ = self.get_magnitude()

        if self._magnitude_area is None:
            self._magnitude_area = mag_area(
                magnitude=self._magnitude,
                ts=self._ts,
                D=self._D,
                t_cut=t_cut,
                integration=integration,  # normalise_by_cardinality=False,
                absolute_area=absolute_area,
                scale=scale,
                plot=plot,
                name=self._name,
            )

        return self._magnitude_area

    def MagDiff(
        self,
        other,
        t_cut=None,
        integration="trapz",
        absolute_area=True,
        scale=False,
        plot=False,
        exact=True,
    ):
        """
        Compute MagDiff i.e. the area between the magnitude functions of two Magnipy objects.

        Parameters
        ----------
        other : Magnipy
            The other Magnipy object.
        t_cut : float
            The scale at which to cut the magnitude functions.
        integration : str
            The method of integration to use.
        absolute_area : bool
            Whether to compute the absolute area.
        scale : bool
            Whether to scale the magnitude functions to be on a domain [0,1] before computing the difference.
        plot : bool
            Whether to plot the magnitude function difference.
        exact : bool
            Whether to compute the magnitude difference exactly.

        Returns
        -------
        mag_difference : float
            The magnitude difference between the two magnitude functions.
        """
        if self._magnitude is None:
            _, _ = self.get_magnitude()
        if other._magnitude is None:
            _, _ = other.get_magnitude()
        mag_difference = mag_diff(
            self._magnitude,
            self._ts,
            self._D,
            other._magnitude,
            other._ts,
            other._D,
            method=self._method,
            exact=exact,
            t_cut=t_cut,
            integration=integration,
            absolute_area=absolute_area,
            scale=scale,
            plot=plot,
            name=self._name + " - " + other._name,
        )
        return mag_difference
