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
    magitude_dimension_profile_interp,
    magnitude_dimension,
    magnitude_dimension_profile_exact,
)
from magnipy.magnitude.distances import (
    get_dist,
    compute_subgraphs_with_dist,
    to_attributed_graph,
)
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
import networkx as nx
from magnipy.magnitude.compute import (
    compute_magnitude_subgraphs,
    compute_magnitude_subgraphs_with_dist,
    compute_magnitude_from_distances,
)

import warnings


class Graphipy:
    def __init__(
        self,
        # Input data parameters
        X=None,
        # Parameters for the evaluation scales
        ts=None,
        n_ts=30,
        log_scale=False,
        return_log_scale=False,
        scale_finding="convergence",
        target_prop=0.95,
        # Parameters for the distance matrix
        metric="diffusion_distance",  # mode structure and metric euclidean not compatible
        custom_dist_fn=None,
        mode="structure",
        G=None,
        # Parameters for the computation of magnitude
        method="cholesky",
        one_point_property=True,
        perturb_singularities=True,
        positive_magnitude=False,
        # Other parameters
        recompute=False,
        name="",
        **kwargs,
    ):
        """
        Initialises a Graphipy object.

        Parameters
        ----------
        Input data parameters:
        X : array_like, shape (`n_obs`, `n_vars`)
            A dataset whose rows are observations and columns are features.
        G : networkx.Graph
            A graph used to compute distances based on its subgraphs.

        Parameters for the evaluation scales:
        ts : array_like, shape (`n_ts`, )
            The scales at which to evaluate the magnitude functions. If None, the scales are computed automatically.
        n_ts : int
            The number of scales at which to evaluate the magnitude functions. Computations are faster for fewer scales and more accurate for more scales.
        log_scale : bool
            Whether to use a log-scale for the evaluation scales.
        return_log_scale : bool
            Whether to return the scales on log-scale when computing the magnitude dimension profile.
        scale_finding : str
            The method to use to find the scale at which to evaluate the magnitude functions. Either 'scattered', 'convergence', or 'median_heuristic'.
        target_prop : float
            The proportion of points that are scattered OR the proportion of cardinality that the magnitude functon converges to.

        Parameters for the distance matrix:
        metric : str
            The distance metric to use. The distance function can be
            'Lp', 'isomap',
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'precomputed',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule',
            "shortest_path_distance", "resistance_distance", "diffusion_distance", "heat_kernel_distance".
        mode : str
            The mode of distance computation. Can be either 'attributes', 'structure', or 'full'.
        **kwargs :
            Additional keyword arguments passed to the distance computation function.

        Parameters for the computation of magnitude:
        method : str
            The method to use to compute the magnitude functions.
            One of 'cholesky', 'scipy', 'scipy_sym', 'spread', 'naive', 'pinv', 'conjugate_gradient_iteration', 'cg'.
        one_point_property : bool
            Whether to enforce the one-point property.
        perturb_singularities : bool
            Whether to perturb the simularity matrix whenever singularities in the magnitude function occure.
        positive_magnitude : bool
            Whether to compute positive magnitude, by taking only the sum of the positive weights.
        recompute : bool
            Whether to recompute the magnitude functions if they have already been computed.
        name : str
            The name of the Graphipy object.

        Returns
        -------
        Graphipy :
            A Graphipy object.
        """

        self._mode = mode
        if mode not in ["attributes", "structure", "full"]:
            raise Exception(
                "The mode of distance computation must be either 'attributes', 'structure', or 'full'."
            )

        ### Check if the input matrix X is valid
        if X is not None:
            if not isinstance(X, np.ndarray):
                raise Exception("The input matrix must be a numpy array.")
            if G.nodes[0].get("feature") is not None:
                warnings.warn(
                    '"The graph already has features assigned to its nodes. Overriding the features with the input X."'
                )
                # raise Warning("The graph already has features assigned to its nodes. Overriding the features with the input X.")
            G = to_attributed_graph(X, G)
        else:
            # if mode == "attributes" or mode == "full":
            if (mode == "structure") or (mode == "full"):
                if G is None:
                    raise Exception(
                        "The graph must be specified when mode is 'structure' or 'full'."
                    )
            if (mode == "attributes") or (mode == "full"):
                if G is not None:
                    if G.nodes[0].get("feature") is not None:
                        X = np.array([G.nodes[i]["feature"] for i in G.nodes])
                    else:
                        raise Exception(
                            "Either the input matrix X must be specified or the graph nodes must have a 'feature' attribute."
                        )

        ### Check if the inputs used for scale-finding are valid
        if isinstance(target_prop, float):
            if X is None:
                min_mag = 1 / G.number_of_nodes()
            else:
                min_mag = 1 / X.shape[0]
            if (target_prop < min_mag) | (target_prop > 1):
                raise Exception(
                    f"The target proportion must be between {min_mag} and 1."
                )
        else:
            raise Exception("The target proportion must be a float.")

        self._proportion_scattered = target_prop
        if (
            (scale_finding != "scattered")
            & (scale_finding != "convergence")
            & (scale_finding != "median_heuristic")
        ):
            raise Exception(
                "The scale finding method must be either 'scattered', 'convergence', or 'median_heuristic'."
            )
        self._scale_finding = scale_finding

        ### Check if the evaluation scales are valid
        self._ts = ts
        if not isinstance(n_ts, int):
            raise Exception("n_ts must be an integer.")
        self._n_ts = n_ts
        # self._compute_subgraphs = False

        ### Check if the adjacency matrix is valid

        # if G is not None:
        if not isinstance(G, nx.Graph):
            raise Exception("The input graph must be a networkx graph.")
        if X is not None:
            if G.number_of_nodes() != X.shape[0]:
                raise Exception(
                    "The number of nodes in the graph must be equal to the number of rows in the dataset."
                )
            for i, feature in enumerate(X):
                G.nodes[i]["feature"] = feature
        # if not nx.is_connected(G):
        #    self._compute_subgraphs = True

        ### Setting up the distance computations and the similarity matrix
        self._G = G
        self._metric = metric

        # self._X = X

        if custom_dist_fn is not None:
            self._get_dist = custom_dist_fn
        else:

            def compute_distances(X=None, X2=None, G=None):
                return get_dist(
                    X=X,
                    X2=X2,
                    G=G,
                    metric=metric,
                    mode=mode,
                    normalise_by_diameter=False,
                    **kwargs,
                )

            self._get_dist = compute_distances

        ### Check if the method for computing the magnitude is valid and set up the magnitude computations
        if method not in [
            "cholesky",
            "scipy",
            "scipy_sym",
            "naive",
            "pinv",
            "conjugate_gradient_iteration",
            "cg",
            "spread",
            "solve_torch",
            "cholesky_torch",
            "naive_torch",
            "lstq_torch",
            "spread_torch",
            "pinv_torch",
        ]:
            raise Exception(
                "The computation method must be one of 'cholesky', 'scipy', 'scipy_sym', 'naive', 'pinv', 'conjugate_gradient_iteration', 'cg', 'spread'."
            )

        # if self._compute_subgraphs:
        def compute_mag(
            Zs,
            ts,
            n_ts=n_ts,  # not used??
            get_weights=False,
            one_point_property=one_point_property,
            perturb_singularities=perturb_singularities,
            positive_magnitude=positive_magnitude,
        ):
            mags = []
            for Z in Zs:
                mag = compute_magnitude_from_distances(
                    Z,
                    ts=ts,
                    method=method,
                    get_weights=get_weights,
                    one_point_property=one_point_property,
                    perturb_singularities=perturb_singularities,
                    positive_magnitude=positive_magnitude,
                    input_distances=False,
                )
                mags.append(mag)
            total_magnitude = np.sum([mag for mag in mags], axis=0)
            return total_magnitude, ts

        subgraphs, Ds = compute_subgraphs_with_dist(
            G, dist_fn=self._get_dist, subgraphs=None
        )
        #### Suggestion Nadja
        # subgraphs, Ds = compute_subgraphs_with_dist( #maybe pass X and X2 to this function?
        #     G, X=X, X2=X, dist_fn=self._get_dist, subgraphs=None
        # )   
        self._subgraphs = subgraphs
        self._Ds = Ds
        # self._D = self._get_dist(X, X2=None, Adj=self._Adj, G=self._G)
        self._n = self._G.number_of_nodes()
        # self._D.shape[0]
        self._target_value = target_prop * G.number_of_nodes()
        # self._Z = similarity_matrix(self._D)
        self._Zs = [similarity_matrix(D) for D in Ds]

        self._compute_mag = compute_mag
        self._method = method
        # self._p = p
        # self._n_neighbors = n_neighbors

        ### Check if the boolean parameters are valid
        for k, arg in enumerate(
            [log_scale, return_log_scale, recompute, positive_magnitude]
        ):
            arg_name = [
                "log_scale",
                "return_log_scale",
                "recompute",
                "positive_magnitude",
            ][k]
            if not isinstance(arg, bool):
                raise Exception(f"{arg_name} must be a boolean.")

        self._log_scale = log_scale
        self._one_point_property = one_point_property
        self._perturb_singularities = perturb_singularities
        self._return_log_scale = return_log_scale
        self._recompute = recompute
        self._positive_magnitude = positive_magnitude

        ### Set the name of the Magnipy object
        self._name = name

        ### Set the other parameters
        self._magnitude = None
        self._weights = None
        self._magnitude_dimension_profile = None
        self._ts_dim = None
        self._t_conv = None
        self._magnitude_dimension = None
        self._magnitude_area = None
        self._t_scattered = None
        self._t_almost_scattered = None
        self._t_median = None

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
        Compute the distance matrices.
        """
        if (self._Ds is None) | self._recompute:
            self._Ds = [self._get_dist(G=s) for s in self._subgraphs]
        return self._Ds

    def get_similarity_matrix(self):
        """
        Compute the similarity matrix.
        """
        if (self._Zs is None) | self._recompute:
            self._Zs = [similarity_matrix(D) for D in self._Ds]
        return self._Zs

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
                    D=self._Zs,
                    comp_mag=comp_mag,
                    target_value=self._target_value,
                    guess=10,
                )

            return self._t_conv
        elif self._scale_finding == "scattered":
            return self._scale_when_almost_scattered(q=None)  # not defined
        elif self._scale_finding == "median_heuristic":
            return self._median_heuristic_scale()

    def get_scales(self):
        """
        Compute the scales at which to evaluate the magnitude functions.
        """
        if (self._ts is None) | self._recompute:
            if self._scale_finding == "convergence":
                if (self._t_conv is None) | self._recompute:
                    _ = self.get_t_conv()
                self._ts = get_scales(
                    self._t_conv,
                    self._n_ts,
                    log_scale=self._log_scale,
                    one_point_property=self._one_point_property,
                )
            elif self._scale_finding == "median_heuristic":
                if (self._t_median is None) | self._recompute:
                    _ = self._median_heuristic_scale()
                self._ts = get_scales(
                    self._t_median,
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

    def _median_heuristic_scale(self):
        """
        Compute the scale using the median heuristic.

        Returns
        -------
        t_median : float
            The scale computed using the median heuristic.
        """
        from magnipy.magnitude.scales import median_heuristic

        if (self._t_median is None) | self._recompute:
            if self._compute_subgraphs:  # no longer used?
                raise Exception("Not implemented for subgraphs.")
            self._t_median = median_heuristic(
                self._get_dist, G=None, subgraphs=self._subgraphs, Ds=self._Ds
            )
        return self._t_median

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
            weights = []
            for Z in self._Zs:
                w, ts = self._compute_mag(
                    Z=Z,
                    ts=ts,
                    get_weights=True,
                )
                weights.append(w)
            self._weights = weights
            self._ts = ts
            if self._ts is None:
                self._t_conv = ts[-1]  # ?
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
                self._Zs,
                ts=ts,
                get_weights=False,
            )

            if self._ts is None:  # why
                self._t_conv = ts[-1]
                self._ts = ts

        elif (self._magnitude is None) & (not (self._weights is None)):
            total_mag = np.zeros(len(self._weights[0][0]))
            for w in self._weights:
                mag = magnitude_from_weights(w)
                total_mag += mag
            self._magnitude = total_mag
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
            self._Zs,
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

            if self._magnitude is None:
                _, _ = self.get_magnitude()
            (
                self._magnitude_dimension_profile,
                self._ts_dim,
            ) = magitude_dimension_profile_interp(
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

    def copy(self):
        """
        Return a copy of the Magnipy object.
        """
        return copy.deepcopy(self)

    def _subtract(self, other, t_cut=None, exact=False):
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

        combined_magnitude, combined_ts = diff_of_functions(
            self._magnitude,
            self._ts,
            None,
            other._magnitude,
            other._ts,
            None,
            method=self._method,
            exact=exact,
            t_cut=t_cut,
            magnitude_from_distances=self._compute_mag,
            magnitude_from_distances2=other._compute_mag,
        )
        # combined._n_ts = len(combined._ts)

        return combined_magnitude, combined_ts

    def _add(self, other, t_cut=None, exact=False):
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

        combined_magnitude, combined_ts = sum_of_functions(
            self._magnitude,
            self._ts,
            None,
            other._magnitude,
            other._ts,
            None,
            method=self._method,
            exact=exact,
            t_cut=t_cut,
            magnitude_from_distances=self._compute_mag,
            magnitude_from_distances2=other._compute_mag,
        )
        # combined._n_ts = len(combined._ts)
        return combined_magnitude, combined_ts

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Diversity Summaries                                      │
    #  ╰──────────────────────────────────────────────────────────╯

    def MagArea(
        self,
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
                D=None,
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
        exact=False,
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
            None,
            other._magnitude,
            other._ts,
            None,
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
