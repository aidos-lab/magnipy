from magnipy import Magnipy
import numpy as np
from magnipy.magnitude.scales import get_scales


class MagDiversity:
    def __init__(
        self,
        Xs,
        ts=None,
        target_prop=0.95,
        n_ts=30,
        metric="euclidean",
        method=None,
        p=2,
        n_neighbors=12,
        names=None,
    ):
        """
        Compute the magnitude functions for a set of datasets / spaces and compare their diversity.

        Parameters
        ----------
        Xs : list of array_like, shape (`n_obs`, `n_vars`)
            A list of datasets whose rows are observations and columns are features.
        ts : array_like, shape (`n_ts`,)
            The scales at which to compute the magnitude functions. If None, the scales are computed automatically.
        target_prop : float
            The proportion of cardinality that the magnitude functon converges to. 
            Used to finding the evaluation scales across datasets.
        n_ts : int
            The number of scales at which to compute the magnitude functions.
        metric : str
            The distance metric to use. The distance function can be
            'Lp', 'isomap',
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule'.
        method : str
            The method to use to compute the magnitude functions.
            One of 'cholesky', 'scipy', 'scipy_sym', 'inv', 'pinv', 'conjugate_gradient_iteration', 'cg'.
        p : float
            Parameter for the Minkowski metric.
        n_neighbors : int
            The number of neighbors to use in the distance metric.
        names : list of str
            The names of the datasets.
        
        Returns
        -------
        MagDiversity : object
            An object that can be used to compute and compare the magnitude functions of multiple spaces.
        """

        self._Xs = Xs

        if method is None:
            if metric in ["euclidean", "Lp", "minowski", "cityblock", "cosine"]:
                method = "cholesky"
            else:
                method = "scipy_sym"

        if names is None:
            names = [f"X_{i}" for i in range(len(Xs))]

        self._names = names

        self._ts = ts
        self._n_ts = n_ts
        self._method = method
        self._metric = metric
        self._p = p
        self._n_neighbors = n_neighbors
        self._target_prop = target_prop
        self._Mags = None
        self._t_convs = None
        self._MagAreas = None
        self._MagDiffs = None


    def get_common_scales(self, quantile=0.5):
        """
        Determine the shared evaluation interval for the magnitude functions.
        To do this, the convergence scales of the magnitude functions are computed and 
        the shared scales are determined as a quantile (e.g. the median) of the convergence scales for all datasets.
        """
        t_cut = np.quantile(self._t_convs, quantile)
        ts = get_scales(
            t_cut,
            self._n_ts,
            log_scale=False,
            one_point_property=True,
        )
        #self._ts = ts
        self._t_cut = t_cut
        return ts


    def change_scales(self, ts= None, t_cut=None):
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
        for Mag in self._Mags:
            Mag.change_scales(ts=self._ts)
        #self._ts = ts
        return self._Mags
    

    def compute_magnitude(self):
        """
        Compute the magnitude functions for all datasets.
        """

        t_convs = []
        Mags = []
        
        for i, X in enumerate(self._Xs):
            Mag = Magnipy(
                X,
                ts=self._ts,
                scale_finding="convergence",
                target_prop=self._target_prop,
                n_ts=2,
                log_scale=False,
                method=self._method,
                metric=self._metric,
                p=self._p,
                one_point_property=True,
                return_log_scale=False,
                perturb_singularities=True,
                recompute=False,
                name=self._names[i],
                positive_magnitude=False,
            )
            t_convs.append(Mag.get_t_conv())
            Mags.append(Mag)
        self._Mags = Mags
        self._t_convs = t_convs

        ts = self.get_common_scales()
        Mags = self.change_scales(ts)
        self._Mags = Mags
        return Mags
    

    def get_magnitude_functions(self):
        """
        Get the magnitude functions for all datasets.

        Returns
        -------
        mag_df : ndarray, shape (`n_datasets`, `n_ts`)
            The magnitude functions for all datasets.
        ts : array_like, shape (`n_ts`,)
            The scales at which the magnitude functions are evaluated.
        """

        if self._Mags is None:
            self.compute_magnitude()
        mag_df = np.zeros((len(self._Mags), len(self._ts)))
        for i, Mag in enumerate(self._Mags):
            mag_df[i, :] = Mag.get_magnitude()[0]
        return mag_df, self._ts
    
    def plot_magnitude_functions(self):
        """
        Plot the magnitude functions for all datasets.
        """
        if self._Mags is None:
            self.compute_magnitude()
        for Mag in self._Mags:
            Mag.plot_magnitude_function()
        return None

    def get_MagAreas(self,
            integration="trapz",
            absolute_area=True,
            scale=True,
            plot=False):
        """
        Compute the areas under the magnitude functions for all datasets.

        Parameters
        ----------
        integration : str
            The integration method to use. One of 'trapz', 'simps'.
        absolute_area : bool
            Whether to compute the absolute area under the magnitude function.
        scale : bool
            Whether to scale the magnitude function to the interval [0, 1].
        plot : bool
            Whether to plot the magnitude functions.
        
        Returns
        -------
        areas : list of float
            The areas under the magnitude functions for all datasets.
        """
        if self._Mags is None:
            self.compute_magnitude()
        areas = []
        for Mag in self._Mags:
            Mag._magnitude_area = None
            areas.append(Mag.MagArea(integration=integration, absolute_area=absolute_area, 
                                     scale=scale, plot=plot))
        self._MagAreas = areas
        return areas
    
    def get_MagDiffs(self,
            integration="trapz",
            absolute_area=True,
            scale=True,
            plot=False):
        """
        Compute the pairwise differences between the magnitude functions for all datasets.
        
        Parameters
        ----------
        integration : str
            The integration method to use. One of 'trapz', 'simps'.
        absolute_area : bool
            Whether to compute the absolute area between the magnitude functions.
        scale : bool
            Whether to scale the magnitude functions to the interval [0, 1].
        plot : bool
            Whether to plot the magnitude function differences.
        
        Returns
        -------
        diffs : ndarray, shape (`n_datasets`, `n_datasets`)
            The pairwise differences between the magnitude functions for all datasets.
        """
        if self._Mags is None:
            self.compute_magnitude()
        diffs = np.zeros((len(self._Mags), len(self._Mags)))
        for i, Mag in enumerate(self._Mags):
            for j in range(i+1, len(self._Mags)):
                diffs[i,j] = Mag.MagDiff(self._Mags[j], integration=integration, absolute_area=absolute_area, 
                                        scale=scale, plot=plot)
        diffs = diffs + diffs.T
        self._MagDiffs = diffs
        return diffs
    
    def plot_MagDiffs_heatmap(self):
        """
        Plot the pairwise differences between the magnitude functions for all datasets.
        """
        if self._MagDiffs is None:
            self.get_MagDiffs()
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.DataFrame(self._MagDiffs, columns=self._names, index=self._names)
        sns.heatmap(df, annot=False)
        plt.show()
        return None


