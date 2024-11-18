from magnipy import Magnipy
import numpy as np
from magnipy.magnitude.scales import get_scales


class MagDiversity:
    def __init__(
        self,
        Xs,
        ts=None,
        scale_finding="convergence",
        target_prop=0.95,
        n_ts=30,
        metric="euclidean",
        p=2,
        n_neighbors=12,
        names=None,
    ):
        """
        Compute the magnitude diversity profile of a dataset X.
        """

        self.Xs = Xs
        #self._Adj = Adj

        if method is None:
            if metric in ["euclidean", "Lp", "minowski", "cityblock", "cosine"]:
                method = "cholesky"
            else:
                method = "scipy_sym"

        t_convs = []
        Mags = []
        if names is None:
            names = [f"X_{i}" for i in range(len(Xs))]
        for i, X in enumerate(Xs):
            Mag = Magnipy(
                X,
                ts=ts,
                scale_finding="convergence",
                target_prop=target_prop,
                n_ts=n_ts,
                log_scale=False,
                method=method,
                metric=metric,
                p=p,
                one_point_property=True,
                return_log_scale=False,
                perturb_singularities=True,
                recompute=False,
                name=names[i],
                positive_magnitude=False,
            )
            t_convs.append(Mag.get_t_conv())
            Mags.append(Mag)

        self._t_convs = t_convs
        self._Mags = Mags
        self._names = names

        self._scale_finding = scale_finding
        self._ts = ts
        self._n_ts = n_ts
        self._method = method
        self._metric = metric
        self._p = p
        self._n_neighbors = n_neighbors

    def get_common_scales(self, quantile=0.5):
        t_cut = np.quantile(self._t_convs, quantile)
        ts = get_scales(
            t_cut,
            self._n_ts,
            log_scale=False,
            one_point_property=self._one_point_property,
            scale_finding="convergence",
        )
        self._ts = ts
        self._t_cut = t_cut
        return ts

    def choose_common_scales(self):
        for Mag in self._Mags:
            Mag.change_scales(ts=self.get_common_scales())
        ts = np.concatenate(ts)
        ts = np.unique(ts)
        return ts
