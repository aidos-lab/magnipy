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
        log_scale=False,
        method="cholesky",
        metric="euclidean",
        p=2,
        Adj=None,
        one_point_property=True,
        n_neighbors=12,
        return_log_scale=False,
        perturb_singularities=True,
        recompute=False,
        name="",
        positive_magnitude=False,
    ):
        """
        Compute the magnitude diversity profile of a dataset X.
        """

        self.Xs = Xs
        self._Adj = Adj

        t_convs = []
        Mags = []
        for X in Xs:
            Mag = Magnipy(
                X,
                ts=ts,
                scale_finding=scale_finding,
                target_prop=target_prop,
                n_ts=n_ts,
                log_scale=log_scale,
                method=method,
                metric=metric,
                p=p,
                one_point_property=one_point_property,
                n_neighbors=n_neighbors,
                return_log_scale=return_log_scale,
                perturb_singularities=perturb_singularities,
                recompute=recompute,
                name=name,
                positive_magnitude=positive_magnitude,
            )
            t_convs.append(Mag.get_t_conv())
            Mags.append(Mag)

        self._t_convs = t_convs
        self._Mags = Mags

        self._scale_finding = scale_finding
        self._ts = ts
        self._n_ts = n_ts
        self._log_scale = log_scale
        self._method = method
        self._metric = metric
        self._p = p
        self._one_point_property = one_point_property
        self._perturb_singularities = perturb_singularities
        self._n_neighbors = n_neighbors
        self._return_log_scale = return_log_scale
        self._recompute = recompute
        self._positive_magnitude = positive_magnitude
        self._name = name

    def get_common_scales(self, quantile=0.5):
        t_cut = np.quantile(self._t_convs, quantile)
        ts = get_scales(
            t_cut,
            self._n_ts,
            log_scale=self._log_scale,
            one_point_property=self._one_point_property,
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
