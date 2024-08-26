from magnipy.magnitude import compute_t_conv, get_scales, scale_when_scattered, scale_when_almost_scattered, compute_magnitude_until_convergence, magnitude_from_weights
from magnipy.magnitude_dimension import magitude_dimension_profile, magnitude_dimension, magnitude_dimension_profile_exact
from magnipy.distances import get_dist
from magnipy.summaries import mag_area, mag_diff
from magnipy.function_utils import diff_of_functions, sum_of_functions, plot_magnitude_function, plot_magnitude_dimension_profile, cut_until_scale, cut_ts
import numpy as np

class Magnipy:
    def __init__(self, X, D=None, ts=None, target_value=None, target_prop=0.95,  n_ts=10, log_scale = False, method="cholesky",
                 metric="Lp", p=2, one_point_property=True, proportion_scattered=None, scale_finding="convergence",
                 n_neighbors=12, return_log_scale=False, perturb_singularities=True, recompute=False, name="", 
                                        positive_magnitude=False):	
        

        self._X = X
        self._target_value = target_value
        if ((X is None) & (D is None)):
            self._D = None
            self._n = None
        elif (D is None):
            self._D = get_dist(X, p=p, metric=metric, normalise_by_diameter=False, n_neighbors=n_neighbors)
            self._n = self._D.shape[0]
            if target_value is None:
                self._target_value = target_prop* self._D.shape[0]
        else:
            self._D = D
            self._n = self._D.shape[0]
            if target_value is None:
                self._target_value = target_prop* self._D.shape[0]

        self._proportion_scattered=proportion_scattered
        if (scale_finding != "scattered") & (scale_finding != "convergence"):
            raise Exception("The scale finding method must be either 'scattered' or 'convergence'.")
        self._scale_finding=scale_finding
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

        self._magnitude = None
        self._weights = None
        self._magnitude_dimension_profile = None
        self._ts_dim = None
        self._ts = None
        self._t_conv = None
        self._magnitude_dimension = None
        self._magnitude_area = None
        self._name=name
        self._t_scattered = None
        self._t_almost_scattered = None
    
    def get_dist(self):
        return self._D

    def get_name(self):
        return self._name

    def get_magnitude_weights(self):
        if (self._weights is None) | self._recompute:
            ts=self.get_scales()
            weights, ts = compute_magnitude_until_convergence(self._D, ts=self._ts, n_ts=self._n_ts, method=self._method, 
                                                                log_scale = self._log_scale, get_weights=True, 
                                                                one_point_property=self._one_point_property, perturb_singularities=self._perturb_singularities, 
                                                                positive_magnitude=self._positive_magnitude)
        if self._ts is None:
            self._t_conv = ts[-1]
        self._weights = weights
        self._ts = ts
        return weights, ts
    
    def get_magnitude(self):
        if ((self._magnitude is None) & (self._weights is None)) | self._recompute:
            ts=self.get_scales()
            self._magnitude, ts = compute_magnitude_until_convergence(self._D, ts=self._ts, n_ts=self._n_ts, method=self._method, 
                                                            log_scale = self._log_scale, get_weights=False, 
                                                            one_point_property=self._one_point_property, perturb_singularities=self._perturb_singularities, 
                                                            positive_magnitude=self._positive_magnitude)
            if self._ts is None:
                self._t_conv = ts[-1]
                self._ts = ts
        elif (self._magnitude is None) & ~(self._weights is None):
             self._magnitude = magnitude_from_weights(self._weights)
        return self._magnitude, self._ts
    
    def plot_magnitude_function(self):
        if (self._magnitude is None) | self._recompute:
            _, _ = self.get_magnitude()
        plot_magnitude_function(self._ts, self._magnitude, name=self._name)
    
    def get_magnitude_dimension_profile(self, exact=False, h=None):
        if (self._magnitude_dimension_profile is None) | self._recompute:
            if exact:
                self._magnitude_dimension_profile, self._ts_dim = magnitude_dimension_profile_exact(self._D, ts=self._ts, h=h, target_value=self._target_value, n_ts=self._n_ts, 
                                      return_log_scale=self._return_log_scale, one_point_property=self._one_point_property, method=self._method, 
                                                            log_scale = self._log_scale)
            else:
                if (self._magnitude is None):
                    _, _ = self.get_magnitude()
                self._magnitude_dimension_profile, self._ts_dim, = magitude_dimension_profile(mag=self._magnitude, ts=self._ts, return_log_scale=self._return_log_scale,
                                                                                                one_point_property=self._one_point_property)
        return self._magnitude_dimension_profile, self._ts_dim

    def plot_magnitude_dimension_profile(self):
        if (self._magnitude_dimension_profile is None) | self._recompute:
            _, _ = self.get_magnitude_dimension_profile()
        plot_magnitude_dimension_profile(ts=self._ts_dim, mag_dim=self._magnitude_dimension_profile, log_scale=self._return_log_scale, name=self._name)

    def get_t_conv(self):
        if self._scale_finding == "convergence":
            if (self._t_conv is None) | self._recompute:
                self._t_conv = compute_t_conv(self._D, target_value=self._target_value, method=self._method, 
                                               positive_magnitude=self._positive_magnitude)
            return self._t_conv
        elif self._scale_finding == "scattered":
            return self._scale_when_almost_scattered(q=None)
    
    def get_scales(self):
        if (self._ts is None) | self._recompute:
            if self._scale_finding == "scattered":
                if self._proportion_scattered is None | self._recompute:
                    _ = self._scale_when_almost_scattered(q=self._proportion_scattered)
                self._ts = get_scales(self._t_almost_scattered, self._n_ts, log_scale = self._log_scale, 
                                       one_point_property = self._one_point_property)
            elif self._scale_finding == "convergence":
                if (self._t_conv is None) | self._recompute:
                    _ = self.get_t_conv()
                self._ts = get_scales(self._t_conv, self._n_ts, log_scale = self._log_scale, 
                                       one_point_property = self._one_point_property)
        return self._ts
    
    def change_scales(self, ts=None, t_cut=None):
        if ts is None:
            if t_cut is None:
                self._ts = None
                #raise Exception("A new evaluation interval or a cut-off scale need to be specified to change the evaluation scales!")
            else:
                self._ts = get_scales(t_cut, self._n_ts, log_scale = self._log_scale, one_point_property = self._one_point_property)
        else:
            self._ts = ts
        self._magnitude = None
        self._magnitude_dimension_profile = None
        self._magnitude_dimension = None
        self._magnitude_area = None
        self._weights = None
        self._ts_dim = None

    def _eval_at_scales(self, ts_new, get_weights=False):
        mag, ts = compute_magnitude_until_convergence(self._D, ts=ts_new, method=self._method, get_weights=get_weights, 
                                                            one_point_property=self._one_point_property, 
                                                            perturb_singularities=self._perturb_singularities,
                                                            positive_magnitude=self._positive_magnitude)
        return mag, ts

    def _cut_until_scale(self, t_cut):
        if self._magnitude is not None:
            self._magnitude, self._ts = cut_until_scale(self._ts, self._magnitude, t_cut=t_cut, D=self._D, 
                                                          method=self._method, positive_magnitude=self._positive_magnitude)
        elif self._ts is not None:
            self._ts = cut_ts(self._ts, t_cut)
        self._magnitude_area = None
        self._magnitude_dimension = None
        if self._magnitude_dimension_profile is not None:
            self._magnitude_dimension_profile, self._ts_dim = cut_until_scale(self._ts_dim, self._magnitude_dimension_profile, t_cut=t_cut, D=None, 
                                                                                method=self._method, positive_magnitude=self._positive_magnitude)
        if self._weights is not None:
            self._weights = self._weights[:len()]

    def get_magnitude_dimension(self, exact=False):
        if self._magnitude_dimension_profile is None:
            _, _ = self.get_magnitude_dimension_profile()
        if (self._magnitude_dimension is None) | self._recompute:
            self._magnitude_dimension = magnitude_dimension(self._magnitude_dimension_profile)
        return self._magnitude_dimension
    
    def get_magnitude_area(self, t_cut=None, integration="trapz",
            absolute_area=True, scale=False, plot=False):
        if self._magnitude is None:
            _, _ = self.get_magnitude()
        if self._magnitude_area is None:
            self._magnitude_area = mag_area(magnitude=self._magnitude, ts=self._ts,  D=self._D, t_cut=t_cut, integration=integration, #normalise_by_cardinality=False, 
            absolute_area=absolute_area, scale=scale, plot=plot, name=self._name, positive_magnitude=self._positive_magnitude)
        return self._magnitude_area
    
    def include_points(self, X_new, update_ts=False):
        if self._X is None:
            self._X = X_new
            self._D = get_dist(X_new, p=self._p, metric=self._metric, normalise_by_diameter=False, n_neighbors=self._n_neighbors)
            self._n = self._D.shape[0]
        else:
            X = np.concatenate((self._X, X_new), axis=0)
            self._D = get_dist(X, p=self._p, metric=self._metric, normalise_by_diameter=False, n_neighbors=self._n_neighbors)
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
    
    def _substract(self, other, t_cut=None, exact=True):
        if self._metric != other._metric:
            raise Exception("Magnitude functions need to share the same notion of distance in order to be substracted across the same scales of t!!")
        combined = Magnipy(None)
        combined._magnitude, combined._ts = diff_of_functions(self._magnitude, self._ts, self._D, 
                                                                other._magnitude, other._ts, other._D, method=self._method, 
                                                                exact=exact, t_cut=t_cut, positive_magnitude=self._positive_magnitude)
        combined._n_ts = len(combined._ts)
        return combined

    def _add(self, other, t_cut=None, exact=True):
        if self._metric != other._metric:
            raise Exception("Magnitude functions need to share the same notion of distance in order to be added across the same scales of t!!")
        combined = Magnipy(None)
        combined._magnitude, combined._ts = sum_of_functions(self._magnitude, self._ts, self._D, 
                                                                other._magnitude, other._ts, other._D, method=self._method, 
                                                                exact=exact, t_cut=t_cut, positive_magnitude=self._positive_magnitude)
        combined._n_ts = len(combined._ts)
        return combined
    
    def get_magnitude_difference(self, other, t_cut=None, integration="trapz",
            absolute_area=True, scale=False, plot=False, exact=True):
        if self._magnitude is None:
            _, _ = self.get_magnitude()
        if other._magnitude is None:
            _, _ = other.get_magnitude()
        mag_difference = mag_diff(self._magnitude, self._ts, self._D, other._magnitude, other._ts, other._D,  method=self._method, 
                                                                exact=exact, t_cut=t_cut, integration=integration, 
                                                                absolute_area=absolute_area, scale=scale, plot=plot, name=self._name + " - "+other._name, 
                                                                positive_magnitude=self._positive_magnitude)
        return mag_difference
    
    def MagDiff(self, other, t_cut=None, integration="trapz",
            absolute_area=True, scale=False, plot=False, exact=True):
        return self.get_magnitude_difference(other, t_cut=t_cut, integration=integration,
            absolute_area=absolute_area, scale=scale, plot=plot, exact=exact)
    
    def MagArea(self, t_cut=None, integration="trapz",
            absolute_area=True, scale=False, plot=False):
        return self.get_magnitude_area(t_cut=t_cut, integration=integration,
            absolute_area=absolute_area, scale=scale, plot=plot)
    
    def _scale_when_scattered(self):
        if (self._t_scattered is None) | self._recompute:
            self._t_scattered = scale_when_scattered(self._D)
        return self._t_scattered
    
    def _scale_when_almost_scattered(self, q=None):
        if (self._t_almost_scattered is None) | self._recompute:
            self._t_almost_scattered = scale_when_almost_scattered(self._D, n=self._n, q=q)
        return self._t_almost_scattered