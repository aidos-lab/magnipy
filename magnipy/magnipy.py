from magnipy.magnitude import compute_t_conv, get_scales, scale_when_scattered, scale_when_almost_scattered, compute_magnitude_until_convergence, magnitude_from_weights
from magnipy.magnitude_dimension import magitude_dimension_profile, magnitude_dimension, magnitude_dimension_profile_exact
from magnipy.distances import get_dist
from magnipy.summaries import mag_area, mag_diff
from magnipy.function_utils import diff_of_functions, sum_of_functions, plot_magnitude_function, plot_magnitude_dimension_profile, cut_until_scale, cut_ts
import numpy as np
#from sklearn.preprocessing import normalize

class Magnipy:
    def __init__(self, X, D=None, ts=None, target_value=None, target_prop=0.95,  n_ts=10, log_scale = False, method="cholesky",
                 metric="Lp", p=2, one_point_property=True, proportion_scattered=None, scale_finding="convergence",
                 n_neighbors=12, return_log_scale=False, perturb_singularities=True, recompute=False, name=""):	
        self.__X = X
        self.__target_value = target_value
        if ((X is None) & (D is None)):
            self.__D = None
            self.__n = None
        elif (D is None):
            self.__D = get_dist(X, p=p, metric=metric, normalise_by_diameter=False, n_neighbors=n_neighbors)
            self.__n = self.__D.shape[0]
            if target_value is None:
                self.__target_value = target_prop* self.__D.shape[0]
        else:
            self.__D = D
            self.__n = self.__D.shape[0]
            if target_value is None:
                self.__target_value = target_prop* self.__D.shape[0]

        self.__proportion_scattered=proportion_scattered
        if (scale_finding != "scattered") & (scale_finding != "convergence"):
            raise Exception("The scale finding method must be either 'scattered' or 'convergence'.")
        self.__scale_finding=scale_finding
        self.__ts = ts
        self.__n_ts = n_ts
        self.__log_scale = log_scale
        self.__method = method
        self.__metric = metric
        self.__p = p
        self.__one_point_property = one_point_property
        self.__perturb_singularities = perturb_singularities
        self.__n_neighbors = n_neighbors
        self.__return_log_scale = return_log_scale
        self.__recompute = recompute

        self.__magnitude = None
        self.__weights = None
        self.__magnitude_dimension_profile = None
        self.__ts_dim = None
        self.__ts = None
        self.__t_conv = None
        self.__magnitude_dimension = None
        self.__magnitude_area = None
        self.__name=name
        self.__t_scattered = None
        self.__t_almost_scattered = None
    
    def get_dist(self):
        return self.__D

    def get_name(self):
        return self.__name

    def get_magnitude_weights(self):
        if (self.__weights is None) | self.__recompute:
            ts=self.get_scales()
            weights, ts = compute_magnitude_until_convergence(self.__D, ts=self.__ts, n_ts=self.__n_ts, method=self.__method, 
                                                                log_scale = self.__log_scale, get_weights=True, 
                                                                one_point_property=self.__one_point_property, perturb_singularities=self.__perturb_singularities)
        if self.__ts is None:
            self.__t_conv = ts[-1]
        self.__weights = weights
        self.__ts = ts
        return weights, ts
    
    def get_magnitude(self):
        if ((self.__magnitude is None) & (self.__weights is None)) | self.__recompute:
            ts=self.get_scales()
            self.__magnitude, ts = compute_magnitude_until_convergence(self.__D, ts=self.__ts, n_ts=self.__n_ts, method=self.__method, 
                                                            log_scale = self.__log_scale, get_weights=False, 
                                                            one_point_property=self.__one_point_property, perturb_singularities=self.__perturb_singularities)
            if self.__ts is None:
                self.__t_conv = ts[-1]
                self.__ts = ts
        elif (self.__magnitude is None) & ~(self.__weights is None):
             self.__magnitude = magnitude_from_weights(self.__weights)
        return self.__magnitude, self.__ts
    
    def plot_magnitude_function(self):
        if (self.__magnitude is None) | self.__recompute:
            _, _ = self.get_magnitude()
        plot_magnitude_function(self.__ts, self.__magnitude, name=self.__name)
    
    def get_magnitude_dimension_profile(self, exact=False, h=None):
        if (self.__magnitude_dimension_profile is None) | self.__recompute:
            if exact:
                self.__magnitude_dimension_profile, self.__ts_dim = magnitude_dimension_profile_exact(self.__D, ts=self.__ts, h=h, target_value=self.__target_value, n_ts=self.__n_ts, 
                                      return_log_scale=self.__return_log_scale, one_point_property=self.__one_point_property, method=self.__method, 
                                                            log_scale = self.__log_scale,)
            else:
                if (self.__magnitude is None):
                    _, _ = self.get_magnitude()
                self.__magnitude_dimension_profile, self.__ts_dim, = magitude_dimension_profile(mag=self.__magnitude, ts=self.__ts, return_log_scale=self.__return_log_scale,
                                                                                                one_point_property=self.__one_point_property)
        return self.__magnitude_dimension_profile, self.__ts_dim

    def plot_magnitude_dimension_profile(self):
        if (self.__magnitude_dimension_profile is None) | self.__recompute:
            _, _ = self.get_magnitude_dimension_profile()
        plot_magnitude_dimension_profile(ts=self.__ts_dim, mag_dim=self.__magnitude_dimension_profile, log_scale=self.__return_log_scale, name=self.__name)

    def get_t_conv(self):
        if self.__scale_finding == "convergence":
            if (self.__t_conv is None) | self.__recompute:
                self.__t_conv = compute_t_conv(self.__D, target_value=self.__target_value, method=self.__method)
            return self.__t_conv
        elif self.__scale_finding == "scattered":
            return self._scale_when_almost_scattered(q=None)
    
    def get_scales(self):
        if (self.__ts is None) | self.__recompute:
            if self.__scale_finding == "scattered":
                if self.__proportion_scattered is None | self.__recompute:
                    _ = self._scale_when_almost_scattered(q=self.__proportion_scattered)
                self.__ts = get_scales(self.__t_almost_scattered, self.__n_ts, log_scale = self.__log_scale, one_point_property = self.__one_point_property)
            elif self.__scale_finding == "convergence":
                if (self.__t_conv is None) | self.__recompute:
                    _ = self.get_t_conv()
                self.__ts = get_scales(self.__t_conv, self.__n_ts, log_scale = self.__log_scale, one_point_property = self.__one_point_property)
        return self.__ts
    
    def change_scales(self, ts=None, t_cut=None):
        if ts is None:
            if t_cut is None:
                self.__ts = None
                #raise Exception("A new evaluation interval or a cut-off scale need to be specified to change the evaluation scales!")
            else:
                self.__ts = get_scales(t_cut, self.__n_ts, log_scale = self.__log_scale, one_point_property = self.__one_point_property)
        else:
            self.__ts = ts
        self.__magnitude = None
        self.__magnitude_dimension_profile = None
        self.__magnitude_dimension = None
        self.__magnitude_area = None
        self.__weights = None
        self.__ts_dim = None

    def _eval_at_scales(self, ts_new, get_weights=False):
        mag, ts = compute_magnitude_until_convergence(self.__D, ts=ts_new, method=self.__method, get_weights=get_weights, 
                                                            one_point_property=self.__one_point_property, perturb_singularities=self.__perturb_singularities)
        return mag, ts

    def _cut_until_scale(self, t_cut):
        if self.__magnitude is not None:
            self.__magnitude, self.__ts = cut_until_scale(self.__ts, self.__magnitude, t_cut=t_cut, D=self.__D, method=self.__method)
        elif self.__ts is not None:
            self.__ts = cut_ts(self.__ts, t_cut)
        self.__magnitude_area = None
        self.__magnitude_dimension = None
        if self.__magnitude_dimension_profile is not None:
            self.__magnitude_dimension_profile, self.__ts_dim = cut_until_scale(self.__ts_dim, self.__magnitude_dimension_profile, t_cut=t_cut, D=None, method=self.__method)
        if self.__weights is not None:
            self.__weights = self.__weights[:len()]

    def get_magnitude_dimension(self, exact=False):
        if self.__magnitude_dimension_profile is None:
            _, _ = self.get_magnitude_dimension_profile()
        if (self.__magnitude_dimension is None) | self.__recompute:
            self.__magnitude_dimension = magnitude_dimension(self.__magnitude_dimension_profile)
        return self.__magnitude_dimension
    
    def get_magnitude_area(self, t_cut=None, integration="trapz",
            absolute_area=True, scale=False, plot=False):
        if self.__magnitude is None:
            _, _ = self.get_magnitude()
        if self.__magnitude_area is None:
            self.__magnitude_area = mag_area(magnitude=self.__magnitude, ts=self.__ts,  D=self.__D, t_cut=t_cut, integration=integration, #normalise_by_cardinality=False, 
            absolute_area=absolute_area, scale=scale, plot=plot, name=self.__name)
        return self.__magnitude_area
    
    def include_points(self, X_new, update_ts=False):
        if self.__X is None:
            self.__X = X_new
            self.__D = get_dist(X_new, p=self.__p, metric=self.__metric, normalise_by_diameter=False, n_neighbors=self.__n_neighbors)
            self.__n = self.__D.shape[0]
        else:
            X = np.concatenate((self.__X, X_new), axis=0)
            self.__D = get_dist(X, p=self.__p, metric=self.__metric, normalise_by_diameter=False, n_neighbors=self.__n_neighbors)
            self.__n = self.__D.shape[0]
        if update_ts:
            self.__ts = None
            self.__t_conv = None
            self.__t_scattered = None
            self.__t_almost_scattered = None
        self.__magnitude = None
        self.__weights = None
        self.__magnitude_dimension_profile = None
        self.__magnitude_dimension = None
        self.__magnitude_area = None
        self.__weights = None
        self.__ts_dim = None
    
    def _substract(self, other, t_cut=None, exact=True):
        if self.__metric != other.__metric:
            raise Exception("Magnitude functions need to share the same notion of distance in order to be substracted across the same scales of t!!")
        combined = Magnipy(None)
        combined.__magnitude, combined.__ts = diff_of_functions(self.__magnitude, self.__ts, self.__D, 
                                                                other.__magnitude, other.__ts, other.__D, method=self.__method, 
                                                                exact=exact, t_cut=t_cut)
        combined.__n_ts = len(combined.__ts)
        return combined

    def _add(self, other, t_cut=None, exact=True):
        if self._metric != other._metric:
            raise Exception("Magnitude functions need to share the same notion of distance in order to be added across the same scales of t!!")
        combined = Magnipy()
        combined.__magnitude, combined.__ts = sum_of_functions(self.__magnitude, self.__ts, self.__D, 
                                                                other.__magnitude, other.__ts, other.__D, method=self.__method, 
                                                                exact=exact, t_cut=t_cut)
        combined.__n_ts = len(combined.__ts)
        return combined
    
    def get_magnitude_difference(self, other, t_cut=None, integration="trapz",
            absolute_area=True, scale=False, plot=False, exact=True):
        if self.__magnitude is None:
            _, _ = self.get_magnitude()
        if other.__magnitude is None:
            _, _ = other.get_magnitude()
        mag_difference = mag_diff(self.__magnitude, self.__ts, self.__D, other.__magnitude, other.__ts, other.__D,  method=self.__method, 
                                                                exact=exact, t_cut=t_cut, integration=integration, 
                                                                absolute_area=absolute_area, scale=scale, plot=plot, name=self.__name + " - "+other.__name)
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
        if (self.__t_scattered is None) | self.__recompute:
            self.__t_scattered = scale_when_scattered(self.__D)
        return self.__t_scattered
    
    def _scale_when_almost_scattered(self, q=None):
        if (self.__t_almost_scattered is None) | self.__recompute:
            self.__t_almost_scattered = scale_when_almost_scattered(self.__D, n=self.__n, q=q)
        return self.__t_almost_scattered