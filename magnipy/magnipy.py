from magnitude import compute_t_conv, get_scales, scale_when_scattered, scale_when_almost_scattered, compute_magnitude_until_convergence, magnitude_from_weights
from magnitude_dimension import magitude_dimension_profile, magnitude_dimension, magnitude_dimension_profile_exact
from distances import get_dist
from summaries import mag_area, mag_diff
from function_utils import diff_of_functions, sum_of_functions, plot_magnitude_function, plot_magnitude_dimension_profile, cut_until_scale, cut_ts
import numpy as np

class Magnipy:
    def __init__(self, X, ts=None, target_value=None, n_ts=10, log_scale = True, method="cholesky",
                 metric="Lp", p=2, one_point_property=True, 
                 n_neighbors=12, return_log_scale=False, perturb_singularities=True, recompute=False, name=""):	
        self.__X = X
        self.__target_value = target_value
        if X is None:
            self.__D = None
            self.__n = None
        else:
            self.__D = get_dist(X, p=p, metric=metric, normalise_by_diameter=False, n_neighbors=n_neighbors)
            self.__n = self.__D.shape[0]
            if target_value is None:
                self.__target_value = 0.95* self.__D.shape[0]
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
            weights, ts = compute_magnitude_until_convergence(self.__D, ts=self.__ts, n_ts=self.__n_ts, method=self.__method, 
                                                                log_scale = self.__log_scale, get_weights=True, 
                                                                one_point_property=self.__one_point_property, perturb_singularities=self.__perturb_singularities)
        if self.__ts is None:
            self.__t_conv = ts[-1]
        self.__weights = weights
        self.__ts = ts
        return weights, ts
    
    def get_magnitude(self):
        if (self.__magnitude is None) & ~(self.__weights is None):
             self.__magnitude = magnitude_from_weights(self.__weights)
        elif (self.__magnitude is None) | self.__recompute:
            self.__magnitude, ts = compute_magnitude_until_convergence(self.__D, ts=self.__ts, n_ts=self.__n_ts, method=self.__method, 
                                                            log_scale = self.__log_scale, get_weights=False, 
                                                            one_point_property=self.__one_point_property, perturb_singularities=self.__perturb_singularities)
            if self.__ts is None:
                self.__t_conv = ts[-1]
                self.__ts = ts
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
        if (self.__t_conv is None) | self.__recompute:
            self.__t_conv = compute_t_conv(self.__D, target_value=self.__target_value, method=self.__method)
        return self.__t_conv
    
    def get_scales(self):
        if (self.__ts is None) | self.__recompute:
            if (self.__t_conv is None) | self.__recompute:
                _ = self.get_t_conv()
            self.__ts = get_scales(self.__t_conv, self.__n_ts, log_scale = self.__log_scale, one_point_property = self.__one_point_property)
        return self.__ts
    
    #def get_scales(self):
    
    def change_scales(self, ts=None, t_cut=None):
        if ts is None:
            if t_cut is None:
                self.__ts = None
                #raise Exception("A new evaluation interval or a cut-off scale need to be specified to change the evaluation scales!")
            else:
                self.__ts = get_scales(t_cut, self.__n_ts, log_scale = self.__log_scale, one_point_property = self.__one_point_property)
                #print(self.__ts[-1])
        else:
            self.__ts = ts
        self.__magnitude = None
        self.__magnitude_dimension_profile = None
        self.__magnitude_dimension = None
        self.__magnitude_area = None
        self.__weights = None
        self.__ts_dim = None
        #self.__recompute = True

    def _cut_until_scale(self, t_cut):
        if self.__magnitude is not None:
            self.__magnitude, self.__ts = cut_until_scale(self.__ts, self.__magnitude, t_cut=t_cut, D=self.__D, method=self.__method)
        elif self.__ts is not None:
            self.__ts = cut_ts(self.__ts, t_cut)
        #self.__t_cut = t_cut
        #self.__magnitude = None
        self.__magnitude_area = None
        self.__magnitude_dimension = None
        #self.__magnitude_dimension_profile = None
        if self.__magnitude_dimension_profile is not None:
            self.__magnitude_dimension_profile, self.__ts_dim = cut_until_scale(self.__ts_dim, self.__magnitude_dimension_profile, t_cut=t_cut, D=None, method=self.__method)
        if self.__weights is not None:
            self.__weights = self.__weights[:len()]
        #self.__weights = None
        #self.__ts_dim = None
        #self.__recompute = True

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
    
    def __substract(self, other, t_cut=None, exact=True):
        if self.__metric != other.__metric:
            raise Exception("Magnitude functions need to share the same notion of distance in order to be substracted across the same scales of t!!")
        combined = Magnipy(None)
        combined.__magnitude, combined.__ts = diff_of_functions(self.__magnitude, self.__ts, self.__D, 
                                                                other.__magnitude, other.__ts, other.__D, method=self.__method, 
                                                                exact=exact, t_cut=t_cut)
        combined.__n_ts = len(combined.__ts)
        return combined

    def __add(self, other, t_cut=None, exact=True):
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
        #combined i= self.__substract(other, t_cut=t_cut)
        #mag_diff = combined.get_magnitude_area(t_cut=t_cut, integration=integration, #normalise_by_cardinality=False, 
        #    absolute_area=absolute_area, scale=scale)
        return mag_difference
    
    def _scale_when_scattered(self):
        if (self.__t_scattered is None) | self.__recompute:
            self.__t_scattered = scale_when_scattered(self.__D)
        return self.__t_scattered
    
    def _scale_when_almost_scattered(self, q=None):
        if (self.__t_almost_scattered is None) | self.__recompute:
            self.__t_almost_scattered = scale_when_almost_scattered(self.__D, n=self.__n, q=q)
        return self.__t_almost_scattered