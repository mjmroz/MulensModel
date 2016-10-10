import numpy as np

from MulensModel.utils import Utils
from MulensModel.fit import Fit
from MulensModel.mulensdata import MulensData
from MulensModel.model import Model

            
class Event(object):
    def __init__(self, datasets=None, model=None):
        self._model = model
        self._set_datasets(datasets)

    @property
    def datasets(self):
        """a list of MulensData instances that represent all event datasets"""
        return self._datasets

    @datasets.setter
    def datasets(self, new_value):
        self._set_datasets(new_value)

    def _set_datasets(self, new_value):
        """sets the value of self._datasets; can be called by __init__ or @datasets.setter; passes datasets to property self._model"""
        self._datasets = new_value
        if isinstance(self._model, Model):
            self._model.set_datasets(self._datasets)
    #    if isinstance(new_value, list):
    #        self._datasets = new_value
    #    else:
    #        raise ValueError('

    def get_chi2(self):
        """calculates chi^2 of current model"""
        self.fit = Fit(data=self.datasets, magnification=self.model.magnification) 
        self.fit.fit_fluxes()
        chi2 = []
        for dataset in self.datasets:
            diff = dataset._input_values - self.fit.get_input_format(data=dataset)
            select = np.logical_not(dataset.bad)
            chi2.append(sum((diff[select] / dataset._input_values_err[select])**2))
        self.chi2 = sum(chi2)
        return self.chi2

