#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:06:47 2024

@author: mmroz
"""

import sys
import yaml
import numpy as np
from astropy import units as u

from ulens_model_fit import UlensModelFit



class UlensModelFitWithGP(UlensModelFit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
   
    
   
    def _set_default_parameters(self):
        """
        Extend the set of available parameters
        """
        super()._set_default_parameters()
        # All physical parameters that we may be interested in:
        self._other_parameters = ['', 'D_l', 'mu_s_N', 'mu_s_E', 'mu_rel']
        self._latex_conversion_other = dict(
            M_l='M_{\\rm lens}',
            D_l='D_{\\rm lens}',
            mu_s_N='\\mu_{{\\rm s}_N}',
            mu_s_E='\\mu_{{\\rm s}_E}',
            mu_rel="\\mu_{\\rm rel}")
