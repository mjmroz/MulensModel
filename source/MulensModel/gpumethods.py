import ctypes
from ctypes.util import find_library

import numpy as np
import twinkle

from MulensModel.binarylens import _LimbDarkeningForMagnification, _FiniteSource, _BinaryLensPointSourceMagnification


class BinaryLensTwinkleGpuMagnification(_BinaryLensPointSourceMagnification, _LimbDarkeningForMagnification,
                                        _FiniteSource):
    """
    Equations for calculating point-source--binary-lens magnification using twinkle for point sources.
    Arguments :
        trajectory: :py:class:`~MulensModel.trajectory.Trajectory`
            Including trajectory.parameters =
            :py:class:`~MulensModel.modelparameters.ModelParameters`
    """

    def __init__(self, gamma=None, u_limb_darkening=None, magnification_setup={}, device_num=0, N_stream=1, RelTol=1e-3, ** kwargs):
        super().__init__(**kwargs)
        self._set_LD_coeffs(u_limb_darkening=u_limb_darkening, gamma=gamma)
        self._set_and_check_rho()
        self._Nsrcs = len(self._trajectory.x)
        self._device_num = self._parse_device_num(device_num)
        self._N_stream = self._parse_N_stream(N_stream)
        self._RelTol = self._parse_accuracy(RelTol)

        self._astrometry = False
        
        key = f'twinkle_{self._Nsrcs:d}'
        if key not in magnification_setup:
            magnification_setup[f'twinkle_{self._Nsrcs:d}'] = twinkle.Twinkle(
                self._Nsrcs, self._device_num, self._N_stream, self._RelTol, self._astrometry)
            print(
                f"Initialized Twinkle with device_num={self._device_num}, N_stream={self._N_stream}, RelTol={self._RelTol}")
            print("If this message appears more than a few times, it means that the Twinkle object is being re-initialized." +
                  "This will slow down the calculations, and most likely is due to reinitializing the MulensData object.")

        self._twinkle = magnification_setup[f'twinkle_{self._Nsrcs:d}']
        self._magnification = np.empty(self._Nsrcs)

    def _parse_device_num(self, device_num):
        """
        Parse the device number and check if it is valid.
        """

        if not isinstance(device_num, int):
            raise TypeError("device_num must be an integer.")
        if device_num < 0:
            raise ValueError("device_num must be a non-negative integer.")
        return device_num


    def _parse_N_stream(self, N_stream):
        """
        Parse the number of streams and check if it is valid.
        """
        if not isinstance(N_stream, int):
            raise TypeError("N_stream must be an integer.")
        if N_stream < 0:
            raise ValueError("N_stream must be a non-negative integer.")
        return N_stream

    def _parse_accuracy(self, RelTol):
        """
        Parse the accuracy and check if it is valid.
        """
        if not isinstance(RelTol, (int, float)):
            raise TypeError("RelTol must be a number.")
        if RelTol <= 0:
            raise ValueError("RelTol must be a positive number.")
        return RelTol

    def get_magnification(self):
        """
        Calculate the magnification

        Parameters : None

        Returns :
            magnification: *np.ndarray*
                The magnification for each point in :py:attr:`~trajectory`.
        """
        if self._zip_kwargs is None:
            self._magnification = np.array(self._get_all_magnification(self._source_x, self._source_y,
                                                                       self._separations))
        else:
            self._magnification = np.array(self._get_all_magnification(
                self._source_x, self._source_y, self._separations, **self._zip_kwargs))

        return self._magnification

    def _get_all_magnification(self, x, y, separation):

        self._twinkle.set_params(np.array(separation, dtype=np.float64), np.float64(self._q), np.float64(self._rho),
                                 np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
        if self._u_limb_darkening is None:
            self._twinkle.run()
        else:
            self._twinkle.runLD(self._u_limb_darkening)
        self._twinkle.return_mag_to(self._magnification)
        return self._magnification
