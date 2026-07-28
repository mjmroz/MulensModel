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

    def __init__(self, gamma=None, u_limb_darkening=None, device_num=0, N_stream=1, RelTol=1e-3, ** kwargs):
        super().__init__(**kwargs)
        self._set_LD_coeffs(u_limb_darkening=u_limb_darkening, gamma=gamma)
        self._set_and_check_rho()
        self._Nsrcs = 1
        self._device_num = self._parse_device_num(device_num)
        self._N_stream = self._parse_N_stream(N_stream)
        self._RelTol = self._parse_accuracy(RelTol)

        self._astrometry = False
        self._twinkle = twinkle.Twinkle(self._Nsrcs, self._device_num, self._N_stream, self._RelTol, self._astrometry)

    def _parse_device_num(self, device_num):
        """
        Parse the device number and check if it is valid.
        """

        if not isinstance(device_num, int):
            raise TypeError("device_num must be an integer.")
        if device_num < 0:
            raise ValueError("device_num must be a non-negative integer.")
        device_count = self._get_gpu_device_count()
        if device_num >= device_count:
            raise ValueError(
                f"device_num must reference an available GPU device; got {device_num}, "
                f"but only {device_count} device(s) are available."
            )
        return device_num

    @staticmethod
    def _get_gpu_device_count():
        """
        Return the number of CUDA GPU devices available on this machine.
        """
        for library_name in (find_library("cudart"), "libcudart.so", "libcudart.so.12", "libcudart.so.11"):
            if not library_name:
                continue
            try:
                cudart = ctypes.CDLL(library_name)
            except OSError:
                continue

            device_count = ctypes.c_int()
            cuda_error = cudart.cudaGetDeviceCount(ctypes.byref(device_count))
            if cuda_error == 0:
                return device_count.value

        raise RuntimeError("CUDA runtime is not available, so GPU device_num cannot be validated.")

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
        Nsrcs = len(x)
        self._twinkle = twinkle.Twinkle(Nsrcs, self._device_num, self._N_stream, self._RelTol, self._astrometry)
        self._twinkle.set_params(np.array(separation, dtype=np.float64), np.float64(self._q), np.float64(self._rho),
                                 np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
        if self._u_limb_darkening is None:
            self._twinkle.run()
        else:
            self._twinkle.runLD(self._u_limb_darkening)
        magnification = np.empty(Nsrcs)
        self._twinkle.return_mag_to(magnification)
        return magnification
