import numpy as np
import jax
import jax.numpy as jnp
import twinkle

from microjax.point_source import mag_point_source
from microjax.caustics.lightcurve import magnifications
from microjax.inverse_ray.lightcurve import mag_triple

from MulensModel.pointlens import _AbstractMagnification
from MulensModel.binarylens import _LimbDarkeningForMagnification, _FiniteSource, _BinaryLensPointSourceMagnification

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

class BinaryLensTwinkleGpuMagnification(_BinaryLensPointSourceMagnification,_LimbDarkeningForMagnification,
                                                _FiniteSource):
    """
    Equations for calculating point-source--binary-lens magnification using twinkle for point sources.
    Arguments :
        trajectory: :py:class:`~MulensModel.trajectory.Trajectory`
            Including trajectory.parameters =
            :py:class:`~MulensModel.modelparameters.ModelParameters`
    """

    def __init__(self, gamma=None, u_limb_darkening=None, device_num=0, N_stream=1, RelTol=1e-6, ** kwargs):
        super().__init__(**kwargs)
        self._set_LD_coeffs(u_limb_darkening=u_limb_darkening, gamma=gamma)
        self._set_and_check_rho()
        self._Nsrcs = 1
        self._device_num = self._parse_device_num(device_num)
        self._N_stream = self._parse_N_stream(N_stream)
        self._RelTol = self._parse_RelTol(RelTol)

        self._astrometry = False
        self._twinkle = twinkle.Twinkle(self._Nsrcs, self._device_num, self._N_stream, self._RelTol, self._astrometry)

    def get_magnification(self):
        """
        Calculate the magnification

        Parameters : None

        Returns :
            magnification: *np.ndarray*
                The magnification for each point in :py:attr:`~trajectory`.
        """
        if len(self._separations) == 1:
            if self._zip_kwargs is None:
                self._magnification = np.array(self._get_all_magnification(
                    self._source_x, self._source_y, self._separations))
            else:
                self._magnification = np.array(self._get_all_magnification(
                    self._source_x, self._source_y, self._separations, **self._zip_kwargs))
        else:
            zip_args = [self._source_x, self._source_y, self._separations]
            out = []
            if self._zip_kwargs is None:
                for (x, y, separation) in zip(*zip_args):
                    out.append(self._get_1_magnification(x, y, separation))
            else:
                zip_args += [self._zip_kwargs]
                for (x, y, separation, kwargs_) in zip(*zip_args):
                    out.append(self._get_1_magnification(
                        x, y, separation, **kwargs_))
            self._magnification = np.array(out)
        return self._magnification

    def _get_1_magnification(self, x, y, separation):
        """
        Calculate 1 magnification using VBM.
        """
        self._twinkle.set_params(separation, self._q, self._rho, x, y)
        if self._u_limb_darkening is None:
            self._twinkle.run()
        else:
            self._twinkle.runLD(self._u_limb_darkening)
        magnification = np.empty(1)
        self._twinkle.return_mag_to(magnification)
        return magnification[0]

    def _get_all_magnification(self, x, y, separation):
        Nsrcs = len(x)
        self._twinkle = twinkle.Twinkle(Nsrcs, self._device_num, self._N_stream, self._RelTol, self._astrometry)
        self._twinkle.set_params(separation, self._q, self._rho, x, y)
        if self._u_limb_darkening is None:
            self._twinkle.run()
        else:
            self._twinkle.runLD(self._u_limb_darkening)
        magnification = np.empty(Nsrcs)
        self._twinkle.return_mag_to(magnification)
        return magnification