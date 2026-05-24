import numpy as np

import jax
import jax.numpy as jnp

from microjax.point_source import mag_point_source
from microjax.caustics.lightcurve import magnifications
from microjax.inverse_ray.lightcurve import mag_triple

from MulensModel.pointlens import _AbstractMagnification
from MulensModel.binarylens import _LimbDarkeningForMagnification, _FiniteSource

jax.config.update("jax_enable_x64", True)  # stabilises the polynomial solver


class _TripleLensPointSourceMagnification(_AbstractMagnification):
    """
    Equations for calculating point-source--triple-lens magnification.
    This is a placeholder class to establish the basic methods and attributes
    and over-write methods from
    :py:class:`~MulensModel.pointlens.PointSourcePointLensMagnification`
    that do not apply to binary lenses.

    Arguments :
        trajectory: :py:class:`~MulensModel.trajectory.Trajectory`
            Including trajectory.parameters =
            :py:class:`~MulensModel.modelparameters.ModelParameters`
    """

    def __init__(self, **kwargs):
        super().__init__(trajectory=kwargs['trajectory'])
        # This speeds-up code for np.float input.
        # Can be manually changed to 'numpy'.

        self._source_x = self.trajectory.x
        self._source_y = self.trajectory.y
        self._geometry = self.trajectory.parameters.get_lens_geometry(self.trajectory.times)
        self._psi = self.trajectory.parameters.psi
        self._zip_kwargs = None

    def get_magnification(self):
        """
        Calculate the magnification

        Parameters : None

        Returns :
            magnification: *np.ndarray*
                The magnification for each point in :py:attr:`~trajectory`.
        """
        if len(self._geometry) == 1:
            if self._zip_kwargs is None:
                self._magnification = np.array(self._get_all_magnification(
                    self._source_x, self._source_y, self._geometry))
            else:
                self._magnification = np.array(self._get_all_magnification(
                    self._source_x, self._source_y, self._geometry, **self._zip_kwargs))
        else:
            zip_args = [self._source_x, self._source_y, self._geometry]

            out = []
            if self._zip_kwargs is None:
                for (x, y, geometry) in zip(*zip_args):
                    out.append(self._get_1_magnification(x, y, geometry))
            else:
                zip_args += [self._zip_kwargs]
                for (x, y, geometry, kwargs_) in zip(*zip_args):
                    out.append(self._get_1_magnification(
                        x, y, geometry, **kwargs_))
            self._magnification = np.array(out)
        return self._magnification

    def _get_w_points(self, x, y, parameters):
        """
        Calculate trajectory for microjaxx not shifted to CM, internally in microjaxx
        the source trajectory is shifted back to CM.
        """
        if isinstance(x, (int, float)):
            x = np.array([x])
            y = np.array([y])
        x_cm = 0.5 * parameters['s'] * (1. - parameters['q_21']) / (1. + parameters['q_21'])
        return jnp.array(x + x_cm + 1j * y, dtype=complex)

    def _get_lens_parameters(self, geometry):
        s = jnp.sqrt((geometry[0] - geometry[3])**2)
        parameters = {'s': s, 'q': jnp.float64(self.trajectory.parameters.q_21),
                      'q3': jnp.float64(self.trajectory.parameters.q_31),
                      # separation between center of masss for m1/m2 and m3
                      'r3': jnp.sqrt(geometry[6]**2. + geometry[7]**2.),
                      'psi': jnp.deg2rad(self._psi)}  # angle of 3rd lens axis in radians

        return parameters


class TripleLensPointSourceMicrojaxxMagnification(_TripleLensPointSourceMagnification):
    """
    Equations for calculating point-source--triple-lens magnification using microjaxx for point sources.
    Arguments :
        trajectory: :py:class:`~MulensModel.trajectory.Trajectory`
            Including trajectory.parameters =
            :py:class:`~MulensModel.modelparameters.ModelParameters`
    """
    def _get_1_magnification(self, x, y, geometry, **kwargs):
        """
        Calculate 1 magnification using VBM.
        """
        parameters = self._get_lens_parameters(geometry)
        return self._get_1_magnification_point_source(x, y, parameters)

    def _get_1_magnification_point_source(self, x, y, parameters):
        """
        Call VBM to get 1 magnification for point source.
        This function is also called by child classes.
        """
        w_points = self._get_w_points(x, y, parameters)
        return mag_point_source(w_points, n_lenses=3, **parameters)[0]

    def _get_all_magnification(self, x, y, geometry, **kwargs):
        """Calculate magnification for all points using microjaxx."""
        parameters = self._get_lens_parameters(geometry)
        w_points = self._get_w_points(x, y, parameters)
        return mag_point_source(w_points, n_lenses=3, **parameters)


class TripleLensMicrojaxxInverseRayMagnification(_TripleLensPointSourceMagnification, _LimbDarkeningForMagnification,
                                                 _FiniteSource):
    """
    Triple lens finite source magnification calculated using microjaxx library that implements
    inverse ray shooting method presented by
    Miyazaki, S., & Kawahara, H. 2025, ApJ, 994, 144, doi:10.3847/1538-4357/ae1005
    For coordinate system convention see
    :py:class:`BinaryLensQuadrupoleMagnification`

    Arguments :
        trajectory: :py:class:`~MulensModel.trajectory.Trajectory`
            Including trajectory.parameters =
            :py:class:`~MulensModel.modelparameters.ModelParameters`

        gamma: *float*
            Linear limb-darkening coefficient in gamma convention.

        u_limb_darkening: *float*
            Linear limb-darkening coefficient in u convention.
            Note that either *gamma* or *u_limb_darkening* can be
            set.  If neither of them is provided then limb
            darkening is ignored.

        microjaxx_kwargs: *dict*
        see for details microjaxx documentation:
        https://shotamiyazaki94.github.io/microjax/api/inverse_ray_lightcurve.html#microjax.inverse_ray.lightcurve.mag_triple
    """
    def __init__(self, gamma=None, u_limb_darkening=None, microjaxx_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self._set_LD_coeffs(u_limb_darkening=u_limb_darkening, gamma=gamma)
        self._set_and_check_rho()
        self._microjaxx_kwargs = microjaxx_kwargs

        if self._u_limb_darkening is None:
            self._u_limb_darkening = 0.0

    def _get_1_magnification(self, x, y, geometry, **kwargs):
        """
        Calculate 1 magnification using microjaxx inverse ray shooting method.
        """
        parameters = self._get_lens_parameters(geometry)
        w_points = self._get_w_points(x, y, parameters)
        parameters['u1'] = self._u_limb_darkening

        return mag_triple(w_points, self._rho, **parameters, **self._microjaxx_kwargs)[0]

    def _get_all_magnification(self, x, y, geometry, **kwargs):
        """Calculate magnification for all points using microjaxx inverse ray shooting method."""
        parameters = self._get_lens_parameters(geometry)
        w_points = self._get_w_points(x, y, parameters)
        parameters['u1'] = self._u_limb_darkening

        return mag_triple(w_points, self._rho, **parameters, **self._microjaxx_kwargs)


class TripleLensCausticsMagnification(_TripleLensPointSourceMagnification, _LimbDarkeningForMagnification,
                                      _FiniteSource):
    """Triple lens finite source magnification calculated using hybrid light-curve evaluation from
    the 'caustics' package implemented in microjaxx. See for details microjaxx documentation:
       https://shotamiyazaki94.github.io/microjax/api/caustics_lightcurve.html
    """

    def __init__(self, gamma=None, u_limb_darkening=None, microjaxx_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self._set_LD_coeffs(u_limb_darkening=u_limb_darkening, gamma=gamma)
        self._set_and_check_rho()
        self._microjaxx_kwargs = microjaxx_kwargs

        if self._u_limb_darkening is None:
            self._u_limb_darkening = 0.0
            self._microjaxx_kwargs['limb_darkening'] = 'false'
        else:
            self._microjaxx_kwargs['limb_darkening'] = 'true'

    def _get_1_magnification(self, x, y, geometry, **kwargs):
        """
        Calculate 1 magnification using microjaxx caustics lightcurve method.
        """
        parameters = self._get_lens_parameters(geometry)
        w_points = self._get_w_points(x, y, parameters)
        parameters['u1'] = self._u_limb_darkening
        parameters['n_lenses'] = 3
        return magnifications(w_points, self._rho, **parameters, **self._microjaxx_kwargs)[0]

    def _get_all_magnification(self, x, y, geometry, **kwargs):
        """
        Calculate magnification for all points using microjaxx caustics lightcurve method.
        """
        parameters = self._get_lens_parameters(geometry)
        w_points = self._get_w_points(x, y, parameters)
        parameters['u1'] = self._u_limb_darkening
        parameters['n_lenses'] = 3
        return magnifications(w_points, self._rho, **parameters, **self._microjaxx_kwargs)
