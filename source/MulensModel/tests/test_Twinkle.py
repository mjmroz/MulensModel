import time

from numpy.testing import assert_almost_equal
import numpy as np
import matplotlib.pyplot as plt
from MulensModel.model import Model
import VBMicrolensing
from microjax.point_source import mag_point_source, critical_and_caustic_curves


plot = True


import jax
import jax.numpy as jnp

print(jax.devices())
print(jax.default_backend())
def test_VBM_vs_Twinkle():
    """
    Test MulensModel.Model() for Twinkle  binary lens magnification vs VBMicrolensing.
    """
    

    num_points = 200
    tmin = -50
    tmax = 50

    parameters = {'s': 2, 'q': 1, 'u_0': 0.006, 'alpha': np.degrees(3.212), 'rho': 0.0567,
                  't_E': 50.13, 't_0': 0,}
    t = np.linspace(parameters['t_0'] + tmin, parameters['t_0'] + tmax, num_points)


    model_VBM = Model(parameters=parameters)
    #model_VBM.set_magnification_methods([float(min(t)), 'vbm_multiple', float(max(t))])
    model_VBM.default_magnification_method = 'vbm'
    time_start = time.time()
    mag_VBM = model_VBM.get_magnification(t)
    time_VBM = time.time() - time_start
    model_VBM.update_caustics()
    caustics_VBM = model_VBM.caustics
    x_VBM, y_VBM = caustics_VBM.get_caustics()
    model_twinkle = Model(parameters=parameters)
    model_twinkle.default_magnification_method = 'Twinkle'

    # Plot VBM caustics and Twinkle caustics
    if plot:
        plt.figure()
        model_VBM.plot_caustics(color='r', label='VBM caustics')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Caustics and critical curves for binary lens')
        plt.savefig('binary_lens_caustics_twinkle.png', dpi=300)
        plt.show()

    time_start = time.time()
    mag_twinkle = model_twinkle.get_magnification(t)
    time_twinkle = time.time() - time_start

    model_twinkle.update_caustics()
    caustics_twinkle = model_twinkle.caustics
    x_twinkle, y_twinkle = caustics_twinkle.get_caustics()
    x_critical_twinkle, y_critical_twinkle = caustics_twinkle._critical_curve.x, caustics_twinkle._critical_curve.y

    print(f"VBM time: {time_VBM:.3f} s, Twinkle time: {time_twinkle:.3f} s")

    if plot:
        plt.figure()
        plt.plot(t, mag_twinkle, color='b', label='Twinkle magnification')
        plt.plot(t, mag_VBM, color='r', label='VBM magnification', alpha=0.5)
        plt.xlabel('Time - t_0')
        plt.ylabel('Magnification')
        plt.legend()
        plt.savefig('binary_lens_magnification_twinkle.png', dpi=300)

    # Compare magnification at first epoch
    assert_almost_equal(mag_twinkle[0], mag_VBM[0], decimal=3, err_msg='Magnification')
    # assert_almost_equal(y_critical_microjax, y_critical_VBM, decimal=3, err_msg='Critical y')

    return 'git'


if __name__ == '__main__':
    plot = True
    test_VBM_vs_Twinkle()