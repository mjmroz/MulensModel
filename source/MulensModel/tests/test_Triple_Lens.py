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
def test_VBM_vs_microjax():
    """
    Test MulensModel.Model() for triple lens vs VBMicrolensing. The test is based on the example from VBMicrolensing:
    https://github.com/valboz/VBMicrolensing/blob/main/examples/python_examples/Triple_lens.ipynb
    """
    
    
    VBM = VBMicrolensing.VBMicrolensing()
    VBM.RelTol = 1e-04
    VBM.Tol = 1e-04

    num_points = 200
    tmin = -50
    tmax = 50

    parameters = {'s_21': 2, 'q_21': 1, 'u_0': 0.006, 'alpha': np.degrees(3.212), 'rho': 0.0567,
                  't_E': 50.13, 't_0': 0, 's_31': 1, 'q_31': 1, 'psi': np.degrees(np.pi/2)}
    t = np.linspace(parameters['t_0'] + tmin, parameters['t_0'] + tmax, num_points)


    model_VBM = Model(parameters=parameters)
    #model_VBM.set_magnification_methods([float(min(t)), 'vbm_multiple', float(max(t))])
    model_VBM.default_magnification_method = 'vbm_multiple'
    time_start = time.time()
    magtriple_VBM = model_VBM.get_magnification(t)
    time_VBM = time.time() - time_start
    model_VBM.update_caustics()
    caustics_VBM = model_VBM.caustics
    x_VBM, y_VBM = caustics_VBM.get_caustics()
    x_critical_VBM, y_critical_VBM = caustics_VBM._critical_curve.x, caustics_VBM._critical_curve.y

    print(jax.devices())
    model_microjax = Model(parameters=parameters)
    #model_microjax.set_magnification_methods([float(min(t)), 'microjax', float(max(t))])
    model_microjax.default_magnification_method = 'microjax'
    
    geometry = model_microjax.parameters.get_lens_geometry()[0]
    s = jnp.sqrt((geometry[0] - geometry[3])**2)
    parameters = {'s': s, 'q': jnp.float64(model_microjax.parameters.q_21),
                    'q3': jnp.float64(model_microjax.parameters.q_31),
                    # separation between center of masss for m1/m2 and m3
                    'r3': jnp.sqrt(geometry[6]**2. + geometry[7]**2.),
                     # microjax expects the angle in radians from the x-axis to the line connecting the center of mass of m1/m2 and m3
                    'psi': jnp.arctan2(geometry[7], geometry[6])}
    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=3, npts=100, **parameters)

    
    if plot:
        plt.figure()
        model_VBM.plot_caustics(color='r', label='VBM caustics')
        for cc in caustic_curves:
            plt.plot(cc.real, cc.imag, color='b', lw=0.7,)
        plt.plot(caustic_curves[0].real, caustic_curves[0].imag, color='b', lw=0.7,label='Microjax caustics')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Caustics and critical curves for triple lens')
        plt.savefig('triple_lens_caustics_microjax.png', dpi=300)
        plt.show()
    
    time_start = time.time()
    magtriple_microjax = model_microjax.get_magnification(t)
    time_microjax = time.time() - time_start
    
    

    
    
    model_microjax.update_caustics()
    caustics_microjax = model_microjax.caustics
    x_microjax, y_microjax = caustics_microjax.get_caustics()
    x_critical_microjax, y_critical_microjax = caustics_microjax._critical_curve.x, caustics_microjax._critical_curve.y

    print(f"VBM time: {time_VBM:.3f} s, Microjax time: {time_microjax:.3f} s")

    if plot:
        plt.figure()
        plt.plot(t, magtriple_microjax, color='b', label='Microjax magnification')
        plt.plot(t, magtriple_VBM, color='r', label='VBM magnification', alpha=0.5)
        plt.xlabel('Time - t_0')
        plt.ylabel('Magnification')
        plt.legend()
        plt.savefig('triple_lens_magnification_microjax.png', dpi=300)
        

# The line `assert_almost_equal(magtriple_microjax[0], magtriple_VBM[0], decimal=3,
# err_msg='Magnification')` is performing an assertion check in the Python test function.
    assert_almost_equal(magtriple_microjax[0], magtriple_VBM[0], decimal=3, err_msg='Magnification')
    # assert_almost_equal(x_microjax, x_VBM, decimal=3, err_msg='Caustics x')
    # assert_almost_equal(y_microjax, y_VBM, decimal=3, err_msg='Caustics y')
    # assert_almost_equal(x_critical_microjax, x_critical_VBM, decimal=3, err_msg='Critical x')
    # assert_almost_equal(y_critical_microjax, y_critical_VBM, decimal=3, err_msg='Critical y')

    return 'git'


if __name__ == '__main__':
    
    
    plot = True
    test_VBM_vs_microjax()
