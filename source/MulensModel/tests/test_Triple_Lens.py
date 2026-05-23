from numpy.testing import assert_almost_equal
import numpy as np
import matplotlib.pyplot as plt
from MulensModel.model import Model
import VBMicrolensing


plot = True


def test_VBM_vs_microjax():
    """
    Test MulensModel.Model() for triple lens vs VBMicrolensing. The test is based on the example from VBMicrolensing:
    https://github.com/valboz/VBMicrolensing/blob/main/examples/python_examples/Triple_lens.ipynb
    """
    VBM = VBMicrolensing.VBMicrolensing()
    VBM.RelTol = 1e-04
    VBM.Tol = 1e-04

    num_points = 1000
    tmin = -50
    tmax = 50

    parameters = {'s_21': 0.765, 'q_21': 0.00066, 'u_0': 0.006, 'alpha': np.degrees(3.212), 'rho': 0.0567,
                  't_E': 50.13, 't_0': 0, 's_31': 1.5, 'q_31': 0.000001, 'psi': np.degrees(-1.5)}
    t = np.linspace(parameters['t_0'] + tmin, parameters['t_0'] + tmax, num_points)

    model_VBM = Model(parameters=parameters)
    model_VBM.set_magnification_methods([float(min(t)), 'vbm_multiple', float(max(t))])
    model_VBM.default_magnification_method = 'vbm_multiple'
    magtriple_VBM = model_VBM.get_magnification(t)
    model_VBM.update_caustics()
    caustics_VBM = model_VBM.caustics
    x_VBM, y_VBM = caustics_VBM.get_caustics()
    x_critical_VBM, y_critical_VBM = caustics_VBM._critical_curve.x, caustics_VBM._critical_curve.y

    model_microjax = Model(parameters=parameters)
    model_microjax.set_magnification_methods([float(min(t)), 'microjax', float(max(t))])
    model_microjax.default_magnification_method = 'microjax'
    magtriple_microjax = model_microjax.get_magnification(t)
    # model_microjax.update_caustics()
    # caustics_microjax = model_microjax.caustics
    # x_microjax, y_microjax = caustics_microjax.get_caustics()
    # x_critical_microjax, y_critical_microjax = caustics_microjax._critical_curve.x, caustics_microjax._critical_curve.y



    if plot:
        plt.scatter(x_VBM, y_VBM, color='r', label='VBM caustics', s=4, alpha=0.1, marker='x')
        # plt.scatter(x_microjax, y_microjax, color='b', label='Microjax caustics', s=1, alpha=0.1, marker='o')
        plt.scatter(
            x_critical_VBM, y_critical_VBM, color='r', label='VBM critical curve', s=4, alpha=0.1, marker='x')
        # plt.scatter(x_critical_microjax, y_critical_microjax, color='b', label='Microjax critical curve', s=1, alpha=0.1, marker='o')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Caustics and critical curves for triple lens')
        plt.show()

    assert_almost_equal(magtriple_microjax[0], magtriple_VBM[0], decimal=3, err_msg='Magnification')
    # assert_almost_equal(x_microjax, x_VBM, decimal=3, err_msg='Caustics x')
    # assert_almost_equal(y_microjax, y_VBM, decimal=3, err_msg='Caustics y')
    # assert_almost_equal(x_critical_microjax, x_critical_VBM, decimal=3, err_msg='Critical x')
    # assert_almost_equal(y_critical_microjax, y_critical_VBM, decimal=3, err_msg='Critical y')

    return 'git'


if __name__ == '__main__':
    plot = True
    test_VBM_vs_microjax()
