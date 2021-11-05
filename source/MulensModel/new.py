import numpy as np
import matplotlib.pyplot as plt

def get_eccentric_anomaly(m, ecc):
    """
    m - mean anomaly (numpy array) - has to be in (-np.pi, np.pi) range
    ecc - eccentricity (float)
    """
    lim = 1.e-8  # We're using Halley's method which gives cubic convergence,
    # hence any next correction would be around floating point precision.
    if ecc == 0.:
        return m
    if ecc > 0.8:  # This limit was found by trial and error.
        e = np.pi * np.sign(m)
    else:
        e = m
    e = e + ecc * np.sin(e)
    
    mask = [True]
    while np.sum(mask) > 0:
        f_bis = ecc * np.sin(e)  # This is second derivative of the function.
        f = e - m - f_bis
        f_prime = 1. - ecc * np.cos(e)
        corr = 2 * f * f_prime / (2 * f_prime**2 - f * f_bis)
        e -= corr
        mask = (np.abs(corr) > lim)
    return e

def get_x_y(m, ecc):
    """
    results are normalized to semi-major axis
    """
    m[m > np.pi] -= 2 * np.pi
    E = get_eccentric_anomaly(m, ecc)
    x = np.cos(E) - ecc
    y = np.sin(E) * (1. - ecc**2)**0.5
    return (x, y)


if __name__ == '__main__':
    n_dat = 10000
    ecc = 0.9

    # m is mean anomaly
    m = 2. * np.pi * (np.arange(n_dat)/n_dat - 0.5)
    (x, y) = get_x_y(m, ecc)
    plt.scatter(x, y, c=m)
    plt.show()

