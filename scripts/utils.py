# Matplot ticks
import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 1.
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 1.
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits# If missing, do "pip install astropy"

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)    



def gauss_PSF(num_rows, num_cols, x, y, FWHM):
    """
    Given num_rows x num_cols of an image, generate PSF
    at location x, y.
    """
    sigma = FWHM / 2.354
    xv = np.arange(0.5, num_rows)
    yv = np.arange(0.5, num_cols)
    yv, xv = np.meshgrid(xv, yv) # In my convention xv corresponds to rows and yv to columns
    PSF = np.exp(-(np.square(xv-x) + np.square(yv-y))/(2*sigma**2))/(np.pi * 2 * sigma**2)

    return PSF

def poisson_realization(D0):
    """
    Given a truth image D0, make a Poisson realization of it.
    """
    D = np.zeros_like(D0)
    for i in range(D0.shape[0]):
        for j in range(D0.shape[1]):
            D[i, j] = np.random.poisson(lam=D0[i, j], size=1)
    return D
