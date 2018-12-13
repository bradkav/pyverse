import numpy as np
from scipy.integrate import quad

# Constants
H0 = 2.195e-18 * (365 * 24 * 60**2)  # yr^-1
Gn = 4.52e-30 * (365 * 24 * 60**2)  # pc^3 / (M_sun yr^2)
rho_c = 1.26e-7  # critical density, M_sun / pc^3
z_eq = 3e3
speed_of_light = 3.064e-1  # pc/yr
a_0 = 1.

# Present-day densities
Omega_m_0 = 0.3089
Omega_cdm_0 = 0.2589  # DM density parameter
Omega_rad_0 = 9.24e-5
Omega_Lambda_0 = 0.6911


def Omega_cdm(a, Omega_cdm_i=Omega_cdm_0, a_i=a_0):
    """DM density parameter.

    Parameters
    ----------
    Omega_cdm_i
        Initial DM density parameter at a = a_i.
    a_i
        Scale factor at which boundary condition is defined.
    """
    return Omega_cdm_i * (a_i / a)**2


def rho_cdm(a, Omega_cdm_i=Omega_cdm_0, a_i=a_0):
    """Total density

    Returns
    -------
        Total density of universe in M_sun / pc^3.
    """
    return Omega_cdm(a, Omega_cdm_i, a_i) * rho_c


def Omega_m(a):
    return Omega_m_0/a**3


def Omega_rad(a):
    return Omega_rad_0/a**4


def Omega_tot(a):
    """ Total density parameter.
    """
    return Omega_rad(a) + Omega_m(a) + Omega_Lambda_0


def rho_tot(a):
    """Total density

    Returns
    -------
        Total density of universe in M_sun / pc^3.
    """
    return Omega_tot(a) * rho_c


def hubble(a):
    """
    H = H0 sqrt(Omega_tot(a))
    """
    return H0 * np.sqrt(Omega_tot(a))


def t_of_a(a):
    """
    Notes
    -----
    t = int_0^{a(t)} da (dt/da) = int_0^{a(t)} da / H
    """
    return quad(lambda a: 1 / hubble(a), 0, a)[0]


def a_of_z(z):
    return 1 / (1. + z)


def t_of_z(z):
    t_of_a(a_of_z(z))


def window_gaussian(k, r):
    """Fourier transform of Gaussian window function.
    """
    return np.exp(-0.5 * (k*r)**2) / (2*np.pi)


def r_hor_phys(a):
    """Physical horizon size.

    Notes
    -----
    r_hor_phys = c / H. Agrees with [wikipedia](https://en.wikipedia.org/wiki/Cosmological_horizon#Hubble_horizon).
    """
    return speed_of_light / hubble(a)


def m_hor(a):
    """Horizon mass.

    Notes
    -----
    Assumes the density fluctuation is much less than 1.
    """
    return 4*np.pi/3 * rho_tot(a) * r_hor_phys(a)**3


def m_hor_cdm(a, Omega_cdm_i=Omega_cdm_0, a_i=a_0):
    """Horizon mass.

    Notes
    -----
    Assumes the density fluctuation is much less than 1.
    """
    return 4*np.pi/3 * rho_cdm(a, Omega_cdm_i, a_i) * r_hor_phys(a)**3


def sigma_spike(r, As, ks, wf=window_gaussian):
    """Computes the density field's variance for a spike modification of the
    power spectrum.

    Notes
    -----
    Assumes P_delta(k) = (4 k^2 / 9 a^2 H^2)^2 P_xi(k), where
    P_xi(k) = As ks delta(k - ks)
    """
    return 4/9 * (ks * r)**2 * np.sqrt(As) * wf(ks, r)


def pr_overdensity(delta, sigma):
    """Probability of an overdensity of size delta, assuming Gaussian
    fluctuations.
    """
    # Factor of two since delta >= 0
    return 2 * np.exp(-0.5 * (delta / sigma)**2) / (np.sqrt(2*np.pi) * sigma)
