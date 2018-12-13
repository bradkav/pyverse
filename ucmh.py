import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from cosmology import Omega_cdm_0, window_gaussian, m_hor, m_hor_cdm
from cosmology import r_hor_phys, rho_c, a_of_z, a_0, rho_tot


def pr_ucmh_collapse(sigma, delta_min=1e-3, delta_max=1/4):
    """Probability of a UCMH forming.

    Notes
    -----
    Assumes Gaussian fluctuations.

    Parameters
    ----------
    sigma : float
        Variance of density fluctuations.
    delta_min : float
        Minimum overdensity threshold for collapse.
    delta_max : float
        Maximum overdensity threshold for collapse. Larger overdensities
        presumably form PBHs.

    Returns
    -------
    float
    """
    # Factor of two since delta >= 0. TODO: double-check this!
    return (erfc(delta_min / (np.sqrt(2) * sigma)) -
            erfc(delta_max / (np.sqrt(2) * sigma)))


def m_ucmh_i_simple(a_i, Omega_cdm_bc=Omega_cdm_0, a_bc=a_0):
    """UCMH mass at formation, assuming purely radial infall.

    Notes
    -----
    * Should really be the DM mass inside turnaround radius rather than the
      horizon, but the calculation is quite approximate already.
    * Assumes Omega_cdm(a) = Omega_cdm_bc * (a_bc / a)**3: ie, not too much of
      the CDM is accumulated into UCMHs over the universe's history. TODO:
      check this!

    Parameters
    ----------
    a_i : float
        Scale factor at formation time.
    Omega_cdm_bc : float
        Initial condition on CDM abundance.
    a_bc : float
        Scale factor setting boundary condition: Omega_cdm(a_bc) = Omega_cdm_bc.

    Returns
    -------
    float
        UCMH mass at formation. This is the CDM mass in the horizon at a.
    """
    return m_hor_cdm(a_i, Omega_cdm_bc, a_bc)


def evolve_m_ucmh(m_ucmh_i, a_i, a=a_0, a_ucmh_end=a_of_z(30)):
    """Determines UCMH mass at a given scale factor.

    Notes
    -----
    * Assumes UCMH accretes CDM linearly in the scale factor.
    * Assumes UCMH is isolated. This breaks down if Omega_UCMH ~ Omega_CDM!

    Parameters
    ----------
    m_ucmh_i : float
        Initial UCMH mass (M_sun).
    a_i : float
        Scale factor at UCMH formation
    a : float
        Scale factor at which to compute UCMH mass. Must be larger than a_i.
    a_ucmh_end : float
        Scale factor at which to stop evolving UCMH.

    Returns
    -------
    m_ucmh : float
        UCMH mass at a.
    """
    assert np.all(a >= a_i)
    return m_ucmh_i * min([a, a_ucmh_end]) / a_i


def mass_fn_ucmh_simple(a_i, beta, a=a_0, Omega_cdm_bc=Omega_cdm_0, a_bc=a_0):
    """Computes the UCMH mass and differential abundance.

    Parameters
    ----------
    a_i : float
        Scale factor at formation.
    beta : float
        Probability of UCMH collapsing as a function of scale factor. This is
        user-supplied since it depends on how the power spectrum is modified.
    a : float
        Scale factor at which to compute UCMH mass and differential abundance.
    Omega_cdm_bc : float
        See `m_ucmh_i_simple`.
    a_bc : float
        See `m_ucmh_i_simple`.

    Returns
    -------
    m_ucmh_i, m_ucmh, df/dm
        Initial UCMH mass, present-day UCMH mass, corresponding differential
        density fraction normalized to total present DM abundance.
    """
    m_pbh_i = m_pbh_i_simple(a_i, gamma)
    m_pbh = evolve_m_pbh(m_pbh_i, a_i, a)
    m_hor_i = 4*np.pi/3 * r_hor_phys(a_i)**3 * rho_tot(a_i)
    dOmega_pbh_dm = m_pbh / m_hor_i * beta(a_i)
    # Redshift factors to account for volume scaling
    dOmega_pbh_dm *= (a_i / a)**3 * rho_tot(a_i) / rho_tot(a)
    return m_ucmh_i, m_ucmh, dOmega_ucmh_dm / Omega_cdm_0
