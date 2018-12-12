import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from cosmology import Omega_cdm_0, window_gaussian, m_hor, m_hor_cdm
from cosmology import r_hor_phys, rho_c, a_0


"""
Notes
-----
* If PBHs make up a significant part of Omega_cdm_0, H(a) and Omega_tot(a) are
  modified from their standard values at early times! Though actually this
  shouldn't matter much since PBHs form during radiation domination.
    > Need a check that Omega_pbh << Omega_rad before a_eq and the CDM
      abundance is close to the standard value after a_eq.
"""


def pr_pbh_collapse(sigma, delta_c=1/3):
    """Probability of a PBH forming.

    Notes
    -----
    Assumes Gaussian fluctuations.

    Parameters
    ----------
    sigma : float
        Variance of density fluctuations.
    delta_c : float
        Overdensity threshold for collapse.
    """
    # Factor of two since delta >= 0. TODO: double-check this!
    return erfc(delta_c / (np.sqrt(2) * sigma))


def m_pbh_i_simple(a_i, gamma=(1./3)**1.5):
    """PBH mass at formation using Carr's original model.

    Parameters
    ----------
    a_i : float
        Scale factor at formation time.
    gamma : float
        Fraction of the horizon mass that goes into the resulting PBH.

    Returns
    -------
    float
        PBH mass.
    """
    return gamma * m_hor(a_i)


def evolve_m_pbh(m_pbh_i, a_i, a=a_0):
    """Determines PBH mass at a given scale factor.

    Parameters
    ----------
    m_pbh_i : float
        Initial PBH mass (M_sun).
    a_i : float
        Scale factor at PBH formation
    a : float
        Scale factor at which to compute PBH mass. Must be larger than a_i.

    Returns
    -------
    m_pbh : float
        PBH mass at a.
    """
    assert np.all(a >= a_i)
    return m_pbh_i


def mass_fn_pbh_simple(a_i, beta, a=a_0, gamma=1/3**1.5):
    """Computes the PBH mass and differential abundance.

    Parameters
    ----------
    a_i : float
        Scale factor at formation.
    beta : float
        Probability of PBH collapsing as a function of scale factor. This is
        user-supplied since it depends on how the power spectrum is modified.
    gamma : float
        PBH formation efficiency: see `m_pbh_i_simple` documentation.

    Returns
    -------
    m_pbh_i, m_pbh, df/dm
        Initial PBH mass, present-day PBH mass, corresponding differential
        density fraction normalized to total present DM abundance.
    """
    m_pbh_i = m_pbh_i_simple(a_i, gamma)
    m_pbh = evolve_m_pbh(m_pbh_i, a_i, a)
    # Compute present-day differential abundance by redshifting the initial
    # abundance.
    m_c = 4*np.pi/3 * r_hor_phys(a_i)**3 * rho_c
    dOmega_pbh_dm = m_pbh / m_c * beta(a_i) * (a_i / a)**3
    return m_pbh_i, m_pbh, dOmega_pbh_dm / Omega_cdm_0
