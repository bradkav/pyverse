import numpy as np
from scipy.integrate import quad

"""
*Redshift-time relation*

The time since the Big Bang is given by:

$$t(z) = \int_z^\infty \frac{\mathrm{d}z'}{(1+z') H(z')}\,,$$

where

$$H(z') = \sqrt{\Omega_\Lambda + \Omega_m (1+z)^3 + \Omega_r (1+z)^4}\,,$$

in a flat universe. We use the Planck 2015 values from [arXiv:1502.01589](https://arxiv.org/abs/1502.01589), noting that there is some tension between different measurements of $H_0$:

- $H_0 = 67.8 \,\,\mathrm{km/s}\,\mathrm{Mpc}^{-1}$
- $\Omega_\Lambda = 0.692$
- $\Omega_m = 0.308$
- $\Omega_r = 9.24 \times 10^{-5}$
"""


#Get H in units of 1/yr
#(km/Mpc) = 3.24e-20
H0 = 67.8 #(km/s)/Mpc
H0_peryr = H0*(3.24e-20)*(60*60*24*365) #1/yr

Omega_L = 0.692
Omega_m = 0.308
Omega_r = 9.24e-5

c = 3e8 #m/s


def Hubble(z):
    """
    Return Hubble parameter in (km/s)/Mpc.
    
    Parameters
    ----------
    * `z`  [float]:
        Redshift
    
    Returns
    -------
    * `H0` [float]:
        Hubble parameter in (km/s)/Mpc
    """
    
    return H0*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z**4))
 

def Hubble_peryr(z):
    """
    Return Hubble parameter in 1/yr.
    
    Parameters
    ----------
    * `z`  [float]:
        Redshift
    
    Returns
    -------
    * `H0` [float]:
        Hubble parameter in 1/yr
    """
    
    return Hubble(z)*H0_peryr/H0
    

def t_univ(z):
    """
    Calculate time from Big Bang until redshift z.
    
    Parameters
    ----------
    * `z` [float]:
        Redshift
    
    Returns
    -------
    * `t` [float]:
        Time from Big Bang (in years)
    
    """
    
    integ = lambda x: 1.0/((1+x)*Hubble_peryr(x))
    return quad(integ, z, np.inf)[0]
    
#Calculate horizon size as a function of redshift (in Mpc)
def R_horizon(z):
    """
    Calculate horizon size.
    
    Parameters
    ----------
    * `z` [float]:
        Redshift
    
    Returns
    -------
    * `R_H` [float]:
        Horizon size (in Mpc)
    
    """
    
    a = 1./(1.+z)
    return (c*1e-3)/(a*Hubble(z))