import numpy as np
import matplotlib.pylab as plt
import Cosmo

fig, axarr = plt.subplots(figsize=(10,8),ncols=2, nrows=2)

#Redshift time relation
#-----------
zlist = np.logspace(-1, 5, 1000)
tvals = np.asarray([Cosmo.t_univ(z) for z in zlist])

ax = axarr[0,0]

ax.loglog(zlist, tvals/1e9)

ax.set_xlabel(r"Redshift, $z$")
ax.set_ylabel(r"Time since Big Bang, $t$ [Gyr]")

#Now also plot z as a function of time
#-----------------------
ax = axarr[1,0]

ax.loglog(tvals/1e9,zlist)

ax.set_ylabel(r"Redshift, $z$")
ax.set_xlabel(r"Time since Big Bang, $t$ [Gyr]")


#Horizon size as a function of scale-factor
#-----------------------
avals = 1./(1.+zlist)

ax = axarr[0,1]

ax.loglog(avals,Cosmo.R_horizon(zlist))

ax.set_ylabel(r"Comoving Horizon size $R_H$ [Mpc]")
ax.set_xlabel(r"Scale factor $a$")


#Horizon mass as a function of redshift
#-----------------------

ax = axarr[1,1]

zlist2 = np.logspace(-1, 15, 1000)

ax.loglog(zlist2, Cosmo.M_horizon(zlist2))

ax.set_ylabel(r"Horizon mass $M_H$ [$M_\odot$]")
ax.set_xlabel(r"Redshift $z$")


#plt.subplots_adjust(hspace=0.2)


plt.figure()

plt.loglog(zlist2, Cosmo.rho_tot(zlist2))

plt.ylabel(r"$\rho_tot$ [$M_\odot/\mathrm{pc}^{-3}$]")
plt.xlabel(r"Redshift $z$")


plt.show()