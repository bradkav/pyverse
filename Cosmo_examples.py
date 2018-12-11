import numpy as np
import matplotlib.pylab as plt
import Cosmo

#Redshift time relation
#-----------
zlist = np.logspace(-1, 5, 1000)
tvals = np.asarray([Cosmo.t_univ(z) for z in zlist])

plt.figure()
plt.plot(zlist, tvals/1e9)

plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"Time since Big Bang, $t$ [Gyr]")

plt.ylim(0, 15)


#Now also plot z as a function of time
#-----------------------
plt.figure()
plt.plot(tvals/1e9,zlist)

plt.ylabel(r"Redshift, $z$")
plt.xlabel(r"Time since Big Bang, $t$ [Gyr]")


#Horizon size as a function of scale-factor
avals = 1./(1.+zlist)

plt.figure()

plt.loglog(avals,Cosmo.R_horizon(zlist))

plt.ylabel(r"Horizon size $R_H$ [Mpc]")
plt.xlabel(r"Scale factor $a$")



plt.show()