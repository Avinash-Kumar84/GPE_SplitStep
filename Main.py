__author__ = 'avinash'

import numpy as np
from numpy import sqrt, pi, exp, linspace, sin, square, cos, fft, arange, absolute, angle
import matplotlib.pyplot as pl
import GS as gs
import Evolve as evolve
import pyfftw

pyfftw.interfaces.cache.enable()

Nx = 512                 # Number of discrete points
Ny = 512
Lx = 10.0                # Size in harmonic oscillator units
Ly = 10.0
eps=1.0                  # Trap anisotropy
lam=0.05                 # Trap anharmonicity

gN = 1000
mu0 = sqrt(gN/2*pi)
Omega=1.2

xi0 = 1/sqrt(2*mu0)
print("Thomas-Fermi chemical potential: ",+ mu0)
print('Minimum healing length: ',(xi0))

def Vinit(x, y):
   rsq=x**2+(eps*y)**2
   return 1.0*(rsq+lam*rsq**2-rsq*Omega**2)                   # Potential in unis or harmonic frequency along x

# The Thomas-Fermi distribution
def nTF(x, y):
   mask=mu0-Vinit(x,y)>0.0
   return ((mu0 - Vinit(x, y)) / gN) * mask.astype(np.int)
   #return ((mu0-Vinit(x,y))/gN)*(abs(Vinit(x,y))<sqrt(2*mu0)).astype(np.int)

xgrid= -Lx+2*(Lx/Nx)*np.asfortranarray(arange(Nx))
ygrid=-Ly+2*(Ly/Ny)*np.asfortranarray(arange(Ny))
xmesh,ymesh=np.meshgrid(xgrid,ygrid)

###########################
# Ground state computation#
###########################

g_s=gs.GroundState(Vinit(xmesh,ymesh), nTF(xmesh,ymesh),gN, Omega, xmesh, ymesh)
psi0=g_s.psifinal
print('Number of steps=',g_s.nsteps)
print('Chemical potential =%.3f' % g_s.mufinal)
print('Healing length =%.3f' % (1/sqrt(2*g_s.mufinal)))


# Define phase to add on ground state wavefunction
def helix(th):
    dphi = 2 * pi * 1.0
    return dphi * (th / (2 * pi))

#########################
#  Real time evolution  #
#########################

psiI = np.copy(psi0)              # Initial phase imprinted can be added here
dt = 1e-3                         # Time step in units of trapping frequency along x
nsteps = 500                      # Total number of dt steps to evolve
measureStep = 10                  # Time step to add operator to compute properties (not used in this version)
nframes = 100                     # Time frames to fetch throughout evolution (not used in this version)
t_e=evolve.TimeEvolution(dt,nsteps,measureStep,nframes,psiI, mu0,gN,eps, xmesh, ymesh)
psit=t_e.psifinal

##########
#Plotting#
##########

fig1 = pl.figure()
ax1 = fig1.add_subplot(121)
ax1.imshow(absolute(psi0), vmin=absolute(psi0).min(), vmax=absolute(psi0).max(), origin='lower',
           extent=[xmesh.min(), xmesh.max(), ymesh.min(), ymesh.max()],cmap=pl.cm.YlGnBu_r)
ax2 = fig1.add_subplot(122)
ax2.imshow(angle(psi0), vmin=angle(psi0).min(), vmax=angle(psi0).max(), origin='lower',
           extent=[xmesh.min(), xmesh.max(), ymesh.min(), ymesh.max()], cmap=pl.cm.YlGnBu_r)

fig2 = pl.figure()
ax1 = fig2.add_subplot(121)
ax1.imshow(absolute(psit), vmin=absolute(psit).min(), vmax=absolute(psit).max(), origin='lower',
           extent=[xmesh.min(), xmesh.max(), ymesh.min(), ymesh.max()],cmap=pl.cm.YlGnBu_r)
ax2 = fig2.add_subplot(122)
ax2.imshow(angle(psit), vmin=angle(psit).min(), vmax=angle(psit).max(), origin='lower',
           extent=[xmesh.min(), xmesh.max(), ymesh.min(), ymesh.max()], cmap=pl.cm.YlGnBu_r)
pl.show()