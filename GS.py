__author__ = 'avinash'

import numpy as np
from numpy import sqrt, pi, exp, linspace, sin, square, cos, fft, linspace, append, arange, log,trapz,arctan2, absolute, angle
from math import floor, ceil
import matplotlib.pyplot as pl
import time
import pyfftw
import multiprocessing
#from numba import jit

pyfftw.interfaces.cache.enable()


class GroundState(object):
    def __init__(self, V, nTF,gN,Omega, xmesh=None, ymesh=None):
        psi=sqrt(nTF)

        Ny, Nx = xmesh.shape

        nthread = multiprocessing.cpu_count()

        dx = xmesh[1,2] - xmesh[1,1]
        dy = ymesh[2,1] - ymesh[1,1]
        print('dx =%.3f' % dx)
        kxgrid = 1. / (Nx * dx) * append(arange(0,floor((Nx - 1) / 2.)+1), arange(-ceil((Nx - 1) / 2.),0))
        kygrid = 1. / (Ny * dy) * append(arange(0,floor((Ny - 1) / 2.)+1), arange(-ceil((Ny - 1) / 2.),0))
        kxmesh, kymesh = np.meshgrid(kxgrid, kygrid)
        dt = -1j * 0.01
        dmu = 1.0
        mu = 0.
        n = 0
        #Omega=0.1

        # Evolution in imaginary time
        start_time = time.time()
        while dmu > 1e-3:
            muBkp = mu
            Ur = exp(-1j * dt * V)
            Uk = exp(-1j * dt * 0.5 * (kxmesh**2 + kymesh**2) )
            dmu = 1.0
            mu = 0.

            while dmu > 1e-5:
                muOld = mu
                psi = Ur* exp(-1j * dt * gN * np.absolute(psi)** 2)*psi

                psi = pyfftw.interfaces.numpy_fft.fft2(psi, threads=nthread)

                psi = Uk* psi
                psi = pyfftw.interfaces.numpy_fft.ifft2(psi, threads=nthread)


                npsi = trapz( trapz(abs(psi)**2, xmesh[0,:], axis=1),ymesh[:,0],axis=0)

                psi = psi / sqrt(npsi)
                mu = -log(npsi) / (2 * abs(dt))
                dmu = abs(mu - muOld) / abs(mu)
                n = n + 1


            dmu = abs(mu - muBkp) / abs(mu)
            dt = dt / 2

        end_time = time.time()
        runtime = end_time - start_time
        print('Time ran for ground state computation %.3fs' % runtime)
        self.mufinal=mu
        self.psifinal=psi
        self.nsteps=n