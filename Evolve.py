__author__ = 'avinash'

import numpy as np
from numpy import sqrt, pi, exp, linspace, sin, square, cos, fft, linspace, append, arange, log,trapz, arctan2
from math import floor, ceil, atan2
import matplotlib.pyplot as pl
import time
import pyfftw
import multiprocessing
import progressbar
from numba import jit

pyfftw.interfaces.cache.enable()

class TimeEvolution(object):
    def __init__(self, dt,nsteps,measureStep,nframes,psiI, mu0,gN,eps, xmesh=None, ymesh=None):

        @jit
        def Vt(x, y, n):                 #Use Vt in the evolution for time varying potential
            eps_fin = eps * 1.5
            return 2.0 * sqrt(x ** 2 + (eps_fin* (1-0.5*(n/nsteps))* y) ** 2)      # Define time varying potential

        V = 2.0 * sqrt(xmesh ** 2 + (eps * (1 - 0.5 * (0.0 / nsteps)) * ymesh) ** 2)

        @jit
        def Potstep(psi,n):
            V=2.0 * sqrt(xmesh ** 2 + (eps * (1 - 0.5 * (0.0 / nsteps)) * ymesh) ** 2)
            Ur = exp(-1j * dt * 0.5 * (V - mu0))
            return Ur * exp(-1j * dt * 0.5 * gN * np.absolute(psi) ** 2) * psi


        def phi(x,y):
            return np.piecewise(x, [x < 0, x >= 0], [0.0, pi])

        Ny, Nx = xmesh.shape
        nthread = multiprocessing.cpu_count()
        a = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
        b = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
        fft_object = pyfftw.FFTW(a, b, axes=(0, 1), threads=4)
        c = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
        ifft_object = pyfftw.FFTW(b, c, axes=(0, 1), threads=4, direction='FFTW_BACKWARD')

        dx = xmesh[1,2] - xmesh[1,1]
        dy = ymesh[2,1] - ymesh[1,1]
        kxgrid = 1. / (Nx * dx) * append(arange(0,floor((Nx - 1) / 2.)+1), arange(-ceil((Nx - 1) / 2.),0))
        kygrid = 1. / (Ny * dy) * append(arange(0,floor((Ny - 1) / 2.)+1), arange(-ceil((Ny - 1) / 2.),0))
        kxmesh, kymesh = np.meshgrid(kxgrid, kygrid)
        Uk = exp(-1j * dt * 0.5 * (kxmesh ** 2 + kymesh ** 2))
        n = 0
        psi = psiI#*exp(1j *phi(xmesh,ymesh))



        bar = progressbar.ProgressBar()
        start_time = time.time()
        for n in progressbar.progressbar(range(1,nsteps)):

            Ur = exp(-1j * dt *0.5* (Vt(xmesh,ymesh,n)-mu0))

            psi = Ur * exp(-1j * dt * 0.5 * gN * np.conj(psi) *psi) * psi

            psi = pyfftw.interfaces.numpy_fft.fft2(psi, threads=nthread)

            psi = Uk * psi

            psi = pyfftw.interfaces.numpy_fft.ifft2(psi, threads=nthread)

            Ur = exp(-1j * dt * 0.5 * (Vt(xmesh,ymesh,n+0.5) - mu0))
            psi = Ur * exp(-1j * dt * 0.5 * gN * np.conj(psi) *psi) * psi


        end_time = time.time()
        runtime = end_time - start_time
        print('Time ran for evolution %.3fs' % runtime)
        self.psifinal=psi