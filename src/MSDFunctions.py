#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:30:10 2024

@author: jbeckwith
"""
import numpy as np
from numba import jit

class MSD():
    def __init__(self):
        self = self
        return
    
    def DSigma2_OLSF_BootStrap(self, coordinates, dT, R=1./6, n_d=1, maxiter=100, min_points=10, n_samples=1000):
        """
        Compute diffusion coefficient error estimate, and estimate of the
        dynamic localisation variance, using bootstrapping of the OLSF MSD approach.
    
        Args:
            coordinates (numpy.ndarray): Input trajectory.
            dT (float): Time interval.
            R (float): Motion blur coefficient.
            n_d (int): number of dimensions. If above 1, coordinates second
                        dimension should be same shape as this number
            maxiter (int): Maximum number of iterations. Defaults to 100.
            min_points (int): minimum number of points for a diffusion estimate.
                            Default is 10.
            n_samples (int): number of bootstrapped samples. Default 1000.

        Returns:
            D_error (float): Diffusion coefficient error estimate.
            var_error (float): var error estimate.
        """
        if n_d > 1:
            try:
                if coordinates.shape[1] != n_d:
                    raise Exception('Dimension of coordinates and n_d are inconsistent.')
            except Exception as error:
                print('Caught this error: ' + repr(error))
                return np.NAN, np.NAN
            
        try:
            if coordinates.shape[0] < min_points:
                raise Exception("""Not enough points to calculate a reliable estimate of diffusion coefficient.
                                Please see section V of Michalet and Berglund, Optimal Diffusion Coefficient Estimation
                                in Single-Particle Tracking. Phys. Rev. E 2012, 85 (6), 061916. https://doi.org/10.1103/PhysRevE.85.061916.""")
        except Exception as error:
            print('Caught this error: ' + repr(error))
            return np.NAN, np.NAN
        
        D_err = np.zeros(n_samples)
        var_err = np.zeros(n_samples)
        
        samples = np.diff(coordinates, axis=0).ravel()
        r0 = np.zeros([n_d]) # initial position is 0s
        for i in np.arange(n_samples):
            displacements = np.random.choice(samples, size=(coordinates.shape[0]-1, n_d))
            new_coordinates = np.vstack([r0, np.cumsum(displacements, axis=0)])
            D_err[i], var_err[i] = self.DSigma2_OLSF(new_coordinates, dT, R=R, n_d=n_d)
        return np.std(D_err), np.std(var_err)
    
    def DSigma2_OLSF(self, coordinates, dT, R=1./6, n_d=1, maxiter=100, min_points=10):
        """
        Compute diffusion coefficient estimate, and estimate of the
        dynamic localisation error, using the OLSF MSD approach.
    
        Args:
            coordinates (numpy.ndarray): Input trajectory.
            dT (float): Time interval.
            R (float): Motion blur coefficient.
            n_d (int): number of dimensions. If above 1, coordinates second
                        dimension should be same shape as this number
            maxiter (int): Maximum number of iterations. Defaults to 100.
            min_points (int): minimum number of points for a diffusion estimate.
                            Default is 10.

        Returns:
            D (float): Diffusion coefficient estimate.
            var (float): var estimate.
        """
        if n_d > 1:
            try:
                if coordinates.shape[1] != n_d:
                    raise Exception('Dimension of coordinates and n_d are inconsistent.')
            except Exception as error:
                print('Caught this error: ' + repr(error))
                return np.NAN, np.NAN
            
        try:
            if coordinates.shape[0] < min_points:
                raise Exception("""Not enough points to calculate a reliable estimate of diffusion coefficient.
                                Please see section V of Michalet and Berglund, Optimal Diffusion Coefficient Estimation
                                in Single-Particle Tracking. Phys. Rev. E 2012, 85 (6), 061916. https://doi.org/10.1103/PhysRevE.85.061916.""")
        except Exception as error:
            print('Caught this error: ' + repr(error))
            return np.NAN, np.NAN


        NXM = len(coordinates.ravel())
        pa = np.array([int(np.floor(NXM / 10))])
        pb = np.array([int(np.floor(NXM / 10))])
        donea = 0
        doneb = 0
        iter = 1
        rho = []
        while iter <= maxiter and (not donea or not doneb):
            if iter == maxiter:
                donea = True
                doneb = True
                D = np.nan
                var = np.nan
                print('OLSF did not converge in {} iterations'.format(maxiter))
            else:
                if np.max(np.hstack([pa, pb])) > len(rho):
                    rho = self.msd_fft(coordinates)
                rho = rho[:np.max(np.hstack([pa, pb]))]
                if not donea:
                    A = np.vstack((np.ones(pa[-1]), np.arange(1, pa[-1] + 1))).T
                    B = rho[:A.shape[0]]
                    if B.shape[0] != A.shape[0]:
                        donea = True
                        doneb = True
                        D = np.nan
                        var = np.nan
                        print('OLSF did not converge in {} iterations'.format(maxiter))
                        return D, var
                    X = np.linalg.lstsq(A, B, rcond=None)[0]
                    aa, ba = X
                    xa = 0 if aa < 0 else np.inf if ba < 0 else aa / ba
                    newpa = self.PMin_XM(xa, NXM)
                    if np.any(np.isin(newpa, pa)):
                        Da = ba / (2 * dT * n_d)
                        var = (aa + 4 * Da * R * dT) / (2 * n_d)
                        donea = True
                    pa = np.hstack([pa, newpa])
                if not doneb:
                    A = np.vstack((np.ones(pb[-1]), np.arange(1, pb[-1] + 1))).T
                    B = rho[:int(pb[-1])]
                    if B.shape[0] != A.shape[0]:
                        donea = True
                        doneb = True
                        D = np.nan
                        var = np.nan
                        print('OLSF did not converge in {} iterations'.format(maxiter))
                        return D, var
                    X = np.linalg.lstsq(A, B, rcond=None)[0]
                    ab, bb = X
                    xb = 0 if ab < 0 else np.inf if bb < 0 else ab / bb
                    newpb = self.PMin_XM(xb, NXM)
                    if np.any(np.isin(newpb, pb)):
                        D = bb / (2 * dT * n_d)
                        doneb = True
                    pb = np.hstack([pb, newpb])
                iter += 1

        return D, var

    def autocorrFFT(self, x):
        """
        Compute the autocorrelation of a 1D signal using the 
        Fast Fourier Transform (FFT) method.
        
        Args:
            x (numpy.ndarray): Input signal.
        
        Returns:
            res (numpy.ndarray): Autocorrelation of the input signal.
        """
        N=len(x)
        F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res= (res[:N]).real   #now we have the autocorrelation in convention B
        n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
        return res/n #this is the autocorrelation in convention A
    
    def msd_fft(self, r):
        """
        Compute the mean squared displacement (MSD) using the
        Fast Fourier Transform (FFT) method.
        
        Args:
            r (numpy.ndarray): Trajectory data, where each row represents the
            coordinates of a particle at different time steps.
        
        Returns:
            S (numpy.ndarray): MSD computed for each time step.
        """
        N=len(r)
        D=np.square(r).sum(axis=1) 
        D=np.append(D,0) 
        S2=sum([self.autocorrFFT(r[:, i]) for i in range(r.shape[1])])
        Q=2*D.sum()
        S1=np.zeros(N)
        for m in range(N):
            Q=Q-D[m-1]-D[N-m]
            S1[m]=Q/(N-m)
        S = S1-2*S2
        return S[1:]
    
    @staticmethod
    @jit(nopython=True)
    def PMin_XM(x, N):
        """
        Calculate optimal fit point from formulae in Michalet, X. Mean Square 
        Displacement Analysis of Single-Particle Trajectories with 
        Localization Error: Brownian Motion in an Isotropic Medium. 
        Phys. Rev. E 2010, 82 (4), 041914. https://doi.org/10.1103/PhysRevE.82.041914.
                
        Args:
            x (float): Input value.
            N (int): Number of trajectory points.
        
        Returns:
            pa (int): Optimal fit point for parameter 'a'.
            pb (int): Optimal fit point for parameter 'b'.
        """
        fa = 2 + 1.6 * (x ** 0.51)
        La = 3 + ((4.5 * (N ** 0.4) - 8.5) ** 1.2)
        
        fb = 2 + 1.35 * x ** 0.6
        Lb = 0.8 + (0.564 * N)
        
        if np.isinf(x):
            pa = int(np.floor(La))
            pb = int(np.floor(Lb))
        else:
            pa = int(np.floor(fa * La / (fa ** 3 + La ** 3) ** 0.33))
            pb = min(int(np.floor(Lb)), int(np.floor(fb * Lb / (fb ** 3 + Lb ** 3) ** 0.33)))
        
        # Make sure nothing is zero
        pa = max(2, pa)
        pb = max(2, pb)
        
        return pa, pb