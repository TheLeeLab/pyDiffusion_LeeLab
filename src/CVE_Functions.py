#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:00:36 2023

@author: jbeckwith
"""

import numpy as np

class CVE():
    def __init__(self):
        self = self
        return
    
    def DSigma_CVE_BootStrap(self, coordinates, dT, R=1./6, n_d=1, min_points=10, n_samples=1000):
        """
        Compute errors on diffusion coefficient estimate, and estimate of the
        dynamic localisation error, using the CVE approach.
    
        Args:
            coordinates (np.ndarray): coordinates over time.
            dT (float): Time step.
            R (float): Motion blur coefficient.
            n_d (int): number of dimensions. If above 1, coordinates second
                        dimension should be same shape as this number
            maxiter (int): maximum number of optimisation iterations to make
            maxfun (int): maximum number of function evaluations to make
            min_points (int): minimum number of points for a diffusion estimate.
                            Default is 10.
            n_samples (int): number of bootstrapped samples.

        Returns:
            D_error (float): estimate of D error.
            sigma_error (float): estimate of error in dynamic localisation std.
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
        
        D_d_i = np.zeros([n_d, n_samples])
        var_d_i = np.zeros([n_d, n_samples])
        
        samples = np.diff(coordinates, axis=0).ravel()
        r0 = np.zeros([n_d]) # initial position is 0s
        for k in np.arange(n_samples):
            displacements = np.random.choice(samples, size=(coordinates.shape[0]-1, n_d))
            new_coordinates = np.vstack([r0, np.cumsum(displacements, axis=0)])
            for i in np.arange(n_d):
                D_d_i[i, k], sigma, var_d_i[i, k] = self.Eq14(new_coordinates[:, i], dT, R)
        
        D_err = np.nanmean(D_d_i, axis=0)
        var_err = np.nanmean(var_d_i, axis=0)
        return np.nanstd(D_err), np.nanstd(np.sqrt(var_err))

    def DSigma_CVE(self, coordinates, dT, R=1./6, n_d=1, min_points=10):
        """
        Compute diffusion coefficient estimate, and estimate of the
        dynamic localisation error, using the CVE approach.
    
        Args:
            coordinates (np.ndarray): coordinates over time.
            dT (float): Time step.
            R (float): Motion blur coefficient.
            n_d (int): number of dimensions. If above 1, coordinates second
                        dimension should be same shape as this number
            maxiter (int): maximum number of optimisation iterations to make
            maxfun (int): maximum number of function evaluations to make
            min_points (int): minimum number of points for a diffusion estimate.
                            Default is 10.

        Returns:
            D (float): estimate of D value.
            sigma (float): estimate of dynamic localisation std.
        """
        # get initial guess of D and Sigma2
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
        
        D_d = np.zeros(n_d)
        var_d = np.zeros(n_d)
        
        for i in np.arange(n_d):
            D_d[i], sigma, var_d[i] = self.Eq14(coordinates[:, i], dT, R)
        
        D = np.mean(D_d)
        var = np.mean(var_d)
        return D, np.sqrt(var)

    def Eq14(self, x, dT, R=1./6, min_points=10):
        """ CVE_Eq14 function
        takes positions and uses equation 14 from
        Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        Phys. Rev. E 2014, 89 (2), 022726. 
        https://doi.org/10.1103/PhysRevE.89.022726.
        to estimate diffusion coefficient and
        localisation precision
        
        Args:
            x (np.1darray): 1D positions
            dT (np.1darray): time step
            R (float): motion blur parameter (see Equation 5 of paper)
            min_points (float): minimum track length
        
        Returns:
            D (float): diffusion coefficient estimate
            varD (float): variance of D 
        """        
        diffX = np.diff(x)
        mult = np.mean(np.multiply(diffX[:-1], diffX[1:]))
        deltaX_sqr = np.mean(np.square(diffX))
        D = np.add(np.divide(deltaX_sqr, np.multiply(2, dT)), np.divide(mult, dT))
        
        sigma = np.sqrt(np.multiply(R, deltaX_sqr) + np.multiply((2*R - 1), mult))
        
        epsilon = np.subtract(np.divide(np.square(sigma), np.multiply(D, dT)), np.multiply(2, R))
        N = len(x)
        varD = np.multiply(np.square(D), (((6 + 4*epsilon + 2*np.square(epsilon))/N) + ((4*np.square(1+epsilon))/np.square(N))))
        return D, sigma, varD
    
    def Eq16(self, x, sigma, dT, R=1./6, min_points=10):
        """ CVE_Eq16 function
        takes positions and uses equation 16 from
        Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        Phys. Rev. E 2014, 89 (2), 022726. 
        https://doi.org/10.1103/PhysRevE.89.022726.
        to estimate diffusion coefficient
        
        Args:
            x (np.1darray): 1D positions
            sigma (np.1darray): precision of estimations of x
            dT (np.1darray): time step
            R (float): R parameter (see Equation 5 of paper)
            min_points (float): minimum track length

        Returns:
            D (float): diffusion coefficient estimate
            varD (float): variance of D 
        """
        try:
            if x.shape[0] < min_points:
                raise Exception("""Not enough points to calculate a reliable estimate of diffusion coefficient.
                                Please see section V of Michalet and Berglund, Optimal Diffusion Coefficient Estimation
                                in Single-Particle Tracking. Phys. Rev. E 2012, 85 (6), 061916. https://doi.org/10.1103/PhysRevE.85.061916.""")
        except Exception as error:
            print('Caught this error: ' + repr(error))
            return np.NAN, np.NAN

        diffX = np.diff(x)
        sigma_squared = np.square(np.mean((sigma)))
        deltaX_sqr = np.mean(np.square(diffX))
        D = np.divide(np.subtract(deltaX_sqr, np.multiply(2., sigma_squared)), 
                      np.multiply(np.multiply(2, np.subtract(1., np.multiply(2., R))), dT))
        varsigma = np.var(np.square(sigma))
        epsilon = np.subtract(np.divide(np.square(sigma), 
                    np.multiply(D, dT)), np.multiply(2, R))
        N = len(x)
        varD = np.add(np.divide(np.multiply(np.square(D), 
                (2 + 4*epsilon + 3*np.square(epsilon))), N*np.square(1 - 2*R)),
                     np.divide(varsigma, (np.square(dT)*np.square(1 - 2*R))))
        
        return D, varD
    
    def Eq10(self, D, deltaT, sigma, k, N, R=1./6):
        """ Eq10 function
        calculates theoretical form of power spectrum from
        Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        Phys. Rev. E 2014, 89 (2), 022726. 
        https://doi.org/10.1103/PhysRevE.89.022726.
        to estimate if particle is undergoing diffusive motion or not

        Args:
            D (float): diffusion coefficient
            deltaT (float): time difference between displacements
            sigma (float): precision
            k (int): modes of discrete sine transform
            N (int) number of data points
            R (float): R parameter (see Equation 5 of paper)
        
        Returns:
            Pk(np.1darray): theoretical form of the power spectrum 
        """
        term1 = np.multiply(D, np.square(deltaT))
        term2_1 = np.multiply(np.square(sigma), deltaT)
        term2_2 = np.multiply(2., np.multiply(D, np.multiply(R, np.square(deltaT))))
        term3 = np.subtract(1., np.cos(np.divide(np.multiply(np.pi, k), np.add(N, 1.))))
        term2 = np.multiply(np.multiply(2., np.subtract(term2_1, term2_2)), term3)
        Pk = np.add(term1, term2)
        return Pk