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
    
    def Eq14(self, x, t, R=1./6, min_points=10):
        """ CVE_Eq14 function
        takes positions and uses equation 14 from
        Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        Phys. Rev. E 2014, 89 (2), 022726. 
        https://doi.org/10.1103/PhysRevE.89.022726.
        to estimate diffusion coefficient and
        localisation precision
        
        Args:
            x (np.1darray): 1D positions
            t (np.1darray): time
            R (float): motion blur parameter (see Equation 5 of paper)
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
        mult = np.mean(np.multiply(diffX[:-1], diffX[1:]))
        deltaX_sqr = np.mean(np.square(diffX))
        deltat = np.mean(np.unique(np.diff(t)))
        D = np.add(np.divide(deltaX_sqr, np.multiply(2, deltat)), np.divide(mult, deltat))
        
        sigma = np.sqrt(np.multiply(R, deltaX_sqr) + np.multiply((2*R - 1), mult))
        
        epsilon = np.subtract(np.divide(np.square(sigma), np.multiply(D, deltat)), np.multiply(2, R))
        N = len(x)
        varD = np.multiply(np.square(D), (((6 + 4*epsilon + 2*np.square(epsilon))/N) + ((4*np.square(1+epsilon))/np.square(N))))
        return D, sigma, varD
    
    def Eq16(self, x, sigma, t, R=1./6, min_points=10):
        """ CVE_Eq16 function
        takes positions and uses equation 16 from
        Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        Phys. Rev. E 2014, 89 (2), 022726. 
        https://doi.org/10.1103/PhysRevE.89.022726.
        to estimate diffusion coefficient
        
        Args:
            x (np.1darray): 1D positions
            sigma (np.1darray): precision of estimations of x
            t (np.1darray): time
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
        deltat = np.mean(np.unique(np.diff(t)))
        D = np.divide(np.subtract(deltaX_sqr, np.multiply(2., sigma_squared)), 
                      np.multiply(np.multiply(2, np.subtract(1., np.multiply(2., R))), deltat))
        varsigma = np.var(np.square(sigma))
        epsilon = np.subtract(np.divide(np.square(sigma), 
                    np.multiply(D, deltat)), np.multiply(2, R))
        N = len(x)
        varD = np.add(np.divide(np.multiply(np.square(D), 
                (2 + 4*epsilon + 3*np.square(epsilon))), N*np.square(1 - 2*R)),
                     np.divide(varsigma, (np.square(deltat)*np.square(1 - 2*R))))
        
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