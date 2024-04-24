#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 08:54:52 2020

@author: jbeckwith
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import numpy as np
import scipy.optimize
from scipy.fft import dst

class D_MLE():
    def __init__(self):
        self = self
        return
    
    def DSigma2_MLE(self, coordinates, dT, R, n_d=1, maxiter=100000, maxfun=100000):
        """
        Compute diffusion coefficient estimate, and estimate of the
        dynamic localisation error, using the MLE approach.
    
        Args:
            coordinates (np.ndarray): coordinates over time.
            dT (float): Time step.
            R (float): Motion blur coefficient.
            n_d (int): number of dimensions. If above 1, coordinates second
                        dimension should be same shape as this number
            maxiter (int): maximum number of optimisation iterations to make
            maxfun (int): maximum number of function evaluations to make
    
        Returns:
            D (float): estimate of D value.
            sigma2 (float): estimate of sigma2 value.
        """
        # get initial guess of D and Sigma2
        diff_coordinates = np.diff(coordinates, axis=0)
        var_diff_coordinates = np.var(diff_coordinates, axis=0)
        D_i = np.mean(var_diff_coordinates) / (2 * dT * 2)        
        sigma2_i = np.mean(var_diff_coordinates) / 2

        # make displacements array
        dX = np.subtract(np.diff(coordinates, axis=0),
                         np.tile(np.mean(np.diff(coordinates, axis=0), axis=0),
                                 (len(coordinates) - 1, 1)))
        
        sine_transform = np.square(dst(dX, 2, axis=0)/2.)
        optimfunc = lambda x: -self.likelihood_subfunction(sine_transform, x[0], x[1], dT, R)
        xopt = scipy.optimize.fmin(func=optimfunc, x0=[D_i, sigma2_i], maxiter=maxiter, maxfun=maxfun)
        D = xopt[0]
        sigma2 = xopt[1]
        return D, sigma2
        
    def likelihood_subfunction(self, d_xx, D, sig2, dT, R, n_d=1):
        """
        Compute log-likelihood for trajectories of particle tracking.
    
        Args:
            d_xx (numpy.ndarray): Square distance of the difference of 
                                trajectory. Second axis should be n_d
            D (float): Diffusion coefficient.
            sig2 (float): Variance.
            dT (float): Time step.
            R (float): Motion blur coefficient.
            n_d (int): number of dimensions. If above 1, d_xx second
                        dimension should be same shape as this number
    
        Returns:
            L (float): Likelihood value.
        """
        if D < 0 or sig2 < 0:
            # Return negative infinity if D or sig2 is negative
            return -np.inf
    
        if n_d > 1:
            try:
                if d_xx.shape[1] != n_d:
                    raise Exception('Dimension of d_xx and n_d are inconsistent.')
            except Exception as error:
                print('Caught this error: ' + repr(error))
                return
            
        L = 0. # initialise L
        N = d_xx.shape[0]
        # Calculate parameters for the likelihood computation
        alpha = 2 * D * dT * (1 - 2 * R) + 2 * sig2
        beta = 2 * R * D * dT - sig2
        # Compute the eigenvalues vector
        eigvec = alpha + 2 * beta * np.cos(np.pi * np.arange(1, N + 1) / (N + 1))
        eigvec = eigvec.reshape(d_xx.shape[0])
        if n_d == 1:
            # Compute the likelihood
            L += -0.5 * np.sum(np.log(eigvec) + (2 / (N + 1)) * d_xx / eigvec)
        else:
            for n in np.arange(n_d):
                # Compute the likelihood
                L += -0.5 * np.sum(np.log(eigvec) + (2 / (N + 1)) * d_xx[:,n] / eigvec)
        return L