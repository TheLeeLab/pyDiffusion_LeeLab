#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:00:36 2023

@author: jbeckwith
"""

import numpy as np
from scipy.sparse import diags_array

class LF():
    def __init__(self):
        self = self
        return
    
    def BrownianTrans_Realistic(self, DT, N, n_d, deltaT, tE, sigma0, s0, R=1./6):
        """ BrownianTrans_Realistic function
        Generates displacements Delta from equations 2--5
        of Michalet, X.; Berglund, A. J.
        Phys. Rev. E 2012, 85 (6), 061916.
        https://doi.org/10.1103/PhysRevE.85.061916.
        More realistic Brownian motion assuming tracked with a camera

        Args:
            DT (float): translational diffusion coefficient
            N (int): number of steps to simulate
            n_d (int): mumber of dimensions of diffusion
            deltaT (flaot): time step between data points
            tE (float): camera exposure duration (can be same as deltaT)
            sigma0 (float): static localisation error
            s0 (float): standard deviation of the PSF
            R (float): motion blur coefficient (see Equation 5 of paper)


        Returns:
            coordinates (np.ndarray): coordinates over time
        """
        
        rng = np.random.default_rng() # initialise rng
        
        sigma = np.multiply(sigma0, np.sqrt(np.add(1., 
                                np.divide(np.multiply(DT, tE), 
                                            np.square(s0))))) # make localisation error parameter
        
        diagonal = 2 * DT * deltaT * (1 - 2 * R) + 2 * np.square(sigma)
        off_diagonal = 2 * R * DT * deltaT - np.square(sigma)
        stack = np.vstack([np.full(shape=N-1, fill_value=off_diagonal), 
                           np.full(shape=N-1, fill_value=diagonal),
                        np.full(shape=N-1, fill_value=off_diagonal)]) # get off diagonals for cov matrix
        
        cov = diags_array(stack, offsets=[-1, 0, 1], shape=(N-1, N-1)).toarray()

        displacements = rng.multivariate_normal(np.zeros(N-1), cov, size=n_d)
        r0 = np.zeros([n_d]) # initial position is 0s
        
        return np.vstack([r0, np.cumsum(displacements, axis=1).T])
    
    def BrownianTrans_Ideal(self, DT, n_d, deltaT, N):
        """ BrownianTrans_Ideal function
            generates random translational motion of N coordinates 
            using DT and TStep
            assumes no localisation error, or effects from imaging
        
        Args:
            DT (float): translational diffusion coefficient
            NAxes (int): mumber of dimensions of diffusion
            TStep (float): time step relative to DT
            NSteps (int): number of steps to simulate
        
        Returns:
            coordinates (np.ndarray): coordinates over time"""
        
        # get dimensionless translational diffusion coefficient (good explanation
        # in Allen, M. P.; Tildesley, D. J. Computer Simulation of Liquids,
        # 2nd ed.; Oxford University Press, 2017)
        sigmaT = np.sqrt(np.multiply(2., np.multiply(DT, deltaT)))
        
        r0 = np.zeros([n_d]) # initial position is 0s
        # generate random numbers for all steps
        rns = np.random.normal(loc=0, scale=sigmaT, size=(n_d, N-1))
        # make coordinates
        coordinates = np.vstack([r0, np.cumsum(rns, axis=1).T])

        return coordinates

    def BrownianRot(self, DR, TStep, NSteps):
        """ BrownianRot function
            generates random rotational motion
            using DR and TStep
        
        Args:
            DR (float): rotational diffusion coefficient
            TStep (float): time step relative to DT
            NSteps (int): number of steps to simulate
        
        Returns:
            sph_coords (np.3Darray): are theta, phi over time """

        # generate spherical coordinates using method of Hunter, G. L.;
        # Edmond, K. V.; Elsesser, M. T.; Weeks, E. R. 
        # Opt. Express 2011, 19 (18), 17189–17202.
        # Equations to convert Cartesian coordinates to Spherical coordinates
        # Rotation matrices
        r0 = np.zeros([3]) + np.sqrt(np.divide(1, 3.)) # initial position is sqrt(1/3)
        
        Rx = lambda alpha: np.array([[1, 0, 0], 
                                    [0, np.cos(alpha), -np.sin(alpha)], 
                                    [0, np.sin(alpha), np.cos(alpha)]])
        Ry = lambda beta: np.array([[np.cos(beta), 0, np.sin(beta)], 
                                    [0, 1, 0], 
                                    [-np.sin(beta), 0, np.cos(beta)]])
        Rz = lambda gamma: np.array([[np.cos(gamma), -np.sin(gamma), 0], 
                                    [np.sin(gamma), np.cos(gamma), 0], 
                                    [0, 0, 1]])
        
        sigmaT = np.sqrt(np.multiply(2., np.multiply(DR, TStep)))
        # equations to convert x y and z to theta and phi
        # see https://en.wikipedia.org/wiki/Spherical_coordinate_system
        theta = lambda x, y, z: np.arccos(np.divide(z, np.sqrt(x**2 + y**2 + z**2)))
        phi = lambda x, y: np.mod(np.arctan2(y, x), np.multiply(2., np.pi))
        
        # Simulate Rotational Diffusion
        coordinates = np.vstack([r0, np.zeros([NSteps-1, 3])]).T
        
        xyzdisp = np.random.normal(loc=0, scale=sigmaT, size=(3, NSteps-1))
        
        for j in np.arange(1, NSteps):
            r_prev = coordinates[:, j-1]
            coordinates[:, j] = np.matmul(np.matmul(np.matmul(Rx(xyzdisp[0, j-1]), Ry(xyzdisp[1, j-1])), Rz(xyzdisp[2, j-1])), r_prev)
            
        # Convert Cartesian coordinates to spherical coordinates
        sph_coords = np.array([theta(coordinates[0,:],coordinates[1,:],coordinates[2,:]), phi(coordinates[0,:],coordinates[1,:])]);
        return sph_coords