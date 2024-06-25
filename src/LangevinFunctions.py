#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:00:36 2023

@author: jbeckwith
"""

import numpy as np
from scipy.sparse import diags_array
from scipy.spatial.distance import cdist
import pathos
from pathos.pools import ThreadPool as Pool

cpu_number = int(pathos.helpers.cpu_count() * 0.8)


class LF:
    def __init__(self):
        self = self
        return

    def MultiBrownianTrans_Par(
        self,
        DT,
        N,
        n_d,
        deltaT,
        tE,
        sigma0,
        s0,
        n_molecules=10,
        volume=np.array([8, 8, 8]),
        R=1.0 / 6,
        min_track_length=3,
    ):
        """MultiBrownianTrans_Volume function
        Generates displacements Delta from equations 2--5
        of Michalet, X.; Berglund, A. J.
        Phys. Rev. E 2012, 85 (6), 061916.
        https://doi.org/10.1103/PhysRevE.85.061916.
        More realistic Brownian motion assuming tracked with a camera
        Simulates a specific volume
        Will split tracks that reach sigma0 from each other

        Args:
            DT (float): translational diffusion coefficient
            N (int): number of steps to simulate
            n_d (int): mumber of dimensions of diffusion
            deltaT (flaot): time step between data points
            tE (float): camera exposure duration (can be same as deltaT)
            sigma0 (float): static localisation error
            s0 (float): standard deviation of the PSF
            n_molecules (int): number of molecules simualted in the volume
            volume (np.1darray): dimensions of volume simulated, in same units as DT
                                (same number of dimensions as n_d)
            R (float): motion blur coefficient (see Equation 5 of paper)

        Returns:
            coordinates (dict): coordinates of n_molecules over time
        """

        coordinates = {}  # make initial dict

        def generate_multiple_molecules(n):
            limits = 0
            while limits == 0:
                starting_position = np.zeros(n_d)
                for axis in np.arange(n_d):
                    starting_position[axis] = np.random.uniform(
                        low=0.0, high=volume[axis]
                    )
                single_track = starting_position + self.BrownianTrans_Realistic(
                    DT, N, n_d, deltaT, tE, sigma0, s0, R
                )  # make single track
                if np.any(np.max(single_track, axis=0) > volume) or np.any(
                    np.min(single_track, axis=0) < np.zeros(3)
                ):
                    cutlimit = np.min(
                        np.hstack(
                            [
                                np.where(single_track > volume)[0],
                                np.where(single_track < np.zeros(3))[0],
                            ]
                        )
                    )
                    if cutlimit < min_track_length:
                        limits = 0
                    else:
                        coordinates[n] = single_track[:cutlimit, :]
                        limits = 1
                else:
                    coordinates[n] = single_track
                    limits = 1

        pool = Pool(nodes=cpu_number)
        pool.restart()
        pool.map(generate_multiple_molecules, np.arange(n_molecules))
        pool.close()
        pool.terminate()

        return coordinates

    def MultiBrownianTrans_Volume(
        self,
        DT,
        N,
        n_d,
        deltaT,
        tE,
        sigma0,
        s0,
        n_molecules=10,
        volume=np.array([8, 8, 8]),
        bleach_probability=0.0,
        R=1.0 / 6,
        min_track_length=10,
    ):
        """MultiBrownianTrans_Volume function
        Generates displacements Delta from equations 2--5
        of Michalet, X.; Berglund, A. J.
        Phys. Rev. E 2012, 85 (6), 061916.
        https://doi.org/10.1103/PhysRevE.85.061916.
        More realistic Brownian motion assuming tracked with a camera
        Simulates a specific volume
        Will split tracks that reach sigma0 from each other

        Args:
            DT (float): translational diffusion coefficient
            N (int): number of steps to simulate
            n_d (int): mumber of dimensions of diffusion
            deltaT (flaot): time step between data points
            tE (float): camera exposure duration (can be same as deltaT)
            sigma0 (float): static localisation error
            s0 (float): standard deviation of the PSF
            n_molecules (int): number of molecules simualted in the volume
            volume (np.1darray): dimensions of volume simulated, in same units as DT
                                (same number of dimensions as n_d)
            bleach_probability (float): if above 0, assigns a random
                                        proportion of frames to "off" and
                                        replaces them with interpolation.
            R (float): motion blur coefficient (see Equation 5 of paper)

        Returns:
            coordinates (dict): coordinates of n_molecules over time
        """
        try:
            if bleach_probability > 0.5:
                raise Exception(
                    """Function won't be able to find enough points
                                to interpolate between if bleaching probability
                                is above 50 percent!"""
                )
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        initial_coordinates = np.zeros([n_molecules, N, n_d])  # make initial tensor
        for n in np.arange(n_molecules):
            limits = 0
            while limits == 0:
                starting_position = np.zeros(n_d)
                for axis in np.arange(n_d):
                    starting_position[axis] = np.random.uniform(
                        low=0.0, high=volume[axis]
                    )
                single_track = starting_position + self.BrownianTrans_Realistic(
                    DT, N, n_d, deltaT, tE, sigma0, s0, R
                )  # make single track
                if np.any(np.max(single_track, axis=0) > volume) or np.any(
                    np.min(single_track, axis=0) < np.zeros(3)
                ):
                    limits = 0
                else:
                    initial_coordinates[n, :, :] = single_track
                    limits = 1

        # now remove steps due to bleaching
        if bleach_probability > 0:
            to_delete = initial_coordinates.ravel()
            deletion_indices = np.random.choice(
                np.arange(len(to_delete)), size=int(bleach_probability * len(to_delete))
            )
            to_delete[deletion_indices] = np.NAN
            initial_coordinates = to_delete.reshape(initial_coordinates.shape)
            nanlocs_mol, nanlocs_t, nanlocs_dim = np.where(
                np.isnan(initial_coordinates)
            )
            for i in np.arange(len(nanlocs_mol)):
                mol = nanlocs_mol[i]
                t = nanlocs_t[i]
                dim = nanlocs_dim[i]
                if t == 0:
                    initial_coordinates[mol, t, dim] = (
                        initial_coordinates[mol, t + 1, dim] / 2.0
                    )  # crude approximation
                elif t == N - 1:
                    initial_coordinates[mol, t, dim] = (
                        initial_coordinates[mol, t - 1, dim] * 2.0
                    )  # crude approximation
                else:
                    initial_coordinates[mol, t, dim] = 0.5 * (
                        initial_coordinates[mol, t - 1, dim]
                        + initial_coordinates[mol, t + 1, dim]
                    )  # crude approximation

        # now chop up the tracks if they get too close
        coordinates = {}
        initial_n = 0
        molecule_list = np.arange(n_molecules)
        for n in molecule_list:
            other_molecule_list = molecule_list[molecule_list != n]
            distance_between_molecules = np.zeros([N, n_molecules - 1])
            for i, k in enumerate(other_molecule_list):  # need some while loop here
                molecular_track = initial_coordinates[n, :, :]
                other_molecular_track = initial_coordinates[k, :, :]
                distance_between_molecules[:, i] = np.diag(
                    cdist(molecular_track, other_molecular_track)
                )
            min_distances = np.min(distance_between_molecules, axis=0)
            break_point = np.where(min_distances < sigma0)[0]
            if len(break_point) > 0:
                if len(break_point) == 1:
                    b = break_point[0]
                    if len(molecular_track[:b, :]) > min_track_length:
                        coordinates[initial_n] = molecular_track[:b, :]
                        initial_n += 1
                    if len(molecular_track[b:, :]) > min_track_length:
                        coordinates[initial_n] = molecular_track[b:, :]
                        initial_n += 1
                else:
                    for i, b in enumerate(break_point):
                        if b == break_point[0]:
                            if len(molecular_track[:b, :]) > min_track_length:
                                coordinates[initial_n] = molecular_track[:b, :]
                                initial_n += 1
                        elif b == break_point[-1]:
                            if len(molecular_track[b:, :]) > min_track_length:
                                coordinates[initial_n] = molecular_track[:b, :]
                                initial_n += 1
                        else:
                            if (
                                len(molecular_track[break_point[i - 1] : b, :])
                                > min_track_length
                            ):
                                coordinates[initial_n] = molecular_track[
                                    break_point[i - 1] : b, :
                                ]
                                initial_n += 1
            else:
                coordinates[initial_n] = molecular_track
                initial_n += 1

        return coordinates

    def BrownianTrans_Realistic(self, DT, N, n_d, deltaT, tE, sigma0, s0, R=1.0 / 6):
        """BrownianTrans_Realistic function
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

        rng = np.random.default_rng()  # initialise rng

        sigma = np.multiply(
            sigma0, np.sqrt(np.add(1.0, np.divide(np.multiply(DT, tE), np.square(s0))))
        )  # make localisation error parameter

        diagonal = 2 * DT * deltaT * (1 - 2 * R) + 2 * np.square(sigma)
        off_diagonal = 2 * R * DT * deltaT - np.square(sigma)
        stack = np.vstack(
            [
                np.full(shape=N - 1, fill_value=off_diagonal),
                np.full(shape=N - 1, fill_value=diagonal),
                np.full(shape=N - 1, fill_value=off_diagonal),
            ]
        )  # get off diagonals for cov matrix

        cov = diags_array(stack, offsets=[-1, 0, 1], shape=(N - 1, N - 1)).toarray()

        displacements = rng.multivariate_normal(np.zeros(N - 1), cov, size=n_d)
        r0 = np.zeros([n_d])  # initial position is 0s

        return np.vstack([r0, np.cumsum(displacements, axis=1).T])

    def BrownianTrans_Ideal(self, DT, n_d, deltaT, N):
        """BrownianTrans_Ideal function
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
        sigmaT = np.sqrt(np.multiply(2.0, np.multiply(DT, deltaT)))

        r0 = np.zeros([n_d])  # initial position is 0s
        # generate random numbers for all steps
        rns = np.random.normal(loc=0, scale=sigmaT, size=(n_d, N - 1))
        # make coordinates
        coordinates = np.vstack([r0, np.cumsum(rns, axis=1).T])

        return coordinates

    def BrownianRot(self, DR, TStep, NSteps):
        """BrownianRot function
            generates random rotational motion
            using DR and TStep

        Args:
            DR (float): rotational diffusion coefficient
            TStep (float): time step relative to DT
            NSteps (int): number of steps to simulate

        Returns:
            sph_coords (np.3Darray): are theta, phi over time"""

        # generate spherical coordinates using method of Hunter, G. L.;
        # Edmond, K. V.; Elsesser, M. T.; Weeks, E. R.
        # Opt. Express 2011, 19 (18), 17189–17202.
        # Equations to convert Cartesian coordinates to Spherical coordinates
        # Rotation matrices
        r0 = np.zeros([3]) + np.sqrt(np.divide(1, 3.0))  # initial position is sqrt(1/3)

        Rx = lambda alpha: np.array(
            [
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)],
            ]
        )
        Ry = lambda beta: np.array(
            [
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)],
            ]
        )
        Rz = lambda gamma: np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1],
            ]
        )

        sigmaT = np.sqrt(np.multiply(2.0, np.multiply(DR, TStep)))
        # equations to convert x y and z to theta and phi
        # see https://en.wikipedia.org/wiki/Spherical_coordinate_system
        theta = lambda x, y, z: np.arccos(np.divide(z, np.sqrt(x**2 + y**2 + z**2)))
        phi = lambda x, y: np.mod(np.arctan2(y, x), np.multiply(2.0, np.pi))

        # Simulate Rotational Diffusion
        coordinates = np.vstack([r0, np.zeros([NSteps - 1, 3])]).T

        xyzdisp = np.random.normal(loc=0, scale=sigmaT, size=(3, NSteps - 1))

        for j in np.arange(1, NSteps):
            r_prev = coordinates[:, j - 1]
            coordinates[:, j] = np.matmul(
                np.matmul(
                    np.matmul(Rx(xyzdisp[0, j - 1]), Ry(xyzdisp[1, j - 1])),
                    Rz(xyzdisp[2, j - 1]),
                ),
                r_prev,
            )

        # Convert Cartesian coordinates to spherical coordinates
        sph_coords = np.array(
            [
                theta(coordinates[0, :], coordinates[1, :], coordinates[2, :]),
                phi(coordinates[0, :], coordinates[1, :]),
            ]
        )
        return sph_coords
