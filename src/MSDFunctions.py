#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:30:10 2024

@author: jbeckwith
"""
import numpy as np


class MSD():
    def __init__(self):
        self = self
        return

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
        return S1-2*S2