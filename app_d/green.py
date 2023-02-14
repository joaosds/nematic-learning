#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:01:45 2021

@author: obernauer
"""

import numpy as np
from model import Model


class Green(Model):
    '''
    A subclass to calculate the Green's function for a toy model of
    bilayer graphene.

    Attributes
    ----------
    frequency : float
        frequency for calculating the Green's function G(R,R,frequency).
    eta : float
        small value 0+ needed for 1/... in Green's function.

    Methods
    -------
    green() :
        Calculates the Green's functions. For each combination of R1 and R2
        one gets a 4x4 matrix according to alpha and beta.
    '''
    def __init__(self):
        '''
        Constructs all the necessary attributes for green class.
        '''
        self.frequency = None
        self.eta = None
        super().__init__()

    def green(self) -> np.ndarray:
        '''
        Calculates the Green's functions. For each combination of R1 and R2
        one gets a 4x4 matrix according to alpha and beta.

        Returns
        -------
        green_arr : np.ndarray
            An R1xR2 array for all R combinations containing 4x4 matrices with the
            Green's function for alpha and beta.
        '''
        green_arr = np.array([
                             [np.zeros((4, 4), dtype = 'complex_') for i in range(len(self.R))]
                             for j in range(len(self.R))
                             ])

        N_inv = 1. / len(self.k_kar)

        #calculation must be based on full hamiltonian with four bands
        #no calculation for hal_ham possible as we need four bands
        if self.nematic:
            ham = self.ham_nematic
        elif self.full:
            ham = self.ham_full

        #make an 4x4 array with the fraction term of alpha beta separately for each k
        frac_arr = np.array([np.zeros((4, 4), dtype = 'complex_') for i in range(len(self.k_kar))])
        #fill array for different k values
        for k in range(len(self.k_kar)):
            h_k = ham(self.k_kar[k][0], self.k_kar[k][1])[0]
            frac = - h_k + (np.eye(4) * (self.frequency + complex(0, self.eta)))
            frac_arr[k] = np.linalg.inv(frac)

        #summation over different R vector combinations
        for i in range(len(self.R)):
            for j in range(len(self.R)):
                diff = np.subtract(self.R[i], self.R[j])
                #summation over different k values
                k_sum = 0
                for k in range(len(self.k_kar)):
                    k_sum += frac_arr[k] * np.exp(complex(0, np.dot(self.k_kar[k], diff)))
                green_arr[i][j] = k_sum

        return N_inv * green_arr


def main(eta: float, frequency: float, R_num: int, k_num: int,
         alpha: list, theta: float) -> None:
    '''
    Main function

    Parameters
    ----------
    eta : float
        small value needed for 1/... in greens function.
    frequency : float
        frequency for G(R,R,frequency).
    R_num : int
        number of Bravais lattice vectors
    k_num : int
        number of k points to calculate sum in greens function formula
    alpha : list
        list of nematic order parameters.
    theta : float
        angle of phi for nematic coupling.
    '''
    green = Green()
    green.eta = eta
    green.frequency = frequency
    #set if nematic order parameters
    green.alpha = alpha
    green.theta = theta
    #generate bravais lattice vectors
    green.bravais_lattice(R_num)
    #generate k points
    green.k_karman(k_num)
    #calculate greens functions and save them
    g_arr = green.green()
    np.save('green_'+str(green.frequency)+'omega'+str(green.eta)+'eta'+str(green.alpha)+'alpha'
            +str(round(green.theta, 2))+'theta.npy', g_arr)

if __name__ == "__main__":

    cfg = {
           'eta': 1e-2, #this has proven to be a good value
           'frequency': 2., #should be taken according to band plot
           'R_num': 7,
           'k_num': 50, #k_num**2 gives the total number of k points
           'alpha': [0., 0., 1., 0.],
           # A = 0., np.pi / 3., 2 * np.pi / 3.
           # B = np.pi / 6., np.pi / 2., 5 * np.pi / 6.
           'theta': 0.,
           }

    main(
         cfg['eta'],
         cfg['frequency'],
         cfg['R_num'],
         cfg['k_num'],
         cfg['alpha'],
         cfg['theta'],
         )
