#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:33:12 2021

@author: obernauer
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pywt
from tqdm import tqdm
from green import Green


def wannier(x: np.float64, y: np.float64, R: np.ndarray, label: int) -> np.float64:
    '''
    Wannier function corresponding to the underlying D3 symmetry with the form
    y(y^2-3x^2)+(x^2+y^2)^2.
    http://gernot-katzers-spice-pages.com/character_tables/D3.html
    The sign change of the exponential function corresponds to "fidget spinners"
    looking up (with -) or down (without -).

    Parameters
    ----------
    x : np.float64
        Continuum point for dI/dV(x,y) as needed for wannier((x,y)-R-delta).
    x : np.float64
        Continuum point for dI/dV(x,y) as needed for wannier((x,y)-R-delta).
    R : np.ndarray
        Bravais lattice vector for wannier(x-R-delta_s).
    label : int
        chooses right wannier function and can have the values in range [0,3]
        as alpha and beta have the same range.

    Returns
    -------
    result : np.float64
        Result of wannier function.
    '''
    #this parameter tunes the extent of the wannier function
    a = 0.19

    # delta_A = np.array([1./np.sqrt(3), 0.]) / a
    # delta_B = - delta_A

    # delta_A = 1 / np.sqrt(3) * np.array([1. / 2., np.sqrt(3) / 2.]) / a
    # delta_B = 1 / np.sqrt(3) * np.array([-1. / 2., np.sqrt(3) / 2.]) / a

    delta_A = np.array([(1. / np.sqrt(3)) * 0.5, 0.]) / a
    delta_B = - delta_A

    R = R / a
    x = x / a
    y = y / a

    x_A = x - R[0] - delta_A[0]
    y_A = y - R[1] - delta_A[1]
    x_B = x - R[0] - delta_B[0]
    y_B = y - R[1] - delta_B[1]

    if label == 0: # s=A, j=1
        result = np.exp(
                    -(- 1.5 * y_A * (y_A ** 2 - 3 * (x_A ** 2))
                      + 0.7 * (x_A ** 2 + y_A ** 2) ** 2))

    elif label == 2: # s=A, j=2
        result = np.exp(
                    -( 1.5 * y_A * (y_A ** 2 - 3 * (x_A ** 2))
                      + 0.7 * (x_A ** 2 + y_A ** 2) ** 2))
    #change of sublattice
    elif label == 1: # s=B, j=1
        result = np.exp(
                    -(- 1.5 * y_B * (y_B ** 2 - 3 * (x_B ** 2))
                      + 0.7 * (x_B ** 2 + y_B ** 2) ** 2))

    elif label == 3: # s=B, j=2
        result = np.exp(
                    -( 1.5 * y_B * (y_B ** 2 - 3 * (x_B ** 2))
                      + 0.7 * (x_B ** 2 + y_B ** 2) ** 2))

    return result

def wavelet_trafo(ldos: list) -> np.ndarray:
    '''
    Function to perform continous wavelet transformation with pywavelets library.
    https://pywavelets.readthedocs.io/en/latest/ref/cwt.html

    Parameters
    ----------
    ldos : list
        List of the LDOS at one symmetry point over an energy range.

    Returns
    -------
    coef : np.ndarray
        This gives the 2D image of the wavelet transform.

    '''
    coef, freqs = pywt.cwt(ldos, np.arange(1,66), 'morl')
    del freqs
    cmap = plt.cm.coolwarm.copy()
    plt.imshow(coef, cmap = cmap, interpolation = 'none',
                extent=[-4,4,1,65], aspect='auto',
                vmax=abs(coef).max(), vmin=-abs(coef).max())
    plt.colorbar()
    plt.show()

    return coef


class STM(Green):
    '''
    A subclass to calculate the dI/dV map for a toy model of
    bilayer graphene.

    Attributes
    ----------
    lim : float
        limit of the xy grid
    step : float
        step size of the xy grid

    Methods
    -------
    didv() :
        Calculates the tunneling current dI(x,V)/dV from the STM tip.
    didv_plot() :
        Plots the STM map.
    ldos() :
        This function calculates and plots the LDOS over
        an energy range for a certain sublattice point.
    '''
    def __init__(self):
        '''
        Constructs all the necessary attributes for stm class.
        '''
        self.lim = 0.65
        self.step = 0.02
        super().__init__()

    def didv(self, g_arr: np.ndarray) -> np.ndarray:
        '''
        Calculates the tunneling current dI(x,V)/dV from the STM tip

        Parameters
        ----------
        g_arr : np.ndarray
            An array with the Green's function for all R from green method.

        Returns
        -------
        didv : np.ndarray
            Tunneling current dI(x,V)/dV from the STM tip.
        '''
        grid = np.arange(-self.lim, self.lim, self.step)
        didv = np.zeros((len(grid), len(grid)), dtype = 'complex_')
        print('\n\nSTM calculation:\n')
        for x in tqdm(range(len(grid)), position=0, leave=True):
            for y in range(len(grid)):
                trafo = 0.
                for i in range(len(self.R)):
                    diff_i = LA.norm(np.array([grid[x], grid[y]]) - self.R[i])
                    if diff_i > 1.:
                        trafo += 0.
                    else:
                        for j in range(len(self.R)):
                            diff_j = LA.norm(np.array([grid[x], grid[y]]) - self.R[j])
                            if diff_j > 1.:
                                trafo += 0.
                            else:
                                for alpha in range(4):
                                    w_1 = wannier(grid[x], grid[y], self.R[i], alpha)
                                    for beta in range(4):
                                        w_2 = wannier(grid[x], grid[y], self.R[j], beta)
                                        trafo += w_1 * (g_arr[i][j][alpha][beta] * w_2.conjugate())
                didv[x][y] = trafo

        return (-1) * didv.imag

    def didv_plot(self, didv: np.ndarray) -> None:
        '''
        Plots the stm map.

        Parameters
        ----------
        didv : np.ndarray
            Tunneling current dI(x,V)/dV from the STM tip.
        '''
        cmap = plt.cm.coolwarm.copy()
        plt.imshow(
                   didv.T, cmap = cmap, interpolation = 'none',
                   extent=[-didv.shape[1]/2.*self.step, didv.shape[1]/2.*self.step,
                           -didv.shape[0]/2.*self.step, didv.shape[0]/2.*self.step],
                   origin='lower'
                   )
        #plt.colorbar()#ticks=[didv.min(), didv.min()/2., 0, didv.max()/2., didv.max()])
        plt.colorbar()
        plt.axis('off')
        plt.gca().set_aspect('equal')
        # plt.scatter(1. / np.sqrt(3) / 2., 0., color='orange', marker='x')
        # plt.xlim(-self.lim, self.lim)
        # plt.ylim(-self.lim, self.lim)

    def ldos(self, pos=np.array([1. / np.sqrt(3) / 2., 0.])):
        '''
        dI/dV is expected to be proportional to the system’s local density
        of states (LDOS). This function calculates and plots the LDOS over
        an energy range for a certain sublattice point. Random normal noise
        can be added.

        Parameters
        ----------
        pos : list, optional
            Position where LDOS is calculated.
            The default is [1. / np.sqrt(3) / 2., 0.].

        Returns
        -------
        ldos : list
            The resulting LDOS values.
        '''
        x = pos[0]
        y = pos[1]
        energy = np.linspace(-4, 4, 65)

        ldos = []
        for w in range(len(energy)):
            self.frequency = energy[w]
            g_arr = self.green()
            trafo = 0.
            for i in range(len(self.R)):
                for j in range(len(self.R)):
                    for alpha in range(4):
                        w_1 = wannier(x, y, self.R[i], alpha)
                        for beta in range(4):
                            w_2 = wannier(x, y, self.R[j], beta)
                            trafo += w_1 * g_arr[i][j][alpha][beta] * w_2.conjugate()
            ldos.append((-1.) * trafo.imag) #+ np.abs(np.random.normal(0, 0.05)))

        # max_value = max(ldos)
        # max_index = ldos.index(max_value)
        # print('\n\nDashed line at: ', energy[max_index])

        plt.plot(energy, ldos, 'k')
        plt.xlabel(r'$\omega (eV)$')
        plt.ylabel(r'$LDOS$')
        plt.grid()
        plt.title(r'$\eta = $'+str(self.eta)+
                  r', $\alpha = $'+str(np.round(self.alpha, 2))+
                  r', $\theta = $'+str(round(self.theta, 2)), fontsize=11)
        # plt.vlines(x=energy[max_index], ymin=0.0, ymax=max_value,
        #            colors='red', linestyles='dashed')
        plt.show()
        wavelet_trafo(ldos)

        return ldos


def main(eta: float, frequency: float, R_num: int, alpha: list, theta: float) -> None:
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
    alpha : list
        list of nematic order parameters.
    theta : float
        angle of phi for nematic coupling.
    '''
    stm = STM()
    stm.eta = eta
    stm.frequency = frequency
    stm.theta = theta
    stm.alpha = alpha
    stm.bravais_lattice(R_num)

    my_green_file = Path("/home/obernauer/Studium/Physik/Masterarbeit/STM/Code/"
                    'green_'+str(stm.frequency)+'omega'+str(stm.eta)+'eta'+str(stm.alpha)+'alpha'
                    +str(round(stm.theta, 2))+'theta.npy')
    #try loading an existing file with greens functions
    if my_green_file.is_file():
        g_arr = np.load(my_green_file)
        didv = stm.didv(g_arr)
        # stm.grid_plot()
        # stm.bravais_plot()
        plt.title(r'$\omega = $'+str(stm.frequency)+r'$, \eta = $'+str(stm.eta)+
                  r', $\alpha = $'+str(stm.alpha)+
                  r', $\theta = $'+str(round(stm.theta, 2)), fontsize=9)
        stm.didv_plot(didv)
        plt.show()

    else:
        print('\nGenerate Green function file with same parameters first!\n')

    #--------------------------------LDOS------------------------------------#
    ldos = STM()
    ldos.nematic = True
    ldos.full = True

    ldos.bravais_lattice(7)
    ldos.k_karman(10)
    ldos.eta = 0.1

    # A = 0., np.pi / 3., 2 * np.pi / 3.
    # B = np.pi / 6., np.pi / 2., 5 * np.pi / 6.
    ldos.theta = 5 * np.pi / 6.
    ldos.alpha = [0.03, 0.02, 0.07, 0.05]

    plt.title(r'$\eta = $'+str(ldos.eta)+
              r', $\alpha = $'+str(ldos.alpha)+
              r', $\theta = $'+str(round(ldos.theta, 2)), fontsize=11)
    ldos.ldos()

if __name__ == "__main__":

    cfg = {
           'eta': 1e-2, #this has proven to be a good value
           'frequency': 2., #should be taken according to band plot
           'R_num': 7,
           'alpha': [0., 0., 1., 0.],
           # A = 0., np.pi / 3., 2 * np.pi / 3.
           # B = np.pi / 6., np.pi / 2., 5 * np.pi / 6.
           'theta': 5 * np.pi / 6.,
           }

    main(
         cfg['eta'],
         cfg['frequency'],
         cfg['R_num'],
         cfg['alpha'],
         cfg['theta'],
         )
