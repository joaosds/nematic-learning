#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:01:45 2021

@author: obernauer
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
#from hexalattice.hexalattice import create_hex_grid


def f_k(k_x: np.float64, k_y: np.float64) -> np.ndarray:
    '''
    The pair of lowest lattice harmonics f_1,k , f_2,k which are smooth and
    periodic on the Brillouin zone, real-valued, and are chosen is such a way
    that g_k in ham_nematic is constrained by the symmetries from Mathias' notes.

    Parameters
    ----------
    k_x : np.float64
        lattice vector out of BZ for x direction.
    k_y : np.float64
        lattice vector out of BZ for y direction.

    Returns
    -------
    f_k : np.ndarray
        The pair of lowest lattice harmonics f_1,k , f_2,k.

    '''
    f_1 = (8. / 3.) * (np.cos(k_y) - np.cos(0.5 * np.sqrt(3) * k_x) * np.cos(0.5 * k_y))
    f_2 = (8. / 3.) * np.sqrt(3) * np.sin(0.5 * np.sqrt(3) * k_x) * np.sin(0.5 * k_y)

    return np.array([f_1, f_2])


class Model:
    '''
    A class to implement a toy model of twisted bilayer graphene.

    Attributes
    ----------
    full : bool
        If True, full model with obstruction and therefore four bands (ham_full())
        will be evaluated. This will be used for STM image generation.
        If False, Haldane model with two bands (hal_ham()) will be evaluated.
    nematic : bool
        If True, nematic order will be added to the hamiltonian via the method
        (ham_nematic()).
    a_1 : np.ndarray
        first primitive lattice vector.
    a_2 : np.ndarray
        second primitive lattice vector.
    b_1 : np.ndarray
        first reciprocal lattice vector.
    b_2 : np.ndarray
        second reciprocal lattice vector.
    R : np.ndarray
        set of discrete bravais translation vectors generated with bravais_lattice().
    K : np.ndarray
        set of reciprocal translation vectors generated with reciprocal_lattice().
    k_brill: np.ndarray
        set of k points in first BZ generated with k_bz(). Not used! Instead see
        k_kar below.
    k_kar : np.ndarray
        set of k points according to Born von Karman boundary conditions generated
        with k_karman().
    alpha : list
        list of nematic order parameters.
    theta : float
        angle of phi for nematic coupling.

    Critical points:
    g_p : np.ndarray
        center of the Brillouin zone
    K_p : np.ndarray
        middle of an edge joining two rectangular faces
    M_p : np.ndarray
        center of a rectangular face

    Methods
    -------
    bz_plot() :
        Plots the 1st BZ, the reciprocal lattice points and symmetry points with
        the irreducible Brillouin zone.
    bravais_lattice() :
        Generates a set of discrete translations for the corresponding
        Bravais lattice.
    bravais_plot() :
        Plots the Bravais lattice R from bravais_lattice().
    grid_plot() :
        Generates the hexagonal grid from the Bravais lattice vectors R using
        a basis delta.
    reciprocal_lattice() :
        Generates a set of reciprocal vectors K.
    reciprocal_plot() :
        Plots the reciprocal lattice K.
    k_bz() :
        This generates a set of k points located in the first BZ.
    k_bz_plot() :
        Plots the points generated with the k_bz method.
    k_karman() :
        This generates a set of k points according to Born von Karman boundary
        conditions.
    k_karman_plot() :
        Plots the points generated with the k_karman method.
    hal_ham() : tuple((np.ndarray, np.ndarray))
        Gives the Haldane Hamiltonian for two orbitals and its eigenvalues.
    ham_full() : tuple((np.ndarray, np.ndarray))
        Gives the full Hamiltonian for four orbitals and its eigenvalues.
    ham_nematic() : tuple((np.ndarray, np.ndarray))
        Gives the Hamiltonian coupled to nematic order and its eigenvalues.
    band_plot() :
        Plots the energy bands for the chosen Hamiltonian.
    bz_contour() :
        Plotting the energy contour of the lowest band.
    f_k_contour() :
        Plotting the energy contour of the f_k function.
    '''

    def __init__(self, a_1=np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)]),
                       a_2=np.array([np.cos(np.pi / 6), -np.sin(np.pi / 6)]),
                       full=True, nematic=True):
        '''
        Constructs the attributes for model class.

        Parameters
        ----------
        a_1 : np.ndarray
            first primitive lattice vector.
        a_2 : np.ndarray
            second primitive attice vector.
        full : bool
            If True, full model with obstruction and therefore four bands (ham_full())
            will be evaluated. This will be used for STM image generation.
            If False, Haldane model with two bands (hal_ham()) will be evaluated.
        nematic : bool
            If True, nematic order will be added to the hamiltonian via the method
            (ham_nematic()).
        '''
        self.a_1 = a_1
        self.a_2 = a_2
        self.full = full
        self.nematic = nematic

        #rotation matrix to set up reciprocal lattice vectors
        rot = np.array([[np.cos(np.pi / 2), np.sin(np.pi / 2)],
                        [-np.sin(np.pi / 2), np.cos(np.pi / 2)]])

        self.b_1 = 2 * np.pi * np.dot(rot, self.a_2) / (np.dot(self.a_1, np.dot(rot, self.a_2)))
        self.b_2 = 2 * np.pi * np.dot(rot, self.a_1) / (np.dot(self.a_2, np.dot(rot, self.a_1)))

        direction = np.add(self.b_1, np.add(self.b_1, self.b_2))
        self.g_p = np.array([0, 0])
        self.K_p = LA.norm(self.b_1) / (2 * np.cos(np.pi / 6)) * direction / LA.norm(direction)
        self.M_p = np.array([self.K_p[0], 0])

        self.R = []
        self.K = []
        self.k_brill = []
        self.k_kar = []
        self.alpha = []
        self.theta = None

    def bz_plot(self) -> None:
        '''
        Plots the 1st BZ, the reciprocal lattice points and symmetry points with
        the irreducible Brillouin zone.
        '''
        for n in range(6):
            #rotate K point by n*pi/3 and (n+1)pi/3 to construct lines of BZ
            x_1 = np.dot(np.array([[np.cos(np.pi * n / 3), np.sin(np.pi * n / 3)],
                                   [-np.sin(np.pi * n / 3), np.cos(np.pi * n / 3)]]), self.K_p)
            x_2 = np.dot(np.array([[np.cos(np.pi * (n+1) / 3), np.sin(np.pi * (n+1) / 3)],
                                   [-np.sin(np.pi * (n+1) / 3), np.cos(np.pi * (n+1) / 3)]]), self.K_p)

            plt.scatter(x_1[0],x_1[1], color=['k'])
            plt.plot([x_1[0], x_2[0]],[x_1[1], x_2[1]], 'k-')

        plt.scatter(self.g_p[0], self.g_p[1], color=['k'])
        plt.scatter(self.K_p[0], self.K_p[1], color=['k'])
        plt.scatter(self.M_p[0], self.M_p[1], color=['k'])
        labels = [r'$\Gamma$', 'K', 'M']
        plt.annotate(labels[0], (self.g_p[0], self.g_p[1]+0.2))
        plt.annotate(labels[1], (self.K_p[0], self.K_p[1]+0.2))
        plt.annotate(labels[2], (self.M_p[0], self.M_p[1]+0.2))
        plt.plot([self.g_p[0], self.K_p[0]], [self.g_p[1], self.K_p[1]], 'k')
        plt.plot([self.g_p[0], self.M_p[0]], [self.g_p[1], self.M_p[1]], 'k')
        plt.plot([self.M_p[0], self.K_p[0]], [self.M_p[1], self.K_p[1]], 'k')
        plt.gca().set_aspect('equal')
        # plt.show()

    def bravais_lattice(self, R_num: list, fixed=False) -> None:
        '''
        Generates a set of translation vectors for the corresponding
        Bravais lattice.

        Parameters
        ----------
        R_num : int
            number of Bravais lattice vectors.
        fixed: bool
            True: Setting a fix set of lattice vectors in correspondence with
            the prefered image size of the stm plot.
            False: The number of lattice vectors agrees with R_num.
        '''
        if fixed:
            self.R = np.array([[-0.8660254   ,-0.5      ],
                               [ 0.          ,-1.       ],
                               [-0.8660254   ,0.5       ],
                               [ 0.          ,0.        ],
                               [ -0.8660254  ,-1.5      ],
                               [ 0.8660254   ,-0.5      ],
                               [ 0.8660254   ,-1.5      ],
                               [ 0.          ,1.        ],
                               [ 0.8660254   ,0.5       ]])
        else:
            n_max = int(np.sqrt(R_num) / 2.)
            n = []
            for i in range(-n_max-1, n_max+2):
                for j in range(-n_max-1, n_max+2):
                    n.append([i, j])

            for i in range(len(n)):
                self.R.append(n[i][0] * self.a_1 + n[i][1] * self.a_2)

            index = 0
            while len(self.R) > R_num:
                x = 0.
                for i in range(len(self.R)):
                    r = np.sqrt(self.R[i][0] ** 2 + self.R[i][1] ** 2)
                    if r > x:
                        x = r
                        index = i
                del self.R[index]

            self.R = np.array(self.R)

    def bravais_plot(self) -> None:
        '''
        Plots the Bravais lattice R from bravais_lattice().
        '''
        plt.scatter(self.R.T[0], self.R.T[1], color='green', marker='x')
        plt.gca().set_aspect('equal')
        #plt.show()

    def grid_plot(self) -> None:
        '''
        Generates the hexagonal grid from the Bravais lattice vectors R using
        the basis delta.
        '''
        #plotting a hexagonal cell
        # hex_centers, _ = create_hex_grid(n=1,
        #                                   rotate_deg=90,
        #                                   do_plot=True)

        #three different atom basises, where only the last two are in accordance
        #with the definition of our model
        # delta_A = np.array([1./np.sqrt(3), 0])
        # delta_B = - delta_A
        # delta_A = 1 / np.sqrt(3) * np.array([1. / 2., np.sqrt(3) / 2.])
        # delta_B = 1 / np.sqrt(3) * np.array([-1. / 2., np.sqrt(3) / 2.])
        delta_A = np.array([1. / np.sqrt(3) / 2., 0])
        delta_B = - delta_A

        grid_A = []
        grid_B = []
        for vector in self.R:
            grid_A.append(vector + delta_A)
            grid_B.append(vector + delta_B)
        grid_A = np.array(grid_A)
        grid_B = np.array(grid_B)

        plt.scatter(grid_A.T[0], grid_A.T[1], color='blue', marker='o', facecolors='none')
        plt.scatter(grid_B.T[0], grid_B.T[1], color='red', marker='o', facecolors='none')

        #plot primitive basis vectors
        # V = np.array([self.a_1, self.a_2])
        # origin = np.array([[0, 0],[0, 0]]) # origin point
        # plt.quiver(*origin, V[:,0], V[:,1], color=['g','g'],
        #            angles='xy', scale_units='xy', scale=1)

        #plot atom basis
        V = np.array([delta_A, delta_B])
        origin = np.array([[0, 0],[0, 0]]) # origin point
        plt.quiver(*origin, V[:,0], V[:,1], color=['b','r'], angles='xy', scale_units='xy', scale=1)

        plt.gca().set_aspect('equal')
        # lim = 0.65
        # plt.xlim(-lim,lim)
        # plt.ylim(-lim,lim)

    def reciprocal_lattice(self, K_num) -> None:
        '''
        Generates a set of reciprocal vectors K.

        Parameters
        ----------
        K_num : int
            number of reciprocal lattice points.
        '''
        n_max = int(np.sqrt(K_num) / 2.)
        n = []
        for i in range(-n_max-1, n_max+2):
            for j in range(-n_max-1, n_max+2):
                n.append([i, j])
        for i in range(len(n)):
            self.K.append(n[i][0] * self.b_1 + n[i][1] * self.b_2)

        index = 0
        while len(self.K) > K_num:
            x = 0.
            for i in range(len(self.K)):
                r = np.sqrt(self.K[i][0] ** 2 + self.K[i][1] ** 2)
                if r > x:
                    x = r
                    index = i
            del self.K[index]

        self.K = np.array(self.K)

    def reciprocal_plot(self) -> None:
        '''
        Plots the reciprocal lattice K.
        '''
        plt.scatter(self.K.T[0], self.K.T[1], color='blue')
        #plt.show()

    def k_bz(self, k_num: int) -> None:
        '''
        This generates a set of k points located in the first BZ.

        Parameters
        ----------
        k_num : int
            The total amount of points will be about k_num^2.
        '''
        #those are the outermost edges of the BZ hexagon.
        k_p = np.array([np.linspace(-3.62759873, 3.62759873, k_num),
                        np.linspace(-4.1887902, 4.1887902, k_num)]).T
        #generate a meshgrid
        k = []
        for x in k_p:
            for y in k_p:
                k.append([x[0], y[1]])
        k = np.array(k).reshape(k_num ** 2, 2)
        #this constructs the hexagon to check whether points are inside of it
        polygon = Polygon([(3.62759873, 2.0943951),
                           (3.62759873, -2.0943951),
                           (1.55431223e-15, -4.18879020e+00),
                           (-3.62759873, -2.0943951 ),
                           (-3.62759873, 2.0943951 ),
                           (0., 4.1887902)])
        #check if points from meshgrid lie within BZ
        for p in k:
            point = Point(p)
            if polygon.contains(point):
                self.k_brill.append(p)

        self.k_brill = np.array(self.k_brill)

    def k_bz_plot(self) -> None:
        '''
        Plots the points generated with the k_bz method.
        '''
        plt.scatter(self.k_brill.T[0], self.k_brill.T[1])
        self.bz_plot()
        plt.show()

    def k_karman(self, k_num: int) -> None:
        '''
        This generates a set of k points according to Born von Karman boundary
        conditions.

        Parameters
        ----------
        k_num : int
            The total amount of points will be k_num^2.
        '''
        m = np.arange(0., k_num, 1)
        for i in range(len(m)):
            for j in range(len(m)):
                k = (m[i] / k_num) * self.b_1 + (m[j] / k_num) * self.b_2
                self.k_kar.append(k)

        self.k_kar = np.array(self.k_kar)

    def k_karman_plot(self) -> None:
        '''
        Plots the points generated with the k_karman method.
        '''
        b_all = np.array([self.b_1, self.b_2])
        plt.scatter(b_all[:,0], b_all[:,1], color=['r'])
        plt.scatter(-b_all[:,0], -b_all[:,1], color=['r'])
        plt.scatter((self.b_1 + self.b_2)[0], (self.b_1 + self.b_2)[1], color=['r'])
        plt.scatter(-(self.b_1 + self.b_2)[0], (self.b_1 + self.b_2)[1], color=['r'])

        plt.scatter(self.k_kar.T[0], self.k_kar.T[1])

        V = np.array([self.b_1, self.b_2])
        origin = np.array([[0, 0],[0, 0]]) # origin point
        plt.quiver(*origin, V[:,0], V[:,1], color=['r','r'], angles='xy', scale_units='xy', scale=1)
        plt.annotate('b1', (self.b_1[0]*0.75+0.3, self.b_1[1]*0.75), color='red')
        plt.annotate('b2', (self.b_2[0]*0.75, self.b_2[1]*0.75+0.3), color='red')

        self.bz_plot()
        plt.show()

    def hal_ham(self, k_x: np.float64, k_y: np.float64,
                coupling=False) -> tuple((np.ndarray, np.ndarray)):
        '''
        Gives the Haldane Hamiltonian for two orbitals and its eigenvalues.

        Parameters
        ----------
        k_x : np.float64
            k vector component out of BZ in x direction.
        k_y : np.float64
            k vector component out of BZ in y direction.
        coupling : bool
            If one takes two thetaC2-related copies of the model and couple them
            one needs a Hamiltonian where alpha_2 -> - alpha_2
            and delta -> - delta. This corresponds to the full hamiltonian.
            By setting True one can construct a Hamiltonian h_2 in the ham_full
            method where these properties hold.

        Returns
        -------
        h : np.ndarray
            the Hamiltonian for the generalized Haldane system.
        e_v : np.ndarray
            the eigenvalues of h.
        '''
        delta = 0
        t_2 = 0.2
        t_3 = -0.1
        alpha_2 = 0.5 * np.pi
        alpha_3 = 0.3 * np.pi

        # by looking at the 4 band model one needs different parameters
        if self.full:
            delta = 0.
            t_2 = 0.6
            t_3 = 0.1
            alpha_2 = 0.3 * np.pi
            alpha_3 = 0.6 * np.pi
            # taking the copies for the coupling
            if coupling:
                delta = -0.
                alpha_2 = -0.3 * np.pi

        h = np.array([
            [delta + t_2 * (np.cos(0.5 * (np.sqrt(3) * k_x - k_y - 2 * alpha_2))
                            + np.cos(k_y - alpha_2)
                            + np.cos(0.5 * (np.sqrt(3) * k_x + k_y + 2 * alpha_2))),

             np.exp(-complex(0, k_y + alpha_3)) * (np.exp(complex(0, k_y + alpha_3))
                                                  + np.exp(0.5
                                                           * complex(0, (np.sqrt(3)
                                                           * k_x + k_y + 2 * alpha_3)))
                                                  * (1 + np.exp(complex(0, k_y)))
                                                  + t_3
                                                  + t_3 * np.exp(complex(0, 2 * k_y))
                                                  + t_3 * np.exp(complex(0, np.sqrt(3) * k_x + k_y)))],

             [1 + np.exp(- 0.5 * complex(0, np.sqrt(3) * k_x + k_y))
              * (1 + np.exp(complex(0, k_y)))
              + t_3 * np.exp(complex(0, alpha_3))
              * (np.cos(np.sqrt(3) * k_x)
                 + 2 * np.cos(k_y) - complex(0, np.sin(np.sqrt(3) * k_x))),

              -delta + t_2 * (np.cos(0.5 * (np.sqrt(3) * k_x + k_y - 2 * alpha_2))
                             + np.cos(k_y + alpha_2)
                             + np.cos(0.5 * (np.sqrt(3) * k_x - k_y + 2 * alpha_2)))]])

        e_v = np.sort(LA.eigvals(h))

        return h, e_v

    def ham_full(self, k_x: np.float64, k_y: np.float64) -> tuple((np.ndarray, np.ndarray)):
        '''
        Gives the full Hamiltonian for four orbitals and its eigenvalues.
        We take two thetaC2-related copies of the haldane model
        (corresponds to the "two orbitals") and couple them.

        Parameters
        ----------
        k_x : np.float64
            k vector component out of BZ in x direction.
        k_y : np.float64
            k vector component out of BZ in y direction.

        Returns
        -------
        h : np.ndarray
            the Hamiltonian for the full four band system.
        e_v : np.ndarray
            the eigenvalues of h.
        '''
        w_1 = 0.6
        w_2 = -0.5
        deltaw = 0.

        h_2 = self.hal_ham(k_x, k_y, coupling=True)

        nnf_pos = 1 + 2 * np.exp(0.5 * complex(0, np.sqrt(3) * k_x)) * np.cos(k_y / 2.)
        nnf_neg = 1 + 2 * np.exp(0.5 * complex(0, np.sqrt(3) * (-k_x))) * np.cos(k_y / 2.)

        h = np.array([
            [self.hal_ham(k_x, k_y, False)[0][0][0],
             self.hal_ham(k_x, k_y, False)[0][0][1],
             w_1 * np.exp(complex(0, deltaw)),
             w_2 * nnf_pos],

             [self.hal_ham(k_x, k_y, False)[0][1][0],
              self.hal_ham(k_x, k_y, False)[0][1][1],
              w_2 * nnf_neg,
              w_1 * np.exp(complex(0, deltaw))],

             [w_1 * np.exp(complex(0, -deltaw)),
              w_2 * nnf_pos,
              h_2[0][0][0],
              h_2[0][0][1]],

             [w_2 * nnf_neg,
              w_1 * np.exp(complex(0, -deltaw)),
              h_2[0][1][0],
              h_2[0][1][1]]])

        e_v = np.sort(LA.eigvals(h))

        return h, e_v

    def ham_nematic(self, k_x: np.float64, k_y: np.float64) -> tuple((np.ndarray, np.ndarray)):
        '''
        Gives the Hamiltonian coupled to nematic order and its eigenvalues.

        Parameters
        ----------
        k_x : np.float64
            k vector component out of BZ in x direction.
        k_y : np.float64
            k vector component out of BZ in y direction.

        Returns
        -------
        h : np.ndarray
            the Hamiltonian for the full nematic system.
        e_v : np.ndarray
            the eigenvalues of h.
        '''
        #pauli matrices
        s0 = np.eye(2)
        s1 = np.array([[0.,1.],[1.,0.]])
        s2 = np.array([[0.,-1j],[1j,0.]])
        s3 = np.array([[1.,0.],[0.,-1.]])

        phi = np.array([np.cos(2 * self.theta), np.sin(2 * self.theta)])

        #define function g in H = h + phi * g
        g_1 = (self.alpha[0] * np.kron(s0, s0) * f_k(k_x, k_y)[0] +
               self.alpha[1] * np.kron(s1, s0) * f_k(k_x, k_y)[0] -
               self.alpha[2] * np.kron(s2, s0) * f_k(k_x, k_y)[1] -
               self.alpha[3] * np.kron(s3, s3) * f_k(k_x, k_y)[1])

        g_2 = (self.alpha[0] * np.kron(s0, s0) * f_k(k_x, k_y)[1] +
               self.alpha[1] * np.kron(s1, s0) * f_k(k_x, k_y)[1] +
               self.alpha[2] * np.kron(s2, s0) * f_k(k_x, k_y)[0] +
               self.alpha[3] * np.kron(s3, s3) * f_k(k_x, k_y)[0])

        if self.full:
            ham = self.ham_full(k_x, k_y)[0]
        else:
            ham = self.hal_ham(k_x, k_y)[0]

        h = ham + phi[0] * g_1 + phi[1] * g_2
        e_v = np.sort(LA.eigvals(h))

        return h, e_v

    def band_plot(self) -> None:
        '''
        Plots the energy bands for the chosen Hamiltonian.
        '''
        l_1 = LA.norm(self.g_p - self.K_p)
        l_2 = LA.norm(self.M_p - self.g_p)
        l_3 = LA.norm(self.K_p - self.M_p)

        grid = np.arange(0, 1., 0.01)

        k_1 = [(point * l_1) for point in grid]
        k_2 = [(l_1 + point * l_2) for point in grid]
        k_3 = [(l_1 + l_2 + point * l_3) for point in grid]

        e_v_1 = []
        e_v_2 = []
        e_v_3 = []

        if self.nematic:
            ham = self.ham_nematic
        elif self.full:
            ham = self.ham_full
        else:
            ham = self.hal_ham

        for point in grid:
            e_v = ham((self.K_p + (self.g_p - self.K_p) * point)[0],
                      (self.K_p + (self.g_p - self.K_p) * point)[1])[1].real
            e_v_1.append(e_v)
            e_v = ham((self.g_p + (self.M_p - self.g_p) * point)[0],
                      (self.g_p + (self.M_p - self.g_p) * point)[1])[1].real
            e_v_2.append(e_v)
            e_v = ham((self.M_p + (self.K_p - self.M_p) * point)[0],
                      (self.M_p + (self.K_p - self.M_p) * point)[1])[1].real
            e_v_3.append(e_v)

        plt.plot(k_1-LA.norm(self.K_p), e_v_1, 'k')
        plt.plot(k_2-LA.norm(self.K_p), e_v_2, 'k')
        plt.plot(k_3-LA.norm(self.K_p), e_v_3, 'k')
        plt.xticks(ticks = [(k_1-LA.norm(self.K_p))[0], self.g_p[0], self.M_p[0],
                            (k_3-LA.norm(self.K_p))[-1]],
                   labels = ['K', r'$\Gamma$', 'M', 'K'])
        # xmin, xmax, ymin, ymax = plt.axis()
        # plt.gca().set_yticks(np.arange(int(ymin), int(ymax+1), 4.))
        # plt.gca().set_yticks(np.arange(int(ymin), int(ymax+1), 1.), minor=True)
        plt.grid()
        plt.xlabel('k')
        plt.ylabel('Energy in eV')
        plt.show()

    def bz_contour(self) -> None:
        '''
        Plotting the energy contour of the lowest band.
        '''
        k_points = np.arange(-1.4 * np.pi, 1.4 * np.pi, 0.07)

        if self.nematic:
            ham = self.ham_nematic
        elif self.full:
            ham = self.ham_full
        else:
            ham = self.hal_ham

        contour = np.zeros((len(k_points), len(k_points)))

        print('\nCalculating contour plot for lowest band:\n')
        for x in tqdm(range(len(k_points)), position=0, leave=True):
            for y in range(len(k_points)):
                contour[x][y] = (ham(k_points[x], k_points[y])[1])[0].real

        plt.contourf(k_points, k_points, contour[:,::-1].T, levels=30)
        plt.colorbar()
        self.bz_plot()
        plt.show()

    def f_k_contour(self) -> None:
        '''
        Plotting the energy contour of the f_k function.
        '''
        X, Y = np.meshgrid(np.linspace(-1.4 * np.pi, 1.4 * np.pi, 1000),
                           np.linspace(-1.4 * np.pi, 1.4 * np.pi, 1000))
        Z1 = f_k(X, Y)[0]
        Z2 = f_k(X, Y)[1]

        # alpha = 0.
        # contour = np.cos(2 * (0.5 + alpha)) * Z1 + np.sin(2 * (0.5 + alpha)) * Z2

        plt.contourf(X, Y, Z1, levels=100)
        plt.colorbar()
        self.bz_plot()
        plt.show()
        plt.contourf(X, Y, Z2, levels=100)
        plt.colorbar()
        self.bz_plot()
        plt.show()
        # plt.contourf(X, Y, contour, levels=50)
        # plt.colorbar()
        # self.bz_plot()
        # plt.show()


def main(full: bool, nematic: bool, alpha: list, theta: float) -> None:
    '''
    Main function

    Parameters
    ----------
    full : bool
        If True, full model with four bands will be evaluated.
        If False, model with two bands will be evaluated.
    nematic : bool
        If True, full model with four bands coupled to nematic order parameter
        will be evaluated.
    alpha : list
        list of nematic order parameters.
    theta : float
        angle of phi for nematic coupling.
    '''
    model = Model(full=full, nematic=nematic)

    # model.bz_plot()
    # model.reciprocal_lattice(7)
    # model.reciprocal_plot()

    # model.bravais_lattice(19, False)
    # model.bravais_plot()
    # model.grid_plot()

    # model.k_bz(100)
    # model.k_bz_plot()

    # model.k_karman(100)
    # model.k_karman_plot()

    model.theta = theta
    model.alpha = alpha    
    
    model.band_plot()
    # model.bz_contour()

    # model.f_k_contour()

if __name__ == "__main__":

    cfg = {
           'full': True,
           'nematic': False,
           'alpha': [0.1, 0.2, 0.23, 0.18],
           # possible orientations for theta:
           # A = 0., np.pi / 3., 2 * np.pi / 3.
           # B = np.pi / 6., np.pi / 2., 5 * np.pi / 6.
           'theta': 0.,
           }

    main(
         cfg['full'],
         cfg['nematic'],
         cfg['alpha'],
         cfg['theta'],
         )
