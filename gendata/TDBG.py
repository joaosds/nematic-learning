# This code implements the Bistritzer-MacDonald continuum model for twisted double bilayer graphene Supports heterostrain and displacement field Can output cuts of the band structure, full DOS, and LDOS images See end of file for example code Simon Turkel References:

# [1] Bistritzer-MacDonald model: Bistritzer and MacDonald, PNAS 108 (30) 12233-12237 (2011). "Moire bands in twisted double-layer graphene."
# [2] TDBG model: Koshino, Phys. Rev. B 99, 235406 (2019). "Band structure and topological properties of twisted bilayer graphenes."
# [3] Heterostrain in continuum models: Bi, Yuan, and Fu, Phys. Rev. B 100, 035448 (2019). "Designing flat bands by strain."
# [4] Microscopic nematicity: Rubio-Verdú, C., Turkel, S., Song, Y. et al. Nat. Phys. 18, 196–202 (2022). "Moiré nematic phase in twisted double bilayer graphene."

# -----------------------------------------------------------------------------------------------

import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import pywt
import sys

# -----------------------------------------------------------------------------------------------
# Helper Functions


def ind_en(array: np.ndarray, val: float) -> int:

    """
    TD: Maybe add some try error here
    Function that returns index of element with certain value in array.

    Parameters
    ----------
    array : np.ndarray
        Vector with energies
    val : float
        energy value

    """
    j = np.where(array == val)

    # Check if first array from np.where is [] or some valid index

    if j[0].size:
        j = int(j[0])
    else:
        # print("")
        print(
            "Energy not found in array. Check if val and array are valid values, and be sure that val is in array."
        )
        # print(sys.stderr, "Energy not found in array. Check if val and array are valid values, and be sure that val is in array")
        # print(sys.stderr, "Exception: %s" % str(e))
        sys.exit(1)
    return j


# generic rotation matrix
def R(x):
    r = np.array(([np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]))
    return r


#  2x2 Identity matrix
I = np.identity(2)

# strain and rotation matrix
def Et(theta, theta_s, e, delta):
    # e - strain magnitude
    # delta - Poison ratio for graphene
    # theta_s - strain direction
    # Antisymmetric part of the strain rotation matrix E
    # theta - rotation angle between layers
    t = np.array(([0, -theta], [theta, 0]))
    r = R(-theta_s) @ np.array(([e, 0], [0, -delta * e])) @ R(theta_s) + t
    return r


#  -----------------------------------------------------------------------------------------------

# Create the continuum model class
def wavelet_trafo(ldos: list) -> np.ndarray:
    """
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

    """
    coef, _ = pywt.cwt(ldos, np.arange(1, 66), "morl")
    # cmap = plt.cm.coolwarm.copy()
    # plt.imshow(
    #     coef,
    #     cmap=cmap,
    #     interpolation="none",
    #     extent=[1, 65, 1, 65],
    #     aspect="auto",
    #     # vmax=abs(coef).max(),
    #     # vmin=-abs(coef).max(),
    # )
    # plt.colorbar()
    # plt.show()

    return coef


class Model:
    def __init__(
        self,
        theta,
        phi,
        epsilon,
        D,  # Empirical parameters (must provide as input)
        a=0.246,
        varphiMN=0,
        varphiIAGN=0,
        PhiMN=0.005,
        PhiIAGN=0.02,
        alphal=0,
        beta=3.14,
        delta=0.16,  # Graphene parameters
        vf=1,
        u=0.0797,
        up=0.0975,
        cut=4,
    ):  # Continuum model parameters
        """
        Constructs the attributes for model class (tDBG). hbar = 1.

        Parameters
        ----------
        theta: float
            Twist angle between layers l=2-3 (l=1-2 & l=3-4 are Bernal stacks).
            (rad).
        phi: float
            Strain direction (rad).
        e: float
            Strain percentage.
        D: np.ndarray
            Displacement field (eV) - array with 4 entries).
        beta: float
            Dimensionless parameter for gauge connection [3].
        a: float
            Lattice constant (nm).
        nu: float
            Poison ratio for graphene [3].
        vf: float
            Fermi velocity factor. For hbar*nu/a = 2.776eV set vf = 1.3.
        v: float
            Rescaled band velocity (eV).
        gamma(3,4): float
            Diagonal hopping elements in Bernal stacks (eV).
        v3: float
            Trigonal warping of the energy band in Bernal stacks(eV).
        v4: float
            Electron-hole asymmetry in Bernal stacks (eV).
        gamma1: float
            Coupling between dimer sites in Bernal stacks (eV).
        Dp: float
            Onsite potential at dimer sites in Bernal stacks (eV).
        u: float
            Diagonal amplitude in moiré interlayer hopping matrix (eV).
            See eq. (4) [2].
        up: float
            Off-diagonal amplitude in moiré interlayer hopping matrix (eV).
            See eq. (4) [2].
        omega: float
            Phase factor omega of 2*np.pi/3 in eq. (4) [2].
        mu: float
            Chemical potential (eV).
        NPhi: float
            Nematicity magnitude (eV).
        Nvarphi: float
            Orientation of nematic director (rad).
        Nemat: str
            Choice of nematicity: NM or IAGN.
        NematOD: np.ndarray
            Nematic order parameter.
        Nalphal: float
            Intravalley graphene nematicity layer-rotation angle (rad).
            See eq. (11) [4].
        """
        # Convert angles from degrees to radians
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180

        # Empirical parameters
        self.theta = theta  # twist angle
        self.phi = phi  # strain angle
        self.epsilon = epsilon  # strain percent
        self.D = D  # displacement field

        # Nematicity parameters Intravalley Graphene nematicity (IAGN)
        self.PhiIAGN = PhiIAGN  # Magnitude
        self.alphal = alphal  # Layer-rotation angle
        self.varphiIAGN = varphiIAGN  # Angle of director
        self.nemaIAGN = self.PhiIAGN * np.array(
            [np.cos(2 * self.varphiIAGN), np.sin(2 * self.varphiIAGN)]
        )

        # Nematicity parameters Moiré nematicity (MN)
        self.PhiMN = PhiMN  # Magnitude
        self.varphiMN = varphiMN  # Angle of director
        self.nemaMN = self.PhiMN * np.array(
            [np.cos(2 * self.varphiMN), np.sin(2 * self.varphiMN)]
        )

        # Filling fraction nu = 0.475
        self.mu = -15 / 1000

        # Graphene parameter.a = a  # lattice constant
        self.beta = beta  # two center hopping modulus
        self.delta = delta  # poisson ratio
        self.A = np.sqrt(3) * self.beta / 2 / a  # gauge connection

        # Continuum model parameters (hbar*v)
        self.v = vf * 2.1354 * a  # vf = 1.3 used in publications
        self.v3 = np.sqrt(3) * a * 0.32 / 2
        self.v4 = np.sqrt(3) * a * 0.044 / 2
        self.gamma1 = 0.4
        self.Dp = 0.05

        self.omega = np.exp(1j * 2 * np.pi / 3)
        self.u = u
        self.up = up

        # Define the graphene lattice in momentum space
        k_d = 4 * np.pi / 3 / a
        # k1 is just the position of the first Dirac point in the lattice
        k1 = np.array([k_d, 0])
        k2 = np.array([np.cos(2 * np.pi / 3) * k_d, np.sin(2 * np.pi / 3) * k_d])
        k3 = -np.array([np.cos(np.pi / 3) * k_d, np.sin(np.pi / 3) * k_d])

        # Generate the strained moire reciprocal lattice vectors
        q1 = Et(theta, phi, epsilon, delta) @ k1
        q2 = Et(theta, phi, epsilon, delta) @ k2
        q3 = Et(theta, phi, epsilon, delta) @ k3
        q = np.array([q1, q2, q3])  # put them all in a single array
        self.q = q
        k_theta = np.max(
            [norm(q1), norm(q2), norm(q3)]
        )  # used to define the momentum space cutoff
        self.k_theta = k_theta
        # "the Q lattice" refers to a lattice of monolayer K points on which the continuum model is defined
        # basis vectors for the Q lattice
        b1 = q[1] - q[2]
        b2 = q[0] - q[2]
        b3 = q[1] - q[0]
        b = np.array([b1, b2, b3])
        self.b = b

        # print(f"Strained with e={epsilon} and phi={phi}")
        b1t = self.q[0] / norm(self.q[0])
        b2t = self.q[1] / norm(self.q[1])
        b3t = self.q[2] / norm(self.q[2])
        # print(b3t, b2t)
        # print("inverse")
        self.a2m = (
            2
            * np.pi
            * np.dot(R(np.pi / 2), self.b[2])
            / (np.dot(self.b[1], np.dot(R(np.pi / 2), self.b[2])))
        )
        self.a1m = (
            2
            * np.pi
            * np.dot(R(np.pi / 2), self.b[1])
            / (np.dot(self.b[2], np.dot(R(np.pi / 2), self.b[1])))
        )
        # self.a1m = self.a1m / norm(self.a1m)
        # self.a2m = self.a2m / norm(self.a2m)
        self.a1m = (
            np.linalg.inv(np.transpose(Et(theta, phi, epsilon, delta))) @ self.a1m
        )
        self.a2m = (
            np.linalg.inv(np.transpose(Et(theta, phi, epsilon, delta))) @ self.a2m
        )

        self.a1m = self.a1m / norm(self.a1m)
        self.a2m = self.a2m / norm(self.a2m)
        # print(self.a1m, self.a2m)
        # generate the Q lattice
        # i, j - m1,m2 - k
        # l - layer index
        # [i, j, 0] @ b - l * q[0] = norm condition
        limk = int(cut + 1)
        self.limk = limk
        Q = np.array(
            [
                np.array(list([i, j, 0] @ b - l * q[0]) + [l])
                for i in range(-limk, limk)
                for j in range(-limk, limk)
                for l in [0, 1]
                if norm([i, j, 0] @ b - l * q[0]) <= np.sqrt(3) * self.k_theta * cut
            ]
        )

        self.Q = Q
        Nq = len(Q)
        self.Nq = Nq
        # print(Q[:, :2]) - All lines, two first columns
        # nearest neighbors on the Q lattice
        # np.round(Q[:, :2], 3) - round to 3 decimals
        # ----------------
        # Only neighbors inside the cutoff
        # if list(np.round(Q[i, :2] + q[j], 3)) in np.round(Q[:, :2], 3).tolist()

        # K vectors are connected by the reciprocal lattice vectors q
        self.Q_nn = {}
        for i in range(Nq):
            self.Q_nn[i] = [
                [
                    np.round(Q[:, :2], 3)
                    .tolist()
                    .index(list(np.round(Q[i, :2] + q[j], 3))),
                    j,
                ]
                for j in range(len(q))
                if list(np.round(Q[i, :2] + q[j], 3)) in np.round(Q[:, :2], 3).tolist()
            ]

        # There's an offset in kd for physical values from 2nd layer
        Q2G = np.array([[l, l] for l in Q[:, 2]]) * q[0] + Q[:, :2]
        # Expanding over the Dirac points q[0] / Simon is calculating this w.r.t K on X [kd,0]
        # # Just from 2nd layer
        self.G = Q2G[Q[:, 2] == 1]

    # A function to create the hamiltonian for a given point kx, ky
    # @jit(nopython=True)
    def gen_ham(self, kx, ky, xi=1):
        k = np.array([kx, ky])  # 2d momentum vector

        # Create moire hopping matrices for valley index xi
        U1 = np.array(([self.u, self.up], [self.up, self.u]))

        U2 = np.array(
            (
                [self.u, self.up * self.omega ** (-xi)],
                [self.up * self.omega ** (xi), self.u],
            )
        )

        U3 = np.array(
            (
                [self.u, self.up * self.omega ** (xi)],
                [self.up * self.omega ** (-xi), self.u],
            )
        )

        # Create and populate Hamiltonian matrix
        ham = np.matrix(np.zeros((4 * self.Nq, 4 * self.Nq), dtype=complex))

        # IAGN
        # See equation (11) of ref. [4]
        unit = np.array([1, -1j * xi]) * np.exp(1j * self.alphal * xi)
        self.IAGN = np.dot(self.nemaIAGN, unit)

        for i in range(self.Nq):
            # layer index l = 1,2
            t = self.Q[i, 2]
            l = np.sign(2 * t - 1)
            M = Et(
                l * xi * self.theta / 2, self.phi, l * xi * self.epsilon / 2, self.delta
            )
            E = (M + M.T) / 2
            # E is just a definition to obtain the the vectors exx, exy
            exx = E[0, 0]
            eyy = E[1, 1]
            exy = E[0, 1]
            # See equation (5) of ref [3]
            kj = (I + M) @ (
                k + self.Q[i, :2] + xi * self.A * np.array([exx - eyy, -2 * exy])
            )

            # Attention: kj=kmoire only without heterostrain
            # TODO: modify this for heterostrain
            kmoire = k + self.Q[i, :2]
            km = xi * kj[0] - 1j * kj[1]
            kp = xi * kj[0] + 1j * kj[1]

            # Moiré Nematicity
            # See equation (5-7) of ref. [4]
            # f1 = np.cos(kmoire[0])
            # f2 = np.cos(kmoire[1] * np.sqrt(3) * 0.5) * np.cos(kmoire[0] * 0.5)
            # f3 = np.sin(kmoire[0] * 0.5) * np.sin(kmoire[1] * np.sqrt(3) * 0.5)
            # fk = np.array([f1 - f2, -f3 * np.sqrt(3)]) * 8.0 / 3.0
            # self.MN = np.dot(self.nemaMN, fk)

            f1 = -(
                np.cos(np.dot(self.a1m, kmoire))
                + np.cos(np.dot(self.a2m, kmoire))
                - 2 * np.cos(np.dot((self.a1m - self.a2m), kmoire))
            )
            f2 = np.sqrt(3.0) * (
                np.cos(np.dot(self.a2m, kmoire)) - np.cos(np.dot(self.a1m, kmoire))
            )
            fk = 4.0 * np.array([f1, f2]) / 3.0
            self.MN = np.dot(self.nemaMN, fk)

            #             self.MN = 0
            # self.D1 = (3 / 4 - t) * self.D
            # self.D2 = (3 / 4 - t) * self.D
            # self.D3 = (1 / 4 - t) * self.D
            # self.D4 = (1 / 4 - t) * self.D
            #
            # Self-consistently calculated screened electric field
            self.D1 = 0.004079
            self.D2 = 0.001021
            self.D3 = -0.001537
            self.D4 = -0.003563
            # self.D1 = 0
            # self.D2 = 0
            # self.D3 = 0
            # self.D4 = 0

            # Populate diagonal blocks
            # Upper triangle hamiltonian (that's why there's a 1/2 factor
            # for diagonal elements)
            ham[4 * i : 4 * i + 4, 4 * i : 4 * i + 4] = np.array(
                (
                    [
                        (self.mu + self.D1 + self.MN) / 2,
                        -self.v * km + self.IAGN,
                        self.v4 * km,
                        self.v3 * kp,
                    ],
                    [
                        0,
                        (self.Dp + self.D2 + self.mu + self.MN) / 2,
                        self.gamma1,
                        self.v4 * km,
                    ],
                    [
                        0,
                        0,
                        (self.Dp + self.D3 + self.mu + self.MN) / 2,
                        -self.v * km + self.IAGN,
                    ],
                    [0, 0, 0, (self.D4 + self.mu + self.MN) / 2],
                )
            )

            # Populate off-diagonal blocks
            nn = self.Q_nn[i]
            for neighbor in nn:
                j = neighbor[0]  # index of neighbor
                p = neighbor[1]  # neighbor number p (0, 1 or 2)
                ham[4 * j + 2 : 4 * j + 4, 4 * i : 4 * i + 2] = (
                    (p == 0) * U1 + (p == 1) * U2 + (p == 2) * U3
                )
                # ham[4 * j + 2 : 4 * j + 4, 4 * i : 4 * i + 2] = (
                #     (p == 0) * 10 + (p == 1) * 20 + (p == 2) * 40
                # )

        # plt.imshow(np.real(ham + ham.H), cmap = plt.cm.Spectral)
        # plt.show()

        # return ham + ham.H + np.eye(4*self.Nq)*self.Phi*self.MN
        return ham + ham.H

    # A function to solve for the bands along the path: K -> Gamma -> M -> K'
    def solve_along_path(
        self, res=30, plot_it=True, return_eigenvectors=False
    ):  # res = number of points per unit length in k space
        l1 = int(res)  # K->Gamma
        l2 = int(np.sqrt(3) * res / 2)  # Gamma->M
        l3 = int(res / 2)  # M->K'

        kpath = []  # K -> Gamma -> M -> K'
        for i in np.linspace(0, 1, l1):
            kpath.append(i * (self.q[0] + self.q[1]))  # K->Gamma
        for i in np.linspace(0, 1, l2):
            kpath.append(
                self.q[0] + self.q[1] + i * (-self.q[0] / 2 - self.q[1])
            )  # Gamma->M
        for i in np.linspace(0, 1, l3):
            kpath.append(self.q[0] / 2 + i * self.q[0] / 2)  # M->K'

        evals_m = []
        evals_p = []
        if return_eigenvectors:
            evecs_m = []
            evecs_p = []

        # Optimize this section
        for kpt in kpath:  # for each kpt along the path
            ham_m = self.gen_ham(
                kpt[0], kpt[1], -1
            )  # generate and solve a hamiltonian for each valley
            ham_p = self.gen_ham(kpt[0], kpt[1], 1)

            val, vec = eigh(ham_m)
            evals_m.append(val)
            if return_eigenvectors:
                evecs_m.append(vec)

            val, vec = eigh(ham_p)
            evals_p.append(val)
            if return_eigenvectors:
                evecs_p.append(vec)

        evals_m = np.array(evals_m)
        evals_p = np.array(evals_p)
        dim = len(evals_m[1, :])
        color = iter(cmap.inferno(np.linspace(0, 1, dim)))
        if plot_it:
            plt.figure(1)
            plt.clf()
            for i in range(len(evals_m[1, :])):
                c = next(color)
                # c = "black"
                plt.plot(evals_m[:, i], linestyle="dashed")
                plt.plot(evals_p[:, i])

            plt.xticks([0, l1, l1 + l2, l1 + l2 + l3], ["K", r"$\Gamma$", "M", "K'"])
            plt.yticks(np.arange(-0.1, 0.1, 0.03))
            ax = plt.gca()
            ax.minorticks_on()
            ax.tick_params(axis="y", direction="in", which="major", bottom=False)
            ax.tick_params(axis="y", direction="in", which="minor", bottom=False)
            ax.tick_params(axis="x", direction="in", which="major")
            ax.tick_params(axis="x", direction="in", which="minor")
            plt.ylim(-0.1, 0.1)
            plt.xlim(0, l1 + l2 + l3)
            # plt.axhline(y=-0.085, color="gray", linestyle="-", linewidth=0.8)
            # plt.axhline(y=-0.066, color="gray", linestyle="-", linewidth=0.8)
            # plt.axhline(y=-0.037, color="gray", linestyle="-", linewidth=0.8)
            # plt.axhline(y=-0.015, color="gray", linestyle="-", linewidth=0.8)
            # plt.axhline(y=0.1, color="gray", linestyle="-", linewidth=0.8)
            # plt.axhline(y=0.024, color="gray", linestyle="-", linewidth=0.8)
            # plt.axhline(y=0.066, color="gray", linestyle="-", linewidth=0.8)
            # plt.axhline(y=0.077, color="gray", linestyle="-", linewidth=0.8)
            plt.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
            plt.axvline(x=l1, color="gray", linestyle="--", linewidth=0.5)
            plt.axvline(x=l1 + l2, color="gray", linestyle="--", linewidth=0.5)
            plt.axvline(x=l1 + l2 + l3, color="gray", linestyle="--", linewidth=0.5)
            plt.ylabel("Energy (eV)")
            # plt.tight_layout()
            # plt.savefig("fig3a)_2019koshino.pdf")
            # plt.show()

        if return_eigenvectors:
            evecs_m = np.array(evecs_m)
            evecs_p = np.array(evecs_p)
            return evals_m, evals_p, evecs_m, evecs_p, kpath

        else:
            return evals_m, evals_p, kpath

    def solve_PDOS(
        self,
        nk=53,
        energies=np.round(np.linspace(-150 / 1000, 150 / 1000, 65), 3),
        xi=1,
        sigma=0,
        px=65,
    ):

        # helper function to convert from dos dictionary (see below) to 1D vector
        def makePdos(D, orbs):
            sublattice = len(orbs)
            y = np.zeros((len(energies), sublattice))
            for o in range(sublattice):
                for j in range(len(energies)):
                    y[j, o] += D[orbs[o]].get(energies[j], 0)
            return np.sum(y, 1) / sublattice

        # Define a grid of k points
        kpts = np.array(
            [
                (i * self.b[0] - j * self.b[1])
                for i in np.linspace(0, 1, nk, endpoint=False)
                for j in np.linspace(0, 1, nk, endpoint=False)
            ]
        )
        # t is an array that has labels correspondent to the bilayer index L (initial L, change this latter) for each k value
        t = np.array(
            [
                val
                for pair in zip(self.Q[:, 2], self.Q[:, 2], self.Q[:, 2], self.Q[:, 2])
                for val in pair
            ]
        )

        # Bilayer index (bottom BL = -1, top BL = 1)
        BL = xi * (2 * t - 1)

        # Create masks for each sublattice/layer degree of freedom (to project the DOS)
        # looo - layer 1, ...

        # TU
        looo = np.array(self.Nq * [1, 0, 0, 0])
        oloo = np.array(self.Nq * [0, 1, 0, 0])
        oolo = np.array(self.Nq * [0, 0, 1, 0])
        oool = np.array(self.Nq * [0, 0, 0, 1])

        A1 = (BL == -1) * ((t == 0) * looo + (t == 1) * oool)
        B1 = (BL == -1) * ((t == 0) * oloo + (t == 1) * oolo)
        A2 = (BL == -1) * ((t == 0) * oolo + (t == 1) * oloo)
        B2 = (BL == -1) * ((t == 0) * oool + (t == 1) * looo)
        A3 = (BL == 1) * ((t == 0) * oool + (t == 1) * looo)
        B3 = (BL == 1) * ((t == 0) * oolo + (t == 1) * oloo)
        A4 = (BL == 1) * ((t == 0) * oloo + (t == 1) * oolo)
        B4 = (BL == 1) * ((t == 0) * looo + (t == 1) * oool)

        # Store masks in dictionary
        M = {
            "A1": A1,
            "B1": B1,
            "A2": A2,
            "B2": B2,
            "A3": A3,
            "B3": B3,
            "A4": A4,
            "B4": B4,
        }

        # Create dictionary to store partial DOS for each sublattice (A/B) and layer (1,2,3,4)
        dos = {
            "A1": {},
            "B1": {},
            "A2": {},
            "B2": {},
            "A3": {},
            "B3": {},
            "A4": {},
            "B4": {},
        }

        dos_sigma = dos
        # Solve model on grid of k points, storing the eigenvector amplitudes for each sublattice/layer
        for kpt in kpts:
            ham = self.gen_ham(kpt[0], kpt[1], xi)
            # vals, vecs = eigh(ham)
            vals, vecs = eigh(ham)

            for j in range(len(vals)):
                val = np.round(vals[j], 3)
                # selecting eigenvector correspondent to the Energy vals[j]
                # For each k=(kx,ky) a diagonalization gives N eigenvectors distributed linewise
                vec = np.array(vecs[:, j])

                for s in M:
                    # if val is not found returns 0 from get(val, 0)

                    dos[s][val] = dos[s].get(val, 0) + np.sum(
                        abs(vec[M[s] == 1, 0]) ** 2
                    )

                    dos_sigma[s][val] = dos[s].get(val, 0) + np.sum(
                        abs(vec[M[s] == 1, 0]) ** 2
                    )

        # create PDOS for each layer
        L1 = makePdos(dos, ["A1", "B1"])
        L2 = makePdos(dos, ["A2", "B2"])
        L3 = makePdos(dos, ["A3", "B3"])
        L4 = makePdos(dos, ["A4", "B4"])
        PDOS = [L1, L2, L3, L4, (L1 + L2 + L3 + L4) / 4]
        #
        L1_sigma = makePdos(dos_sigma, ["A1", "B1"])
        L2_sigma = makePdos(dos_sigma, ["A2", "B2"])
        L3_sigma = makePdos(dos_sigma, ["A3", "B3"])
        L4_sigma = makePdos(dos_sigma, ["A4", "B4"])
        PDOS_sigma = [
            L1_sigma,
            L2_sigma,
            L3_sigma,
            L4_sigma,
            (L1_sigma + L2_sigma + L3_sigma + L4_sigma) / 4,
        ]
        for j in range(5):
            for val in range(len(PDOS[0])):
                PDOS_sigma[j][val] = np.abs(np.random.normal(PDOS_sigma[j][val], 3.00))
        # return list of layer projected DOS and full DOS
        m = np.zeros((px, px, 5))
        m_sigma = np.zeros((px, px, 5))
        for i in range(5):
            m[:, :, i] = wavelet_trafo(PDOS[i])
            m_sigma[:, :, i] = wavelet_trafo(PDOS_sigma[i])
        return m, m_sigma

    # A function to solve for the local density of states
    # returns an array of 2D LDOS images at the specified energies
    # @jit(nopython=True)
    def solve_LDOS(
        self,
        nk=16,
        px=65,
        sz=30,
        l1=1,
        l2=1,
        # energies=np.round(np.linspace(-0.1, 0.1, 201), 3),
        energies=np.array([-15 / 1000]),
        xi=1,
    ):
        # nk^2 = momentum space grid size
        # px = number of real space pixels
        # sz = size of image in nm
        # l1/l2 = layer weights for outer and inner layers respectively

        # helper function to create 2D image from Fourier components
        def im(energy, px, eiGr, rho_G, G):
            gap = np.ones(len(G))
            amps = np.real(np.sum(rho_G.get(energy, gap) * eiGr.T, 1))
            amps = np.reshape(amps, (px, px))
            return amps

        energies = np.array(energies)
        # Energies from the 2nd Layer
        G = self.G

        # sublattice/layer masks
        t = np.array(
            [
                val
                for pair in zip(self.Q[:, 2], self.Q[:, 2], self.Q[:, 2], self.Q[:, 2])
                for val in pair
            ]
        )

        looo = np.array(self.Nq * [1, 0, 0, 0])
        oloo = np.array(self.Nq * [0, 1, 0, 0])
        oolo = np.array(self.Nq * [0, 0, 1, 0])
        oool = np.array(self.Nq * [0, 0, 0, 1])

        #
        kpts = np.array(
            [
                i * self.b[0] - j * self.b[1]
                for i in np.linspace(0, 1, nk, endpoint=False)
                for j in np.linspace(0, 1, nk, endpoint=False)
            ]
        )
        # l_theta = a/(2*sin(theta/2))
        l_theta = 4 * np.pi / self.k_theta / 3
        xhat = np.array([1, 0])
        yhat = np.array([0, 1])
        # One lattice is dislocated l_theta w.r.t. the other
        rpts = (
            np.array(
                [
                    # offset for better visualization of Moiré Unit cell
                    (-(40 - sz) / 2) * xhat
                    + (-(25 - sz) / 2) * yhat
                    + (-sz / 2 + i * sz / px) * xhat
                    + (-sz / 2 + j * sz / px) * yhat
                    for i in range(px)
                    for j in range(px)
                ]
            )
            - (l_theta / 8) * xhat
        )

        # Phase matrix
        # kj means the transpose of rpts is taken (just to multiply correctly)
        eiGr = np.exp(-1j * np.einsum("ij,kj", G, rpts))

        # Create a matrix mapping indices of g in G and g' in G to the index of g + g' in G
        GxGp = np.zeros((len(G), len(G)), dtype="int16")
        for g in range(len(G)):
            for gp in range(len(G)):
                if list(np.round(G[g] + G[gp], 3)) in np.round(G, 3).tolist():
                    GxGp[g, gp] = (
                        np.round(G, 3).tolist().index(list(np.round(G[g] + G[gp], 3)))
                    )
                else:
                    GxGp[g, gp] = -1

        # Create dictionary to store DOS
        dos = {}

        # Solve the model on a grid of kpts
        for kpt in kpts:
            ham = self.gen_ham(kpt[0], kpt[1], xi)
            values, vectors = eigh(ham)
            for j in range(len(values)):
                val = np.round(values[j], 3)
                vec = np.array(
                    vectors[:, j].flatten().tolist()[0] + [0]
                )  # add a zero on the end to mask -1 in GxGp
                if val in dos:
                    dos[val].append(vec)
                else:
                    dos[val] = [vec]

        # Generate fourier components of the LDOS
        rho_G = {}

        for val in dos:
            # If the eigenvalue is not on the desired energy, don't calculate the LDOS
            if val > np.max(energies) or val < np.min(energies):
                continue

            rho_G[val] = np.zeros(len(G), dtype="complex")

            psiA1 = np.array(dos[val])[:, np.append(oolo * t == 1, True)]
            psiB1 = np.array(dos[val])[:, np.append(oool * t == 1, True)]
            psiA2 = np.array(dos[val])[:, np.append(looo * t == 1, True)]
            psiB2 = np.array(dos[val])[:, np.append(oloo * t == 1, True)]

            rho_G[val] += l1 * (
                np.einsum("ijk,ik", psiA1[:, GxGp], np.conjugate(psiA1[:, : len(G)]))
                + np.einsum("ijk,ik", psiB1[:, GxGp], np.conjugate(psiB1[:, : len(G)]))
            ) + l2 * (
                np.einsum("ijk,ik", psiA2[:, GxGp], np.conjugate(psiA2[:, : len(G)]))
                + np.einsum("ijk,ik", psiB2[:, GxGp], np.conjugate(psiB2[:, : len(G)]))
            )

        m = np.zeros((px, px, len(energies)))
        # m = np.zeros((px, px))
        # en = energies[0]
        for i, en in enumerate(energies):
            m[:, :, i] = im(en, px, eiGr, rho_G, G)
            # Rotation for comparison with paper [4]
            m[:, :, i] = m[:, :, i].T
            m[:, :, i] = m[::-1, ::-1, i]
        # m = np.zeros((px, px, len(en)))
        # for i, en in enumerate(energies):
        #     m[:, :, i] = im(en, px, eiGr, rho_G, G)
        #     m[:, :, i] = m[:, :, i].T
        #     m[:, :, i] = m[::-1, ::-1, i]

        # m[:, :] = im(en, px, eiGr, rho_G, G)
        # # # Rotation for comparison with paper [4]
        # m[:, :] = m[:, :].T
        # m[:, :] = m[::-1, ::-1]
        #         np.savez(data = data, label=)

        return m

    def solve_LDOSen(
        self,
        nk=16,
        px=65,
        sz=30,
        l1=1,
        l2=1,
        energies=np.round(np.linspace(-0.1, 0.1, 100), 3),
        xi=1,
    ):
        # nk^2 = momentum space grid size
        # px = number of real space pixels
        # sz = size of image in nm
        # l1/l2 = layer weights for outer and inner layers respectively

        # helper function to create 2D image from Fourier components
        def im(energy, px, eiGr, rho_G, G):
            gap = np.ones(len(G))
            amps = np.real(np.sum(rho_G.get(energy, gap) * eiGr.T, 1))
            amps = np.reshape(amps, (px, px))
            return amps

        energies = np.array(energies)
        # Energies from the 2nd Layer
        G = self.G

        # sublattice/layer masks
        t = np.array(
            [
                val
                for pair in zip(self.Q[:, 2], self.Q[:, 2], self.Q[:, 2], self.Q[:, 2])
                for val in pair
            ]
        )

        looo = np.array(self.Nq * [1, 0, 0, 0])
        oloo = np.array(self.Nq * [0, 1, 0, 0])
        oolo = np.array(self.Nq * [0, 0, 1, 0])
        oool = np.array(self.Nq * [0, 0, 0, 1])

        #
        kpts = np.array(
            [
                i * self.b[0] - j * self.b[1]
                for i in np.linspace(0, 1, nk, endpoint=False)
                for j in np.linspace(0, 1, nk, endpoint=False)
            ]
        )
        # l_theta = a/(2*sin(theta/2))
        l_theta = 4 * np.pi / self.k_theta / 3
        xhat = np.array([1, 0])
        yhat = np.array([0, 1])
        # One lattice is dislocated l_theta w.r.t. the other

        rpts = (
            np.array(
                [
                    # offset for better visualization of Moiré Unit cell
                    (-(40 - sz) / 2) * xhat
                    + (-(25 - sz) / 2) * yhat
                    + (-sz / 2 + i * sz / px) * xhat
                    + (-sz / 2 + j * sz / px) * yhat
                    for i in range(px)
                    for j in range(px)
                ]
            )
            - (l_theta / 8) * xhat
        )
        rpts2 = np.array(
            [
                # offset for better visualization of Moiré Unit cell
                (-(40 - sz) / 2) * xhat
                + (-(25 - sz) / 2) * yhat
                + (-sz / 2 + i * sz / px) * xhat
                + (-sz / 2 + j * sz / px) * yhat
                for i in range(px)
                for j in range(px)
            ]
        )
        # x=[40, 36, 26], y=[25, 35, 20], c="b", s=60
        # plt.scatter(rpts[:, 0], rpts[:, 1])
        # plt.scatter(rpts[0:64, 0], rpts[0:64, 1], c='b')
        # plt.scatter(rpts[40*25, 0], rpts[40*25, 1], c='r')
        # plt.scatter(rpts[36*35, 0], rpts[36*35, 1], c='r')
        # plt.scatter(rpts[26*20, 0], rpts[26*20, 1], c='orange')
        # plt.scatter(rpts[26*49, 0], rpts[26*49, 1], c='orange')
        # plt.scatter(rpts[51*34, 0], rpts[51*34, 1], c='b')
        # plt.scatter(rpts[51*62, 0], rpts[51*62, 1], c='b')
        # plt.scatter(0, l_theta, c='pink')
        # plt.show()
        # print(rpts.shape)
        # Phase matrix
        # kj means the transpose of rpts is taken (just to multiply correctly)
        eiGr = np.exp(-1j * np.einsum("ij,kj", G, rpts))

        # Create a matrix mapping indices of g in G and g' in G to the index of g + g' in G
        GxGp = np.zeros((len(G), len(G)), dtype="int16")
        for g in range(len(G)):
            for gp in range(len(G)):
                if list(np.round(G[g] + G[gp], 3)) in np.round(G, 3).tolist():
                    GxGp[g, gp] = (
                        np.round(G, 3).tolist().index(list(np.round(G[g] + G[gp], 3)))
                    )
                else:
                    GxGp[g, gp] = -1

        # Create dictionary to store DOS
        dos = {}

        # Solve the model on a grid of kpts
        for kpt in kpts:
            ham = self.gen_ham(kpt[0], kpt[1], xi)
            values, vectors = eigh(ham)
            for j in range(len(values)):
                val = np.round(values[j], 3)
                vec = np.array(
                    vectors[:, j].flatten().tolist()[0] + [0]
                )  # add a zero on the end to mask -1 in GxGp
                if val in dos:
                    dos[val].append(vec)
                else:
                    dos[val] = [vec]

        # Generate fourier components of the LDOS
        rho_G = {}

        for val in dos:
            # If the eigenvalue is not on the desired energy, don't calculate the LDOS
            if val > np.max(energies) or val < np.min(energies):
                continue

            rho_G[val] = np.zeros(len(G), dtype="complex")

            psiA1 = np.array(dos[val])[:, np.append(oolo * t == 1, True)]
            psiB1 = np.array(dos[val])[:, np.append(oool * t == 1, True)]
            psiA2 = np.array(dos[val])[:, np.append(looo * t == 1, True)]
            psiB2 = np.array(dos[val])[:, np.append(oloo * t == 1, True)]

            rho_G[val] += l1 * (
                np.einsum("ijk,ik", psiA1[:, GxGp], np.conjugate(psiA1[:, : len(G)]))
                + np.einsum("ijk,ik", psiB1[:, GxGp], np.conjugate(psiB1[:, : len(G)]))
            ) + l2 * (
                np.einsum("ijk,ik", psiA2[:, GxGp], np.conjugate(psiA2[:, : len(G)]))
                + np.einsum("ijk,ik", psiB2[:, GxGp], np.conjugate(psiB2[:, : len(G)]))
            )

        m = np.zeros((px, px, len(energies)))
        for i, en in enumerate(energies):
            m[:, :, i] = im(en, px, eiGr, rho_G, G)
            m[:, :, i] = m[:, :, i].T
            m[:, :, i] = m[::-1, ::-1, i]

        def im(energy, px, eiGr, rho_G, G):
            # px = number of real spzace pixels
            # eiGr = fourier Phase
            # rho_G = wavefunction weight
            gap = np.ones(len(G))
            amps = np.real(np.sum(rho_G.get(energy, gap) * eiGr.T, 1))
            # amps = np.reshape(amps, (px, px))
            return amps

        # Add plot option if needed
        # plt.plot(energies[:], m[43, 22, :], c="green")
        # plt.plot(energies[:], m[34, 37, :], c="red")
        # plt.plot(energies[:], m[26, 22, :], color="blue")
        #
        # fig, ax = plt.subplots()
        # a1 = np.max(m[25, 38, :])
        # a2 = np.max(m[36, 35, :])
        # a3 = np.max(m[20, 26, :])
        # ax.tick_params(axis="x", which="minor", bottom=False)
        # ax.tick_params(axis="y", which="minor", bottom=False)
        # plt.plot(energies[:], m[25, 38, :] / a1 + 1, c="green", label="BAAC")
        # plt.plot(energies[:], m[36, 35, :] / a2, c="red", label="ABCA")
        # plt.plot(energies[:], m[20, 26, :] / a3 + 2, color="blue", label="ABAB")

        # Find the points more precisely
        m2 = np.array([m[25, 38, :], m[36, 35, :], m[20, 26, :]])
        m2_sigma = np.array([m[25, 38, :], m[36, 35, :], m[20, 26, :]])
        sca = np.zeros((px, px, 3))  # Scaleogram of 3 stack points
        sca_sigma = np.zeros((px, px, 3))  # Scaleogram of 3 stack points
        m3 = np.zeros((px, px, 4))  # Scaleogram of 3 stack points

        # LDOS(r) for 4 energies
        # (j1,) = np.where(energies == -0.017)
        for i in range(3):
            m2_sigma[i, :] = np.abs(np.random.normal(m2[i, :], 0.31))
            sca[:, :, i] = wavelet_trafo(m2[i, :])
            sca_sigma[:, :, i] = wavelet_trafo(m2_sigma[i, :])
        # plt.plot(energies[:], m2_sigma[0, :] / a1 + 1, c="black", label="BAAC")
        # plt.plot(energies[:], m2_sigma[1, :] / a2, c="black", label="ABCA")
        # plt.plot(energies[:], m2_sigma[2, :] / a3 + 2, color="black", label="ABAB")
        # return list of layer projected DOS and full DOS
        ind_rv1 = ind_en(energies, -35 / 1000)
        ind_vfb = ind_en(energies, -15 / 1000)
        ind_cfb = ind_en(energies, 1 / 1000)
        ind_rc1 = ind_en(energies, 23 / 1000)
        # print(ind_rv1,ind_vfb,ind_cfb,ind_rc1)
        m3 = np.array(
            [m[:, :, ind_rv1], m[:, :, ind_vfb], m[:, :, ind_cfb], m[:, :, ind_rc1]]
        )
        return m2, m2_sigma, sca, sca_sigma, m3


