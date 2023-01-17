#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import TDBG
import os
import re

# get the name of the folder
cwd2 = os.path.split(os.getcwd())[1]
print(str(re.findall("\d+", cwd2)))
cwd2int = int("".join(filter(str.isdigit, cwd2)))
npzfile = f"temptestnorm{cwd2int}.npz"
print(npzfile)
# f ----------------------------ML file generation---------------------------#
vectorized_images = []
LDOS = []
LDOS_sigma = []
sca = []
sca_sigma = []
alphas = []
params = []
theta = 2 * np.pi / 3
en = np.round(np.linspace(-0.085, 0.077, 65), 3)
N = 1000

for i in range(N):
    PhiMNt = np.random.default_rng().uniform(low=0.001, high=0.1, size=1)[0]
    PhiIAGNt = np.random.default_rng().uniform(low=0.001, high=0.1, size=1)[0]
    phit = np.random.default_rng().uniform(low=0, high=np.pi, size=1)[0]
    alpha = [np.cos(2.0 * phit), np.sin(2.0 * phit)]
    param = [phit, PhiMNt, PhiIAGNt]
    model = TDBG.Model(
        1.05,
        0,
        0,
        0,
        PhiMN=PhiMNt,
        PhiIAGN=PhiIAGNt,
        varphiMN=phit,
        varphiIAGN=phit,
        alphal=0,
        vf=1.3,
        cut=3,
    )
    m1, m2, m3, m4, m5 = model.solve_LDOSen(nk=35, l2=0, energies=en)
    print(alpha)
    for j in range(4):
        # LDOS(r) for 4 energies: -0.017, -0.016, 0.001, 0.003
        vectorized_images.append(m5[j, :, :])
    for j in range(3):
        # LDOS(w) for BAAC, ABCA, ABAB
        sca.append(m3[:, :, j])
        sca_sigma.append(m4[:, :, j])
    alphas.append(alpha)
    params.append(param)

# save final npzfile for ml
np.savez(
    "expdata.npz",
    DataX=vectorized_images,
    DataY=alphas,
    DataZ=sca,
    DataW=sca_sigma,
    dataP=params,
)
