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
npzfile = f"temptest{cwd2int}.npz"
print(npzfile)

# ----------------------------ML file generation---------------------------#
vectorized_images = []
LDOS = []
LDOS_sigma = []
sca = []
sca_sigma = []
alphas = []
theta = np.pi / 3

# en = np.round(np.linspace(-0.085, 0.077, 65), 3)
en = np.round(np.linspace(-0.07, 0.06, 65), 3)
print(en)
N = 1000
for i in range(N):
    PhiMNt = np.random.default_rng().uniform(low=0.001, high=0.1, size=1)[0]
    PhiIAGNt = np.random.default_rng().uniform(low=0.001, high=0.1, size=1)[0]
    epsilont = np.random.default_rng().uniform(low=0.000, high=0.008, size=1)[0]
    phit = np.random.default_rng().uniform(low=0, high=np.pi / 3, size=1)[0]
    alpha = [PhiMNt, PhiIAGNt, epsilont, phit]
    model = TDBG.Model(
        1.05,
        phit,
        epsilont,
        0,
        PhiMN=PhiMNt,
        PhiIAGN=PhiIAGNt,
        varphiMN=theta,
        varphiIAGN=theta,
        alphal=0,
        vf=1.3,
        cut=3,
    )
    m1, m2, m3, m4, m5 = model.solve_LDOSen(nk=35, l2=0, energies=en)
    print(alpha)
    for j in range(4):
        # LDOS(r) for 4 energies: -0.035, -0.015, 0.001, 0.023 (RC1, VFB,CFB, RC1)
        vectorized_images.append(m5[j, :, :])
    for j in range(3):
        # LDOS(w) for BAAC, ABCA, ABAB
        # LDOS.append(m1[j,:])
        # LDOS_sigma.append(m2[j,:])
        # LDOS(w) in scaleogram for BAAC, ABCA, ABAB
        sca.append(m3[:, :, j])
        sca_sigma.append(m4[:, :, j])
    alphas.append(alpha)

# save final npzfile for ml
np.savez(npzfile, DataX=vectorized_images, DataY=alphas, DataZ=sca, DataW=sca_sigma)
