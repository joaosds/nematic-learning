#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: joaosds
"""
import numpy as np

path = '/scratch/c7051184/'
data_dict = {}

n2 = 12 + 1 # number of files + 1
n1 = 11
data_dict = {'data_'+str(i) : np.load(path + 'station{0}/temptestmap{0}.npz'.format(i)) for i in range(n1,n2)}

## Merge arrays
# DataX -> [size(0:3), 65, 65] LDOS for energies [-15, -10, -5, 1] for the alpha = [] parameters
# DataY -> [] alpha = [PhiMN, PhiIAGN, alphal]
# DataZ -> LDOS as a function of energies, from [-100, 100, 65] me, [L1, L2, L3, L4, (L1+L2+L3+L4)/4]
# DataW -> Same as DataZ but with random gaussian noise added in layers. 
arr_0 = np.concatenate([data_dict['data_{0}'.format(i)]['DataX'] for i in range(n1,n2)])
arr_1 = np.concatenate([data_dict['data_{0}'.format(i)]['DataY'] for i in range(n1,n2)])
arr_2 = np.concatenate([data_dict['data_{0}'.format(i)]['DataZ'] for i in range(n1,n2)])
arr_3 = np.concatenate([data_dict['data_{0}'.format(i)]['DataW'] for i in range(n1,n2)])
arr_4 = np.concatenate([data_dict['data_{0}'.format(i)]['DataP'] for i in range(n1,n2)])
#
#print(arr_0.shape, arr_1.shape, arr_2.shape, arr_3.shape, arr_4.shape)
np.savez('Teststrainnewmap.npz', DataX=arr_0, DataY=arr_1, DataZ=arr_2, DataW=arr_3, DataP=arr_4)
#np.savez('Teststrainnewmap.npz', DataX=arr_0, DataY=arr_1, DataZ=arr_2, DataW=arr_3)
