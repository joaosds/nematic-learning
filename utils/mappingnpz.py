import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmath

path = '/scratch/c7051184/'
data_dict = {}

for j in range(1, 15):
    fileold = 'station{0}/temptest{0}.npz'.format(j)
    filenew = 'station{0}/temptestmap{0}.npz'.format(j)
    data = np.load(path + fileold)
    print(data)
    dataset = pd.DataFrame(data["DataY"]) 
    func = []
    func2 = []
    print(dataset.info)

    for i in range(len(data["DataY"])):
        phimn = data["DataY"][i][0]
        phign = data["DataY"][i][1]
        eps = data["DataY"][i][2]
        varphi_eps = data["DataY"][i][3]
        #print(data["DataY"][i])
        #print(phimn, phign, eps, varphi_eps)
        temp = eps*cmath.exp(1j*varphi_eps)
        re_eps = temp.real
        img_eps = temp.imag
        #print(temp, re_eps, img_eps)
        alpha = [phimn, phign, re_eps, img_eps]
        alpha2 = [eps, varphi_eps]
        func.append(alpha)
        func2.append(alpha2)

    vectorized_images = data["DataX"]
    sca = data["DataZ"]
    scanoise = data["DataW"]

    np.savez(path + filenew, DataX=vectorized_images, DataY=func, DataZ=sca, DataW=scanoise, DataP=func2)




# Organize code for option to just read, and to change angle

#data = np.load("/scratch/c7051184/ml/Trainingangle.npz")
#data = np.load("/scratch/c7051184/station1/temptestmap1.npz")
#npz = data
#a = np.random.default_rng().uniform(low=0.01, high=0.1, size=1)[0]
#b = np.random.default_rng().uniform(low=0.01, high=0.1, size=1)
#print(data["DataX"].shape)
#print(data["DataY"].shape)
#print(data["DataW"].shape)
#print(data["DataZ"].shape)
##vectorized_images = []
#dataset = pd.DataFrame(data["DataP"]) 
###thetas = []
##func = []
#print(dataset.info)
#####print(dataset[2])
##### Add column with cos and sin to dataset
#
##for i in range(len(data["DataY"])):
##    theta = data["DataY"][i][3]
##    theta2 = data["DataY"][i][0]
##    phim = data["DataY"][i][0]
##    phiiagn = data["DataY"][i][1]
##    epsilon = data["DataY"][i][2]
##    print(theta, phim, phiiagn, epsilon)
##    # cos = data["DataY"][i][1]
##    sin = np.sin(theta)
##    cos = np.cos(theta)
##    # print(theta, theta2, theta3)
##    alpha = [phim, phiiagn, epsilon,  cos, sin]
##    thetas.append(theta)
##    print(data["DataY"][i])
##    # vectorized_images.append(data["DataX"][i])
##    func.append(alpha)
##    #thetas.append(theta)
## # # #
##vectorized_images = data["DataX"]
##sca = data["DataZ"]
##scanoise = data["DataW"]
#### # # # #
#### # # # #
##np.savez("Teststrainanglesepsn.npz", DataX=vectorized_images, DataY=func, DataZ=sca, DataW=scanoise, DataA=thetas)
### # print(len(vectorized_images))
### # print(len(data["DataY"]))
### print(len(thetas))
### dataset['sin'] = np.sin(dataset[0])
### dataset['cos'] = np.cos(dataset[0])
##
##
### dataset.drop_duplicates()
### df = pd.DataFrame(data, columns=['matrices', 'labels'])
### df.info()
