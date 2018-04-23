import numpy as np
import random
import os
from scipy import misc
import h5py as h5

beta = 0.5
def sigma(x):
	global beta
	return 1./(1+np.exp(-2.*beta*x))

fin = h5.File('Network1.h5','r')
M01 = np.array([[0 for j in range(len(fin['M01'][0]))] for i in range(len(fin['M01']))])
M12 = np.array([[0 for j in range(len(fin['M12'][0]))] for i in range(len(fin['M12']))])
M23 = np.array([[0 for j in range(len(fin['M23'][0]))] for i in range(len(fin['M23']))])
M01[:] = fin['M01'][:]
M12[:] = fin['M12'][:]
M23[:] = fin['M23'][:]
fin.close()

print M01

dim = 100
N = np.array([dim*dim,16,16,10])

img = (255. - np.flip( misc.imread('data/prueba0.bmp',flatten=1) , 0 )) / 255.

V0 = img.reshape(-1)
H1 = np.array([ np.dot(M01[i][:] , V0[:]) for i in range(N[1])])
V1 = sigma( H1 )
H2 = np.array([ np.dot(M12[i][:] , V1[:]) for i in range(N[2])])
V2 = sigma( H2 )
H3 = np.array([ np.dot(M23[i][:] , V2[:]) for i in range(N[3])])
V3 = sigma( H3 )
max_val = V3.tolist().index(np.amax(V3))
print V3
print "Maximun value found found in ", max_val,"\n"

img = (255. - np.flip( misc.imread('data/prueba1.bmp',flatten=1) , 0 )) / 255.

V0 = img.reshape(-1)
H1 = np.array([ np.dot(M01[i][:] , V0[:]) for i in range(N[1])])
V1 = sigma( H1 )
H2 = np.array([ np.dot(M12[i][:] , V1[:]) for i in range(N[2])])
V2 = sigma( H2 )
H3 = np.array([ np.dot(M23[i][:] , V2[:]) for i in range(N[3])])
V3 = sigma( H3 )
max_val = V3.tolist().index(np.amax(V3))
print V3
print "Maximun value found found in ", max_val,"\n"

img = (255. - np.flip( misc.imread('data/prueba2.bmp',flatten=1) , 0 )) / 255.

V0 = img.reshape(-1)
H1 = np.array([ np.dot(M01[i][:] , V0[:]) for i in range(N[1])])
V1 = sigma( H1 )
H2 = np.array([ np.dot(M12[i][:] , V1[:]) for i in range(N[2])])
V2 = sigma( H2 )
H3 = np.array([ np.dot(M23[i][:] , V2[:]) for i in range(N[3])])
V3 = sigma( H3 )
max_val = V3.tolist().index(np.amax(V3))
print V3
print "Maximun value found found in ", max_val,"\n"
