import numpy as np
import random
import os
from scipy import misc

beta = 0.5
def sigma(x):
	global beta
	return 1./(1+np.exp(-2.*beta*x))
def sigma_(x):
	global beta
	return 2.*beta*np.exp(-2*beta*x)/((1+np.exp(-2.*beta*x))**2)

N_layers = 4
dim=100
N = np.array([dim*dim,16,16,10])
Ideal = np.eye(10)

#Weighs matrix. Mab[i][j] is the weigh of a[j] for b[i]
M01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
M12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
M23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])

#Begins network computing
def iteration(img, Id):
	global N
	global M01
	global M12
	global M23
	eta = 1
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
	
	#Back propagation begins here:
	delta23 = sigma_(H3)*(Id-V3)
	x , y = np.meshgrid(V2,delta23)
	DeltaM23 = eta*x*y
	
	delta12 = sigma_(H2) * np.array([np.dot( M23.transpose()[j] , delta23 ) for j in range(N[2])])
	x , y = np.meshgrid(V1,delta12)
	DeltaM12 = eta*x*y
	
	#Some error pops up in sigma_ function here:
	delta01 = sigma_(H1) * np.array([np.dot( M12.transpose()[j] , delta12 ) for j in range(N[2])])
	x , y = np.meshgrid(V0,delta01)
	DeltaM01 = eta*x*y
	
	M01 += DeltaM01
	M12 += DeltaM12
	M23 += DeltaM23

img = np.flip( misc.imread('prueba.bmp',flatten=1) , 0 )
result = iteration(img, Ideal[1])
img = np.flip( misc.imread('prueba.bmp',flatten=1) , 0 )
