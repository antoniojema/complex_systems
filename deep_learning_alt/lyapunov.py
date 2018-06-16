import numpy as np
import random
from scipy import misc

BETA = 0.5 #Sigmoid parameter
ETA = 1 #Gradient descent parameter
GAMMA = 0.1 #Parametro para darle menos valor a los valores bajos
ALPHA = 0.1 #Momentum parameter

def sigma(x):
	global BETA
	return 1./(1+np.exp(-2.*BETA*x))

def sigma_(x):
	global BETA
	return 2.*BETA*np.exp(-2*BETA*x)/((1+np.exp(-2.*BETA*x))**2)

def iteration(img, Id, N, w01, w12, w23, a01, a12, a23, th1, th2, th3):
	global ETA
	
	V0 = img.reshape(-1)
	H1 = np.array([ np.dot((a01*w01)[i][:] , V0[:]) for i in range(N[1])]) - th1
	V1 = sigma( H1 )
	H2 = np.array([ np.dot((a12*w12)[i][:] , V1[:]) for i in range(N[2])]) - th2
	V2 = sigma( H2 )
	H3 = np.array([ np.dot((a23*w23)[i][:] , V2[:]) for i in range(N[3])]) - th3
	V3 = sigma( H3 )
	max_val = V3.tolist().index(np.amax(V3))
	#print V3
	#print "Maximum value found found in ", max_val,"\n"

	#Back propagation begins here:
	delta23 = sigma_(H3)*(Id-V3)
	x , y = np.meshgrid(V2,delta23)
	Deltaw23 = ETA*x*y + ETA*GAMMA * w23/((1+w23**2)**2)
	Deltath3 = -1.*ETA*delta23 + ETA*GAMMA * th3/((1+th3**2)**2)

	delta12 = sigma_(H2) * np.array([np.dot( (a23*w23).transpose()[j] , delta23 ) for j in range(N[2])])
	x , y = np.meshgrid(V1,delta12)
	Deltaw12 = ETA*x*y + ETA*GAMMA * w12/((1+w12**2)**2)
	Deltath2 = -1.*ETA*delta12 + ETA*GAMMA * th2/((1+th2**2)**2)

	delta01 = sigma_(H1) * np.array([np.dot( (a12*w12).transpose()[j] , delta12 ) for j in range(N[1])])
	x , y = np.meshgrid(V0,delta01)
	Deltaw01 = ETA*x*y + ETA*GAMMA * w01/((1+w01**2)**2)
	Deltath1 = -1.*ETA*delta01 + ETA*GAMMA * th1/((1+th1**2)**2)

	return Deltaw01 , Deltaw12 , Deltaw23, Deltath1, Deltath2, Deltath3

def dist(a1,a2,a3,b1,b2,b3):
	return np.sqrt( (a1**2).sum() + (a2**2).sum() + (a3**2).sum() + (b1**2).sum() + (b2**2).sum() + (b3**2).sum() )

N_layers = 4
dim = 28
#N = np.array([dim*dim,16,16,10])
N = np.array([dim*dim,49,16,10])
Ideal = np.eye(10)

#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
#Alive matrix. If TRUE it means weigh exists
a01 = np.array([[False for j in range(N[0])] for i in range(N[1])])
a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
#Thresholds
th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])

#Modified weighs and thresholds
d01 = np.array([[1e-15*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
d12 = np.array([[1e-15*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
d23 = np.array([[1e-15*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
d1 = np.array([1e-15*(random.random()-0.5) for i in range(N[1])])
d2 = np.array([1e-15*(random.random()-0.5) for i in range(N[2])])
d3 = np.array([1e-15*(random.random()-0.5) for i in range(N[3])])
w01_ = w01 + d01
w12_ = w12 + d12
w23_ = w23 + d23
th1_ = th1 + d1
th2_ = th2 + d2
th3_ = th3 + d3

'''
for n in range(4):
	for m in range(4):
		for i in range(7):
			for j in range(7):
				a01[4*n+m][28*(7*n+i)+(7*m+i)] = True
'''

for n in range(7):
	for m in range(7):
		for i in range(4):
			for j in range(4):
				a01[7*n+m][28*(4*n+i)+(4*m+i)] = True


#Begins network computing
N_fonts = 700
fonts = np.random.choice(range(1000),N_fonts,replace=False)
iterations=20
Dw01,Dw12,Dw23,Dth1,Dth2,Dth3 = 0,0,0,0,0,0
for n in range(iterations):
	#Deltaw01 = 0
	#Deltaw12 = 0
	#Deltaw23 = 0
	for j in fonts:
		for i in range(10):
			Dw01_,Dw12_,Dw23_,Dth1_,Dth2_,Dth3_ = Dw01,Dw12,Dw23,Dth1,Dth2,Dth3
			img = (255. - np.flip( misc.imread('data/img'+str(i)+'{0:03}'.format(j)+'.bmp',flatten=1) , 0 )) / 255.
			Dw01,Dw12,Dw23,Dth1,Dth2,Dth3 = iteration(img, Ideal[i], N, w01, w12, w23, a01, a12, a23, th1, th2, th3)
			w01 += Dw01 + ALPHA * Dw01_
			w12 += Dw12 + ALPHA * Dw12_
			w23 += Dw23 + ALPHA * Dw23_
			th1 += Dth1 + ALPHA * Dth1_
			th2 += Dth2 + ALPHA * Dth2_
			th3 += Dth3 + ALPHA * Dth3_
	#w01 += Deltaw01
	#w12 += Deltaw12
	#w23 += Deltaw23

print 'First weighs computed'

#Compute modified weighs
for n in range(iterations):
	for j in fonts:
		for i in range(10):
			Dw01_,Dw12_,Dw23_,Dth1_,Dth2_,Dth3_ = Dw01,Dw12,Dw23,Dth1,Dth2,Dth3
			img = (255. - np.flip( misc.imread('data/img'+str(i)+'{0:03}'.format(j)+'.bmp',flatten=1) , 0 )) / 255.
			Dw01,Dw12,Dw23,Dth1,Dth2,Dth3 = iteration(img, Ideal[i], N, w01_, w12_, w23_, a01, a12, a23, th1_, th2_, th3_)
			w01_ += Dw01 + ALPHA * Dw01_
			w12_ += Dw12 + ALPHA * Dw12_
			w23_ += Dw23 + ALPHA * Dw23_
			th1_ += Dth1 + ALPHA * Dth1_
			th2_ += Dth2 + ALPHA * Dth2_
			th3_ += Dth3 + ALPHA * Dth3_

print 'Second weighs computed'

lyapunov = 1./iterations * np.log( dist(w01-w01_,w12-w12_,w23-w23_,th1-th1_,th2-th2_,th3-th3_)/dist(d01,d12,d23,d1,d2,d3) )
print 'Lyapunov exponent:',lyapunov

