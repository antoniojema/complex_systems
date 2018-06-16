import numpy as np
import random
import h5py as h5
import back_prop as bp

bp.ITERATIONS = 10
n_fonts = 500

def set_rand_omega():
	dim = 28
	N = np.array([dim*dim,49,16,10])
	#N = np.array([dim*dim,16,16,10])
	#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
	w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
	w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
	w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
	#Alive matrix. If TRUE it means weigh exists
	a01 = np.array([[True for j in range(N[0])] for i in range(N[1])])
	a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
	a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
	#Thresholds
	th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
	th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
	th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])
	return N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]]
N, W = set_rand_omega()
error_min = bp.evaluate(N, W, np.random.choice(range(bp.N_FONTS),300,replace=False))[0]

while(True):
	print '\n\nNueva semilla'
	N, W = set_rand_omega()
	fonts = np.random.choice(range(bp.N_FONTS),700,replace=False)
	W = bp.back_prop(N,W,fonts)[0]
	error = bp.evaluate(N, W, [k for k in range(bp.N_FONTS) if not k in fonts])[0]
	print 'Error minimo:',error_min
	if(error < error_min):
		error_min = error
		fout = h5.File('Best.h5','w')
		fout.attrs['fonts'] = fonts
		fout.attrs['N'] = N
		fout['w01'] = W[0][0]
		fout['w12'] = W[0][1]
		fout['w23'] = W[0][2]
		fout['a01'] = W[1][0]
		fout['a12'] = W[1][1]
		fout['a23'] = W[1][2]
		fout['th1'] = W[2][0]
		fout['th2'] = W[2][1]
		fout['th3'] = W[2][2]
		fout.close()

