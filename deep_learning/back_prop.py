import numpy as np
import random
import os
from scipy import misc
import h5py as h5
import time

BETA = 0.5	# Sigmoid parameter
ETA = 1.4e-4 	# Gradient descent parameter
GAMMA = 0	# Parametro para darle menos valor a los valores bajos
ALPHA = 0	# Momentum parameter
A = 0  		# Parameters for adaptative ETA 
B = 0

def sigma(x):
	global BETA
	return 1./(1+np.exp(-2.*BETA*x))

def sigma_(x):
	global BETA
	return 2.*BETA*np.exp(-2*BETA*x)/((1+np.exp(-2.*BETA*x))**2)

def iteration(img, Id, N, w01, w12, w23, a01, a12, a23, th1, th2, th3, verbose=False):
	global BETA,ETA,GAMMA,ALPHA,A,B
	V0 = img.reshape(-1)
	H1 = np.array([ np.dot((a01*w01)[i][:] , V0[:]) for i in range(N[1])]) - th1
	V1 = sigma( H1 )
	H2 = np.array([ np.dot((a12*w12)[i][:] , V1[:]) for i in range(N[2])]) - th2
	V2 = sigma( H2 )
	H3 = np.array([ np.dot((a23*w23)[i][:] , V2[:]) for i in range(N[3])]) - th3
	V3 = sigma( H3 )
	max_val = V3.tolist().index(np.amax(V3))
	if verbose:
		print Id.tolist().index(1)
		print V3
		print "Maximum value found found in ", max_val,"\n"

	#Back propagation begins here:
	delta23 = sigma_(H3)*(Id-V3)
	x , y = np.meshgrid(V2,delta23)
	Deltaw23 = ETA*x*y - ETA*GAMMA * w23/((1+w23**2)**2)
	Deltath3 = -1.*ETA*delta23 - ETA*GAMMA * th3/((1+th3**2)**2)

	delta12 = sigma_(H2) * np.array([np.dot( (a23*w23).transpose()[j] , delta23 ) for j in range(N[2])])
	x , y = np.meshgrid(V1,delta12)
	Deltaw12 = ETA*x*y - ETA*GAMMA * w12/((1+w12**2)**2)
	Deltath2 = -1.*ETA*delta12 - ETA*GAMMA * th2/((1+th2**2)**2)

	delta01 = sigma_(H1) * np.array([np.dot( (a12*w12).transpose()[j] , delta12 ) for j in range(N[1])])
	x , y = np.meshgrid(V0,delta01)
	Deltaw01 = ETA*x*y - ETA*GAMMA * w01/((1+w01**2)**2)
	Deltath1 = -1.*ETA*delta01 - ETA*GAMMA * th1/((1+th1**2)**2)

	return Deltaw01 , Deltaw12 , Deltaw23, Deltath1, Deltath2, Deltath3

def back_prop(W,N_fonts, title='Network.h5', verbose=False):
	global BETA,ETA,GAMMA,ALPHA,A,B
	ETA = 1
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
	fonts = np.random.choice(range(1000),N_fonts,replace=False)
	iterations=10
	error = []
	for n in range(iterations):
		'''
		if n>0:
			Deltaw01_,Deltaw12_,Deltaw23_ = Deltaw01[:],Deltaw12[:],Deltaw23[:]
			Deltath1_,Deltath2_,Deltath3_ = Deltath1[:],Deltath2[:],Deltath3[:]
		else:
			Deltaw01_,Deltaw12_,Deltaw23_ = 0*w01,0*w12,0*w23
			Deltath1_,Deltath2_,Deltath3_ = 0*th1,0*th2,0*th3
		Deltaw01,Deltaw12,Deltaw23 = 0*w01,0*w12,0*w23
		Deltath1,Deltath2,Deltath3 = 0*th1,0*th2,0*th3
		'''
		for j in fonts:
			for i in range(10):
				
				if n>0:
					Dw01_,Dw12_,Dw23_,Dth1_,Dth2_,Dth3_ = Dw01[:],Dw12[:],Dw23[:],Dth1[:],Dth2[:],Dth3[:]
				else:
					Dw01_,Dw12_,Dw23_,Dth1_,Dth2_,Dth3_ = 0*w01,0*w12,0*w23,0*th1,0*th2,0*th3
				
				if verbose:
					print str(n+1)+'/'+str(iterations)+': data/img'+str(i)+'{0:03}'.format(j)+'.bmp'
				img = (255. - np.flip( misc.imread('data/img'+str(i)+'{0:03}'.format(j)+'.bmp',flatten=1) , 0 )) / 255.
				Dw01,Dw12,Dw23,Dth1,Dth2,Dth3 = iteration(img, Ideal[i], N, w01, w12, w23, a01, a12, a23, th1, th2, th3, verbose)
				
				w01 += Dw01 + ALPHA * Dw01_
				w12 += Dw12 + ALPHA * Dw12_
				w23 += Dw23 + ALPHA * Dw23_
				th1 += Dth1 + ALPHA * Dth1_
				th2 += Dth2 + ALPHA * Dth2_
				th3 += Dth3 + ALPHA * Dth3_
				'''
				Deltaw01 += Dw01
				Deltaw12 += Dw12
				Deltaw23 += Dw23
				Deltath1 += Dth1
				Deltath2 += Dth2
				Deltath3 += Dth3
				'''
		'''
		w01 += Deltaw01 + ALPHA * Deltaw01_
		w12 += Deltaw12 + ALPHA * Deltaw12_
		w23 += Deltaw23 + ALPHA * Deltaw23_
		th1 += Deltath1 + ALPHA * Deltath1_
		th2 += Deltath2 + ALPHA * Deltath2_
		th3 += Deltath3 + ALPHA * Deltath3_
		
		print Deltaw01, Deltaw12, Deltaw23, Deltath1, Deltath2, Deltath3
		
		fout = h5.File('temp.h5','w')
		fout.attrs['dim'] = dim
		fout.attrs['N_layers'] = N_layers
		fout.attrs['N'] = N
		fout.attrs['fonts'] = [k for k in range(1000) if not k in fonts]
		fout['w01'] = w01
		fout['w12'] = w12
		fout['w23'] = w23
		fout['a01'] = a01
		fout['a12'] = a12
		fout['a23'] = a23
		fout['th1'] = th1
		fout['th2'] = th2
		fout['th3'] = th3
		fout.close()
		error += [evaluate('temp.h5', printerror=False)]
		if n > 0:
			if error[n] < error[n-1]:
				ETA += A
			else:
				ETA += -1.*B*ETA
		'''
	
	os.system('rm temp.h5')
	fout = h5.File(title,'w')
	fout.attrs['dim'] = dim
	fout.attrs['N_layers'] = N_layers
	fout.attrs['N'] = N
	fout.attrs['fonts'] = fonts
	fout['w01'] = w01
	fout['w12'] = w12
	fout['w23'] = w23
	fout['a01'] = a01
	fout['a12'] = a12
	fout['a23'] = a23
	fout['th1'] = th1
	fout['th2'] = th2
	fout['th3'] = th3
	fout.close()
	
	return error

def evaluate(title='Network.h5', printerror=True, verbose=False):
	global BETA,ETA,GAMMA,ALPHA,A,B
	fin = h5.File(title,'r')
	dim = fin.attrs['dim']
	N = fin.attrs['N']
	fonts = fin.attrs['fonts']
	w01 = fin['w01'][:]
	w12 = fin['w12'][:]
	w23 = fin['w23'][:]
	a01 = fin['a01'][:]
	a12 = fin['a12'][:]
	a23 = fin['a23'][:]
	th1 = fin['th1'][:]
	th2 = fin['th2'][:]
	th3 = fin['th3'][:]
	fin.close()
	
	Ideal = np.eye(10)
	
	error = 0.
	aciertos = 0
	fallos = 0
	for j in [k for k in range(1000) if not k in fonts]:
		for i in range(10):
			img = (255. - np.flip( misc.imread('data/img'+str(i)+'{0:03}'.format(j)+'.bmp',flatten=1) , 0 )) / 255.
	
			V0 = img.reshape(-1)
			H1 = np.array([ np.dot((a01*w01)[k][:] , V0[:]) for k in range(N[1])]) - th1
			V1 = sigma( H1 )
			H2 = np.array([ np.dot((a12*w12)[k][:] , V1[:]) for k in range(N[2])]) - th2
			V2 = sigma( H2 )
			H3 = np.array([ np.dot((a23*w23)[k][:] , V2[:]) for k in range(N[3])]) - th3
			V3 = sigma( H3 )
			error += sum((V3-Ideal[i])**2)
			max_val = V3.tolist().index(np.amax(V3))
			if max_val == i:
				aciertos += 1
			else: 
				fallos +=1
			if verbose:
				print i
				print V3
				print "Maximum value found found in ", max_val,"\n"
	
	error = 1.*error/(10*300)
	if printerror:
		print 'Total:',aciertos+fallos
		print 'Aciertos:',aciertos
		print 'Fallos:',fallos
		print 'Error cuadratico:',error
	
	return error
