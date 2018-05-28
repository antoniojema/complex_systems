import numpy as np
import random
import os
from scipy import misc
import h5py as h5
import time

N_FUENTES = 1000
BETA = 0.5		# Sigmoid parameter
ETA = 1.4e-4 	# Gradient descent parameter
GAMMA = 0		# Parametro para darle menos valor a los valores bajos
ALPHA = 0		# Momentum parameter
A = 0  			# Parameters for adaptative ETA 
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

def back_prop(N, W, fonts, f_extra=None, title='Network.h5', verbose=False, save=False, calculate_error=False):
	global BETA,ETA,GAMMA,ALPHA,A,B
	ETA = 1
	
	Ideal = np.eye(10)
	
	#Reads network parameters
	w01 = W[0][0]; w12 = W[0][1]; w23 = W[0][2];
	a01 = W[1][0]; a12 = W[1][1]; a23 = W[1][2];
	th1 = W[2][0]; th2 = W[2][1]; th3 = W[2][2];

	#Begins network computing
	iterations=20
	error = []	
	hits = []
	hits_percent = []
	error_prediction = []
	for n in range(iterations):
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
		
		if f_extra!=None:
			error_prediction += [evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[f_extra], printerror=False)[0]]
		
		if calculate_error:		#calculate_error==True -> Cada vez que ha pasado por todas las imagenes, calcula el error con el resto
			#Calcula el error, aciertos y aciertos(%) y lo anade a sus correspondientes vectores 
			err,hit,tot = evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[k for k in range(N_FUENTES) if not k in fonts], printerror=False)
			error		 += [err]
			hits		 += [hit]
			hits_percent += [100.*hit/tot]
			if n > 0:
				if error[n] < error[n-1]:
					ETA += A
				else:
					ETA += -1.*B*ETA
	
	if save:	#save==True -> Guarda en H5 la red y las fuentes con las que ha sido entrenada, devuelve el error
		fout = h5.File(title,'w')
		fout.attrs['fonts'] = fonts
		fout.attrs['N'] = N
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
		return error, hits, hits_percent
	else:		#save==False -> Devuelve como parametros la red y el error
		return [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], [error, hits, hits_percent, error_prediction]

def evaluate(N, W, fonts, printerror=True, verbose=False):
	global BETA,ETA,GAMMA,ALPHA,A,B
	
	Ideal = np.eye(10)
	
	#Reads network parameters
	w01 = W[0][0]; w12 = W[0][1]; w23 = W[0][2];
	a01 = W[1][0]; a12 = W[1][1]; a23 = W[1][2];
	th1 = W[2][0]; th2 = W[2][1]; th3 = W[2][2];
	
	#Evaluates network
	error = 0.
	aciertos = 0
	fallos = 0
	for j in fonts:
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
	
	error = 1.*error/(10*len(fonts))
	if printerror:
		print 'Total:',aciertos+fallos
		print 'Aciertos:',aciertos
		print 'Fallos:',fallos
		print 'Error cuadratico:',error
	
	return error, aciertos, aciertos+fallos

