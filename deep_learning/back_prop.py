import numpy as np
import random
import os
from scipy import misc
import h5py as h5
import time

ITERATIONS = 20
ITERATIONS_ALL = 100
N_FONTS = 1000
BETA = 0.5		# Sigmoid parameter
ETA = 1 		# Gradient descent parameter
ETA_ALL =0.1	# Gradient descent all parameter
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
	global ITERATIONS,ITERATIONS_ALL,N_FONTS,BETA,ETA,ETA_ALL,GAMMA,ALPHA,A,B
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
	Deltaw23 = ETA*x*y
	Deltath3 = -1.*ETA*delta23

	delta12 = sigma_(H2) * np.array([np.dot( (a23*w23).transpose()[j] , delta23 ) for j in range(N[2])])
	x , y = np.meshgrid(V1,delta12)
	Deltaw12 = ETA*x*y
	Deltath2 = -1.*ETA*delta12

	delta01 = sigma_(H1) * np.array([np.dot( (a12*w12).transpose()[j] , delta12 ) for j in range(N[1])])
	x , y = np.meshgrid(V0,delta01)
	Deltaw01 = ETA*x*y
	Deltath1 = -1.*ETA*delta01

	return Deltaw01 , Deltaw12 , Deltaw23, Deltath1, Deltath2, Deltath3

def back_prop(N, W, fonts, f_extra=None, title='Network.h5', verbose=False, save=False, calculate_error=False, calculate_train_error=False,converge_criteria=False,max_iterations=10000,n_fonts_error=100,error_criteria=0.1):
	global ITERATIONS,ITERATIONS_ALL,N_FONTS,BETA,ETA,ETA_ALL,GAMMA,ALPHA,A,B
	
	Ideal = np.eye(10)
	
	#Reads network parameters
	w01 = W[0][0]; w12 = W[0][1]; w23 = W[0][2];
	a01 = W[1][0]; a12 = W[1][1]; a23 = W[1][2];
	th1 = W[2][0]; th2 = W[2][1]; th3 = W[2][2];

	#Begins network computing
	training_error = []
	error = []
	hits = []
	hits_percent = []
	error_prediction = []
	not_converged=True
	n_iterations=0
	#for n in range(ITERATIONS):
	
	while True:
		for j in fonts:
			for i in range(10):
				n_iterations+=1
				
				if n_iterations>1:
					Dw01_,Dw12_,Dw23_,Dth1_,Dth2_,Dth3_ = Dw01[:],Dw12[:],Dw23[:],Dth1[:],Dth2[:],Dth3[:]
				else:
					Dw01_,Dw12_,Dw23_,Dth1_,Dth2_,Dth3_ = 0*w01,0*w12,0*w23,0*th1,0*th2,0*th3
				
				if verbose:
					print str(n_iterations)+': data/img'+str(i)+'{0:03}'.format(j)+'.bmp'
				
				img = (255. - np.flip( misc.imread('data/img'+str(i)+'{0:03}'.format(j)+'.bmp',flatten=1) , 0 )) / 255.
				Dw01,Dw12,Dw23,Dth1,Dth2,Dth3 = iteration(img, Ideal[i], N, w01, w12, w23, a01, a12, a23, th1, th2, th3, verbose)
				
				w01 += Dw01 + ALPHA * Dw01_ - ETA*GAMMA * w01/((1+w01**2)**2)
				w12 += Dw12 + ALPHA * Dw12_ - ETA*GAMMA * w12/((1+w12**2)**2)
				w23 += Dw23 + ALPHA * Dw23_ - ETA*GAMMA * w23/((1+w23**2)**2)
				th1 += Dth1 + ALPHA * Dth1_ - ETA*GAMMA * th1/((1+th1**2)**2)
				th2 += Dth2 + ALPHA * Dth2_ - ETA*GAMMA * th2/((1+th2**2)**2)
				th3 += Dth3 + ALPHA * Dth3_ - ETA*GAMMA * th3/((1+th3**2)**2)
			
			if n_iterations%(10*n_fonts_error)==0:
				if f_extra!=None:
					error_prediction += [evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[f_extra], printerror=False)[0]]
				
				if calculate_error:		#calculate_error==True -> Cada vez que ha pasado por todas las imagenes, calcula el error con el resto
					#Calcula el error, aciertos y aciertos(%) y lo anade a sus correspondientes vectores 
					err,hit,tot = evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[k for k in range(N_FONTS) if not k in fonts], printerror=False)
					error		 += [err]
					hits		 += [hit]
					hits_percent += [100.*hit/tot]
					if n_iterations > 1:
						if error[n_iterations-1] < error[n_iterations-2]:
							ETA += A
						elif error[n_iterations-1] > error[n_iterations-2]:
							ETA += -1.*B*ETA
				
				if calculate_train_error:
					training_error += [evaluate(N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], fonts, printerror=False)[0]]
		
			#Criterio de convergencia
			if converge_criteria:
				if evaluate(N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], fonts, printerror=False)[0] < error_criteria:
					not_converged = False
				elif n_iterations >= max_iterations:
					not_converged = False
					n_iterations += 1
			
			elif n_iterations >= max_iterations:
				not_converged = False
			
			if not not_converged:
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
					return training_error, error, hits, hits_percent, error_prediction, n_iterations
				else:		#save==False -> Devuelve como parametros la red y el error
					return [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], [training_error, error, hits, hits_percent, error_prediction, n_iterations]
		
		if len(fonts) == 0:
			n_iterations+=1
			if n_iterations%(10*n_fonts_error)==0:
				if f_extra!=None:
					error_prediction += [evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[f_extra], printerror=False)[0]]
				
				if calculate_error:		#calculate_error==True -> Cada vez que ha pasado por todas las imagenes, calcula el error con el resto
					#Calcula el error, aciertos y aciertos(%) y lo anade a sus correspondientes vectores 
					err,hit,tot = evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[k for k in range(N_FONTS) if not k in fonts], printerror=False)
					error		 += [err]
					hits		 += [hit]
					hits_percent += [100.*hit/tot]
					if n_iterations > 1:
						if error[n_iterations-1] < error[n_iterations-2]:
							ETA += A
						elif error[n_iterations-1] > error[n_iterations-2]:
							ETA += -1.*B*ETA
				
				if calculate_train_error:
					training_error += [evaluate(N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], fonts, printerror=False)[0]]
		
			#Criterio de convergencia
			if converge_criteria:
				if evaluate(N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], fonts, printerror=False)[0] < error_criteria:
					not_converged = False
				elif n_iterations >= max_iterations:
					not_converged = False
					n_iterations += 1
			
			elif n_iterations >= max_iterations:
				not_converged = False
			
			if not not_converged:
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
					return training_error, error, hits, hits_percent, error_prediction, n_iterations
				else:		#save==False -> Devuelve como parametros la red y el error
					return [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], [training_error, error, hits, hits_percent, error_prediction, n_iterations]

def back_prop_all(N, W, fonts, n_fonts=1, f_extra=None, title='Network.h5', verbose=False, save=False, calculate_error=False):
	global ITERATIONS,ITERATIONS_ALL,N_FONTS,BETA,ETA,ETA_ALL,GAMMA,ALPHA,A,B
	
	Ideal = np.eye(10)
	
	#Reads network parameters
	w01 = W[0][0]; w12 = W[0][1]; w23 = W[0][2];
	a01 = W[1][0]; a12 = W[1][1]; a23 = W[1][2];
	th1 = W[2][0]; th2 = W[2][1]; th3 = W[2][2];

	#Begins network computing
	error = []	
	hits = []
	hits_percent = []
	error_prediction = []
	for n in range(ITERATIONS_ALL):
		
		if n>0:
			Deltaw01_,Deltaw12_,Deltaw23_ = Deltaw01[:],Deltaw12[:],Deltaw23[:]
			Deltath1_,Deltath2_,Deltath3_ = Deltath1[:],Deltath2[:],Deltath3[:]
		else:
			Deltaw01_,Deltaw12_,Deltaw23_ = 0*w01,0*w12,0*w23
			Deltath1_,Deltath2_,Deltath3_ = 0*th1,0*th2,0*th3
		
		Deltaw01, Deltaw12, Deltaw23 = 0*w01, 0*w12, 0*w23
		Deltath1, Deltath2, Deltath3 = 0*th1, 0*th2, 0*th3
		
		for j in np.random.choice(fonts,n_fonts,replace=False):
			for i in range(10):
				
				if verbose:
					print str(n+1)+'/'+str(ITERATIONS_ALL)+': data/img'+str(i)+'{0:03}'.format(j)+'.bmp'
				
				img = (255. - np.flip( misc.imread('data/img'+str(i)+'{0:03}'.format(j)+'.bmp',flatten=1) , 0 )) / 255.
				Dw01,Dw12,Dw23,Dth1,Dth2,Dth3 = iteration(img, Ideal[i], N, w01, w12, w23, a01, a12, a23, th1, th2, th3, verbose)
				
				Deltaw01 += Dw01
				Deltaw12 += Dw12
				Deltaw23 += Dw23
				Deltath1 += Dth1
				Deltath2 += Dth2
				Deltath3 += Dth3
		
		w01 += Deltaw01 + ALPHA * Deltaw01_ - ETA_ALL*GAMMA * w01/((1+w01**2)**2)
		w12 += Deltaw12 + ALPHA * Deltaw12_ - ETA_ALL*GAMMA * w12/((1+w12**2)**2)
		w23 += Deltaw23 + ALPHA * Deltaw23_ - ETA_ALL*GAMMA * w23/((1+w23**2)**2)
		th1 += Deltath1 + ALPHA * Deltath1_ - ETA_ALL*GAMMA * th1/((1+th1**2)**2)
		th2 += Deltath2 + ALPHA * Deltath2_ - ETA_ALL*GAMMA * th2/((1+th2**2)**2)
		th3 += Deltath3 + ALPHA * Deltath3_ - ETA_ALL*GAMMA * th3/((1+th3**2)**2)
		
		if f_extra!=None:
			error_prediction += [evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[f_extra], printerror=False)[0]]
		
		if calculate_error:		#calculate_error==True -> Cada vez que ha pasado por todas las imagenes, calcula el error con el resto
			#Calcula el error, aciertos y aciertos(%) y lo anade a sus correspondientes vectores 
			err,hit,tot = evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],[k for k in range(N_FONTS) if not k in fonts], printerror=False)
			error		 += [err]
			hits		 += [hit]
			hits_percent += [100.*hit/tot]
			if n > 0:
				if error[n] < error[n-1]:
					ETA_ALL += A
				else:
					ETA_ALL += -1.*B*ETA_ALL
			print ETA_ALL
	
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
	global ITERATIONS,ITERATIONS_ALL,N_FONTS,BETA,ETA,ETA_ALL,GAMMA,ALPHA,A,B
	
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
				print 'Maximum value found found in ', max_val,'\n'
	
	if not len(fonts) == 0:
		error = 1.*error/(10*len(fonts))
	else:
		error = 0.
	
	if printerror:
		print 'Total:',aciertos+fallos
		print 'Aciertos:',aciertos
		print 'Fallos:',fallos
		print 'Error cuadratico:',error
	
	return error, aciertos, aciertos+fallos

def set_rand_omega(N):
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
	
	return [[w01, w12, w23],[a01, a12, a23],[th1, th2, th3]]