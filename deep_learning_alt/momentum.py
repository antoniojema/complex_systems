import numpy as np
import random
import matplotlib.pyplot as plt
import h5py as h5
import back_prop as bp
import time

T = 50
n_omegas = 10
dim = 28
N = np.array([dim*dim,16,16,10])

training_error_av = np.zeros((int(T/10),int(bp.MAX_ITERATIONS/(10*bp.N_IMAGES_ERROR))))
error_av = np.zeros((int(T/10),int(bp.MAX_ITERATIONS/(10*bp.N_IMAGES_ERROR))))
hits_av = np.zeros((int(T/10),int(bp.MAX_ITERATIONS/(10*bp.N_IMAGES_ERROR))))

for i in range(n_omegas):
	print '\n###',i,'###'
	
	for momentum in [0.2,0.4,0.6,0.8,1]:
		bp.ALPHA=momentum
		print '\n# ALPHA =',momentum,'#'
		
		[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.set_rand_omega(N)
		
		training_error = []
		error = []
		hits = []
		
		for T in np.arange(10000,60000+1,10000):
			print 'T =',T
			t0 = time.time()
			
			images = np.random.choice(range(bp.N_IMAGES_TRAIN),T,replace=False)
			
			tr_err,err,aux,hit = bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],images,calculate_error=True,calculate_train_error=True)[1][:4]
			
			training_error += [tr_err]
			error += [err]
			hits += [hit]
			del tr_err, err, aux, hit
			
			print time.time()-t0,'s'
		
		
		training_error_av += training_error
		error_av += error
		hits_av += hits
		
		print 'Generando archivo...'
		fout = h5.File('Error_momentum_'+str(momentum)+'.h5','w')
		fout.attrs['n_omegas'] = i+1
		fout['training_error'] = 1.*training_error_av/(i+1)
		fout['error'] = 1.*error_av/(i+1)
		fout['hits_percent'] = 1.*hits_av/(i+1)
		fout.close()
		print 'Done'
