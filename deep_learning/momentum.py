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

error_av = np.zeros((int(T/10),bp.ITERATIONS))
hits_av = np.zeros((int(T/10),bp.ITERATIONS))
error_prediction_av = np.zeros((int(T/10),bp.ITERATIONS))

for momentum in [0.2,0.4,0.6,0.8,1]:
	bp.ALPHA=momentum
	
	for i in range(n_omegas):
		print '\n###',i,'###'
		[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.set_rand_omega(N)
		
		training_error = []
		error = []
		hits = []
		error_prediction = []
		err_pred = []
		
		print 0
		
		t0=time.time()
		fonts = []
		font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
		err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra)[1][4]]
		print np.array(err_pred)
		print len(err_pred[0])
		print time.time()-t0,'s'
		
		for n_fonts in np.arange(1,T+1,1):
			print n_fonts
			
			t0 = time.time()
			if((n_fonts+1)%10==0):
				fonts += [font_extra]
				font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
				err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra)[1][4]]
				print np.array(err_pred)
				print len(err_pred[n_fonts])
				
				error_prediction += [1./n_fonts * (np.array(err_pred)).sum(axis=0)]
			
			elif(n_fonts%10==0):
				fonts += [font_extra]
				font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
				tr_err,err,aux,hit,errp = bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,calculate_error=True,calculate_train_error=True)[1]
				err_pred += [errp]
				print np.array(err_pred)
				print len(err_pred[n_fonts])
				
				training_error += [tr_err]
				error += [err]
				hits += [hit]
				del tr_err, err, aux, hit, errp
			
			else:
				fonts += [font_extra]
				font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
				
				err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra)[1][4]]
				print np.array(err_pred)
				print len(err_pred[n_fonts])
			
			print time.time()-t0,'s'
		
		
		error_av += error
		hits_av += hits
		error_prediction_av += error_prediction
		
		print 'Generando archivo...'
		fout = h5.File('Error_momentum_'+str(momentum)+'.h5','w')
		fout.attrs['n_omegas'] = i+1
		fout['training_error'] = 1.*training_error_av/(i+1)
		fout['error'] = 1.*error_av/(i+1)
		fout['hits_percent'] = 1.*hits_av/(i+1)
		fout['error_prediction'] = 1.*error_prediction_av/(i+1)
		fout.close()
		print 'Done'
