import numpy as np
import random
import matplotlib.pyplot as plt
import h5py as h5
import back_prop as bp
import time

Tmax = 700/10
DT = 100/10
n_fonts_error = 2000/10
max_iterations = 20000

n_omegas = 100
dim = 28
N = np.array([dim*dim,16,16,10])

for i in range(n_omegas):
	print '\n###',i,'###'
	
	training_error = []
	test_error = []
	hits_perc = []
	error_prediction = []
	err_pred = []
	
	print 'T =',0
	t0=time.time()
	
	[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.set_rand_omega(N)
	
	fonts = []
	font_extra = np.random.choice([k for k in range(700) if not k in fonts],1)[0]
	err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,max_iterations=max_iterations,n_fonts_error=n_fonts_error)[1][4]]
	
	print time.time()-t0,'s'
	
	for T in np.arange(1,Tmax+1,1):
		print 'T =',T
		t0 = time.time()
		
		[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.set_rand_omega(N)
		
		if (T+1)%DT == 0:
			fonts += [font_extra]
			font_extra = np.random.choice([k for k in range(700) if not k in fonts],1)[0]
			err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,max_iterations=max_iterations,n_fonts_error=n_fonts_error)[1][4]]
			
			error_prediction += [1./(T+1) * (np.array(err_pred)).sum(axis=0)]
		
		elif T%DT == 0:
			fonts += [font_extra]
			font_extra = np.random.choice([k for k in range(700) if not k in fonts],1)[0]
			tr_err,ts_err,aux,hits,errp = bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,calculate_error=True,calculate_train_error=True,max_iterations=max_iterations,n_fonts_error=n_fonts_error)[1][:5]
			err_pred += [errp]
			
			training_error += [tr_err]
			test_error += [ts_err]
			hits_perc += [(1.*np.array(hits)/30).tolist()]
			del tr_err, ts_err, aux, hits, errp
		
		else:
			fonts += [font_extra]
			font_extra = np.random.choice([k for k in range(700) if not k in fonts],1)[0]
			err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,max_iterations=max_iterations,n_fonts_error=n_fonts_error)[1][4]]
		
		print time.time()-t0,'s'
	
	print 'Generando archivo...'
	if i!=0:
		fin = h5.File('Error_prediction.h5','r')
		training_error = 1.*( (fin['training_error'][:])*i + np.array(training_error) ) / (i+1)
		test_error = 1.*( (fin['test_error'][:])*i + np.array(test_error) ) / (i+1)
		error_prediction = 1.*( (fin['prediction_error'][:])*i + np.array(error_prediction) ) / (i+1)
		hits_perc = 1.*( (fin['hits_percent'][:])*i + np.array(hits_perc) ) / (i+1)
		fin.close()
	
	fout = h5.File('Error_prediction.h5','w')
	fout.attrs['n_omegas'] = i+1
	fout.attrs['T'] = np.arange(DT,Tmax+1,DT)
	fout.attrs['max_iterations'] = max_iterations
	fout.attrs['n_fonts_error'] = n_fonts_error
	fout['training_error'] = training_error
	fout['test_error'] = test_error
	fout['prediction_error'] = error_prediction
	fout['hits_percent'] = hits_perc
	fout.close()
	print 'Done'
	
	'''
	training_error_av += training_error
	error_av += error
	hits_av += hits
	error_prediction_av += error_prediction
	
	print 'Generando archivo...'
	fout = h5.File('Error_momentum_0_2.h5','w')
	fout.attrs['n_omegas'] = i+1
	fout['training_error'] = 1.*training_error_av/(i+1)
	fout['error'] = 1.*error_av/(i+1)
	fout['hits_percent'] = 1.*hits_av/(i+1)
	fout['error_prediction'] = 1.*error_prediction_av/(i+1)
	fout.close()
	print 'Done'
	'''
