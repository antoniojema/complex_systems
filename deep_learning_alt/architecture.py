import numpy as np
import h5py as h5
import back_prop as bp
import time
import architecture_selection as arch

Tmax = 7000/10
DT = 1000/10
n_fonts_error = 500/10
max_iterations = 14000

n_omegas = 100
dim = 28

for i in range(n_omegas):
	print '\n###',i,'###'
	
	for architecture in ['2x2','4x4','7x7','14x14'#,'solapado','solapado+'
	]:
		print '\n# ARCHITECTURE =',architecture,'#'
		
		training_error = []
		test_error = []
		hits_perc = []
		
		for T in np.arange(DT,Tmax+1,DT):
			print 'T =',T
			t0 = time.time()
			
			N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = arch.architecture(architecture)
			fonts = np.random.choice(range(700),T,replace=False)
			
			[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,max_iterations=max_iterations,n_fonts_error=n_fonts_error)[0]
			
			tr_err = bp.evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,printerror=False)[0]
			ts_err, hits = bp.evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],np.arange(700,1000,1),printerror=False)[:2]
			
			print tr_err,ts_err,hits
			
			training_error += [tr_err]
			test_error += [ts_err]
			hits_perc += [1.*hits/30]
			
			print time.time()-t0,'s'
		
		print 'Generando archivo...'
		if i!=0:
			fin = h5.File('Error_architecture_'+str(architecture)+'.h5','r')
			training_error = 1.*( (fin['training_error'][:])*i + np.array(training_error) ) / (i+1)
			test_error = 1.*( (fin['test_error'][:])*i + np.array(test_error) ) / (i+1)
			hits_perc = 1.*( (fin['hits_percent'][:])*i + np.array(hits_perc) ) / (i+1)
			fin.close()
		
		fout = h5.File('Error_architecture_'+str(architecture)+'.h5','w')
		fout.attrs['n_omegas'] = i+1
		fout.attrs['T'] = np.arange(DT,Tmax+1,DT)
		fout['training_error'] = training_error
		fout['test_error'] = test_error
		fout['hits_percent'] = hits_perc
		fout.close()
		print 'Done'
