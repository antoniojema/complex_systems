import numpy as np
import h5py as h5
import back_prop as bp
import time
import architecture_selection as arch

Tmax = 6000
DT = 1000
bp.N_IMAGES_ERROR = 1000
'''
#PROTEUS
Tmax = 60000
DT = 10000
'''

n_omegas = 10
dim = 28
N = np.array([dim*dim,16,16,10])

training_error_av = np.zeros((int(Tmax/DT) , int(bp.MAX_ITERATIONS/bp.N_IMAGES_ERROR)))
error_av = np.zeros((int(Tmax/DT) , int(bp.MAX_ITERATIONS/bp.N_IMAGES_ERROR)))
hits_av = np.zeros((int(Tmax/DT) , int(bp.MAX_ITERATIONS/bp.N_IMAGES_ERROR)))

for i in range(n_omegas):
	print '\n###',i,'###'
	
	for architecture in ['2x2','4x4','7x7','14x14','solapado'#,'solapado+'
	]:
		print '\n# Architecture =',architecture,'#'
		
		N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = arch.architecture(architecture)
		
		training_error = []
		error = []
		hits = []
		
		for T in np.arange(DT,Tmax+1,DT):
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
		fout = h5.File('Error_architecture_'+architecture+'.h5','w')
		fout.attrs['n_omegas'] = i+1
		fout['training_error'] = 1.*training_error_av/(i+1)
		fout['error'] = 1.*error_av/(i+1)
		fout['hits_percent'] = 1.*hits_av/(i+1)
		fout.close()
		print 'Done'
