#############################################################
##                                                         ##
##                       OJO CUIDAO!                       ##
##           Este código no tiene sentido si no            ##
##             se activa calculate_train_error             ##
##                                                         ##
#############################################################
import numpy as np
import h5py as h5
import back_prop as bp
import time

Tmax = 7000/10
DT = 1000/10
n_fonts_error = 2000/10
max_iterations = 20000

n_omegas = 100
dim = 28
N = np.array([dim*dim,16,16,10])

for i in range(n_omegas):
	print '\n###',i,'###'
	
	for A in [0.3,0.5,0.7]:
		for B in [0.3,0.7]:
			bp.A = A
			bp.B = B
			print '\n# A =',A,'#\n# B =',B,'#'
			
			training_error = []
			test_error = []
			hits_perc = []
			
			for T in np.arange(DT,Tmax+1,DT):
				print 'T =',T
				t0 = time.time()
				
				[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.set_rand_omega(N)
				fonts = np.random.choice(range(700),T,replace=False)
				
				tr_err, ts_err, hits = bp.back_prop(
				N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,max_iterations=max_iterations,n_fonts_error=n_fonts_error,calculate_error=True,calculate_train_error=True)[1][:3]
				
				#[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.back_prop(
				#N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,max_iterations=max_iterations,n_fonts_error=n_fonts_error)[0]
				
				#tr_err = bp.evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,printerror=False)[0]
				#ts_err, hits = bp.evaluate(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],np.arange(700,1000,1),printerror=False)[:2]
				
				training_error += [tr_err]
				test_error += [ts_err]
				hits_perc += [(1.*np.array(hits)/30).tolist()]
				
				print time.time()-t0,'s'
			
			print 'Generando archivo...'
			if i!=0:
				fin = h5.File('Error_adaptative_'+str(A)+'_'+str(B)+'.h5','r')
				training_error = 1.*( (fin['training_error'][:])*i + np.array(training_error) ) / (i+1)
				test_error = 1.*( (fin['test_error'][:])*i + np.array(test_error) ) / (i+1)
				hits_perc = 1.*( (fin['hits_percent'][:])*i + np.array(hits_perc) ) / (i+1)
				fin.close()
			
			fout = h5.File('Error_adaptative_'+str(A)+'_'+str(B)+'.h5','w')
			fout.attrs['n_omegas'] = i+1
			fout.attrs['T'] = np.arange(DT,Tmax+1,DT)
			fout.attrs['max_iterations'] = max_iterations
			fout.attrs['n_fonts_error'] = n_fonts_error
			fout['training_error'] = training_error
			fout['test_error'] = test_error
			fout['hits_percent'] = hits_perc
			fout.close()
			print 'Done'
