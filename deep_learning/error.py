import numpy as np
import random
import matplotlib.pyplot as plt
import h5py as h5
import back_prop as bp

def set_rand_omega(N):
	global w01, w12, w23, a01, a12, a23, th1, th2, th3
	N = np.array([dim*dim,49,16,10])
	#N = np.array([dim*dim,16,16,10])
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

#### CODIGO IMPORTANTE #####
T = 30
n_omegas = 1000
dim = 28
N = np.array([dim*dim,49,16,10])

error_av = np.zeros((int(T/10),bp.ITERATIONS))
hits_av = np.zeros((int(T/10),bp.ITERATIONS))
error_prediction_av = np.zeros((int(T/10),bp.ITERATIONS))

for i in range(n_omegas):
	print '\n###',i,'###'
	set_rand_omega(N)
	
	error = []
	hits = []
	error_prediction = []
	err_pred = []

	print 0

	fonts = []
	font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
	err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra)[1][3]]

	for n_fonts in np.arange(1,T+1,1):
		print n_fonts
		if((n_fonts+1)%10==0):
			fonts += [font_extra]
			font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
			err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,calculate_error=True)[1][3]]
			
			error_prediction += [1./n_fonts * (np.array(err_pred)).sum(axis=0)]
		
		elif(n_fonts%10==0):
			fonts += [font_extra]
			font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
			err,aux,hit,errp = bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,calculate_error=True)[1]
			err_pred += [errp]
		
			error += [err]
			hits += [hit]
			del err, aux, hit, errp
		
		else:
			fonts += [font_extra]
			font_extra = np.random.choice([k for k in range(bp.N_FONTS) if not k in fonts],1)[0]
			err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra)[1][3]]
	
	error_av += error
	hits_av += hits
	error_prediction_av += error_prediction
	
	print 'Generando archivo...'
	fout = h5.File('Error_night.h5','w')
	fout['error'] = 1.*error_av/(i+1)
	fout['hits_percent'] = 1.*hits_av/(i+1)
	fout['error_prediction'] = 1.*error_prediction_av/(i+1)
	fout.close()
	print 'Done'

error_av = 1.*error_av/n_omegas
hits_av = 1.*hits_av/n_omegas
error_prediction_av = 1.*error_prediction_av/n_omegas

#############################

plt.figure()
n=0
for i in error_av:
	plt.plot(i, label = 'error T='+str(10*(n+1)))
	n += 1
n=0
for i in error_prediction_av:
	plt.plot(i, label = 'prediction error T='+str(10*(n+1)))
	n += 1
plt.legend()
plt.figure()
n=0
for i in hits_av:
	plt.plot(i, label='hits(%) T='+str(10*(n+1)))
	n += 1
plt.legend()
plt.show()

