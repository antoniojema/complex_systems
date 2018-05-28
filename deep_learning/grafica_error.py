import numpy as np
import random
import matplotlib.pyplot as plt
import h5py as h5
import back_prop as bp

N_FONTS = 1000

dim = 28
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

values = np.arange(50,900+1,50)
error_training = []
error_test = []

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
T = 20
err_pred = []

print 1
fonts = np.random.choice(range(N_FONTS),1,replace=False).tolist()
font_extra = np.random.choice([k for k in range(N_FONTS) if not k in fonts],1)[0]
err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra)[1][3]]

for n_fonts in np.arange(2,T,1):
	print n_fonts
	fonts += [font_extra]
	font_extra = np.random.choice([k for k in range(N_FONTS) if not k in fonts],1)[0]
	err_pred += [bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra)[1][3]]

print T
fonts += [font_extra]
font_extra = np.random.choice([k for k in range(N_FONTS) if not k in fonts],1)[0]
err,hit,hit_perc,aux = bp.back_prop(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,f_extra=font_extra,calculate_error=True)[1]
err_pred += [aux]; del aux;

#############################

fout = h5.File('ErrorT'+str(T)+'.h5','w')
fout['error'] = err
fout['hits'] = hit
fout['hits_percent'] = hit_perc
fout['error_prediction'] = err_pred
fout.close()

plt.figure()
plt.plot(err, label='error')
plt.plot(1./T * (np.array(err_pred)).sum(axis=0), label='prediction error')
plt.legend()
plt.figure()
plt.plot(hit_perc, label='hits(%)')
plt.legend()
plt.show()

