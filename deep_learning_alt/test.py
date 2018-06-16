import numpy as np
import random
import h5py as h5
import back_prop as bp
from scipy import misc
import time
import matplotlib.pyplot as plt

'''
fin = h5.File('Best.h5','r')
fonts = fin.attrs['fonts']
N = fin.attrs['N']
w01 = fin['w01'][:]
w12 = fin['w12'][:]
w23 = fin['w23'][:]
a01 = fin['a01'][:]
a12 = fin['a12'][:]
a23 = fin['a23'][:]
th1 = fin['th1'][:]
th2 = fin['th2'][:]
th3 = fin['th3'][:]
fin.close()

bp.evaluate(N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]], np.random.choice(range(1000),300,replace=False))

img = (255. - np.flip( misc.imread('test/img8.bmp',flatten=1) , 0 )) / 255.
V0 = img.reshape(-1)
H1 = np.array([ np.dot((a01*w01)[k][:] , V0[:]) for k in range(N[1])]) - th1
V1 = bp.sigma( H1 )
H2 = np.array([ np.dot((a12*w12)[k][:] , V1[:]) for k in range(N[2])]) - th2
V2 = bp.sigma( H2 )
H3 = np.array([ np.dot((a23*w23)[k][:] , V2[:]) for k in range(N[3])]) - th3
V3 = bp.sigma( H3 )
max_val = V3.tolist().index(np.amax(V3))

print 1
print V3
print "Maximum value found found in ", max_val,"\n"
'''

dim =28
N=[dim*dim,16,16,10]
W = bp.set_rand_omega(N)

fonts = range(300)
train_error,caca1,caca2,caca3,caca4,n_iterations = bp.back_prop(N, W, fonts, calculate_train_error=True,converge=True,verbose=True)[1]
del caca1,caca2,caca3,caca4

print n_iterations

plt.plot(train_error)
plt.show()