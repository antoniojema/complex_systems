import numpy as np
import random
import matplotlib.pyplot as plt
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

fonts = np.random.choice(range(N_FONTS),700,replace=False)

bp.ITERATIONS_ALL = 10000
bp.ETA_ALL = 1
bp.A = 0
bp.B = 0
bp.ALPHA = 0.5
W = bp.back_prop_all(N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,1,verbose=True)[0]
bp.evaluate(N, W, [k for k in range(N_FONTS) if not k in fonts])
