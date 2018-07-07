import back_prop as bp
import numpy as np
import h5py as h5

dim=28
N = [28*28,16,16,10]
max_iterations = 700*10*10

lyapunov = []
n=0
while True:
	W = np.array(bp.set_rand_omega(N))
	W_ = W + np.array([
		2e-10*(np.array([np.random.random((N[1],N[0])),np.random.random((N[2],N[1])),np.random.random((N[3],N[2]))])-0.5),
		0*W[1],
		2e-10*(np.array([np.random.random(N[1]),np.random.random(N[2]),np.random.random(N[3])])-0.5)
		])
	
	Wt = bp.back_prop(N, W, range(700),max_iterations=max_iterations)[0]
	W_t = bp.back_prop(N, W_, range(700),max_iterations=max_iterations)[0]
	
	num = 0
	den = 0
	for i in range(3):
		for j in range(3):
			num += ((Wt[i][j]-W_t[i][j])**2).sum()
			den += ((W[i][j]-W_[i][j])**2).sum()
	num = np.sqrt(num)
	den = np.sqrt(den)
	
	lyapunov += [(700.*10/max_iterations)*np.log(num/den)]
	n+=1
	
	#print lyapunov
	fout = h5.File('lyapunov.h5','w')
	fout.attrs['n_samples'] = n
	fout['lyapunov'] = lyapunov
	fout.close()
