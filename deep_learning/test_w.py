import numpy as np
import back_prop as bp
import matplotlib.pyplot as plt

n_fonts_error = 2000/10
max_iterations = 20000
bp.GAMMA=0.2

dim=28
N = np.array([dim*dim,16,16,10])
[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]] = bp.set_rand_omega(N)
fonts = np.random.choice(range(700),300,replace=False)

tr_error, ts_error = bp.back_prop(
	N,[[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]],fonts,max_iterations=max_iterations,n_fonts_error=n_fonts_error,calculate_error=True,calculate_train_error=True,verbose=True)[1][:2]

plt.plot(ts_error,label='Test error')
plt.plot(tr_error,label='Training error')
plt.legend()
plt.show()