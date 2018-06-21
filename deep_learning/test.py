import numpy as np
import back_prop as bp

dim =28
N=[dim*dim,16,16,10]
W = bp.set_rand_omega(N)

fonts = range(700)
W = bp.back_prop(N,W,fonts,verbose=True)[0]

print np.array([abs(W[0][0]).max(),abs(W[0][1]).max(),abs(W[0][2]).max()]).max()
print np.array([abs(W[0][0]).min(),abs(W[0][1]).min(),abs(W[0][2]).min()]).min()

bp.evaluate(N,W,np.arange(700,1000,1))