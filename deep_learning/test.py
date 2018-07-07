import numpy as np
import back_prop as bp

dim =28
N=[dim*dim,16,16,10]
W = bp.set_rand_omega(N)

fonts = range(700)
W = bp.back_prop(N,W,fonts,verbose=True)[0]

bp.evaluate(N,W,np.arange(700,1000,1))