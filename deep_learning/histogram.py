import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from mpl_toolkits.mplot3d import Axes3D

dim=28
fin = h5.File('Network1.h5','r')

M01 = fin['M01'][:]
M12 = fin['M12'][:]
M23 = fin['M23'][:]

plt.hist(M23, 'auto', facecolor='g')
'''
x=M23
fig = plt.figure()
ax = fig.gca(projection='3d')
I = [[i for j in range(len(x[0]))] for i in range(len(x))]
J = [[j for j in range(len(x[0]))] for i in range(len(x))]
ax.plot_surface(I,J,x)
'''
plt.show()
