import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

P = np.array([[0.,0.],[1.,0.],[0.5,np.sqrt(3.)/2]])

N = 100000
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
x[0] = 0.5
y[0] = np.sqrt(3.)/4
z[0] = 0.5

p1 = 0.5
p2 = 1.-p1
for i in np.arange(1,N,1):
	x[i],y[i] = p1 * np.array([x[i-1],y[i-1]]) + p2 * P[int(len(P)*random.random())] 

fig = plt.figure()
#ax = fig.gca(projection='3d')
plt.plot(x,y,'o',markersize=0.5,color='black')
plt.show()
		

