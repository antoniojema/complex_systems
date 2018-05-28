import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.arange(-10,15,0.2)
y = np.arange(-10,10,0.2)
x,y = np.meshgrid(x,y)

z = 5*np.sin(x)+5*np.cos(0.8*y)+0.5*x+0.1*x**2+0.01+0.01*y**3

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(x,y,z,cmap=cm.viridis, antialiased=False)
plt.show()
