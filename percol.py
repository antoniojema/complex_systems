import random
import matplotlib.pyplot as plt
import pylab

NX = 100
NY = 100
p = 1

def value():
	if random.random() < p:
		return 0
	else:
		return 1

M = [[value() for j in range(NY)] for i in range(NX)]

pylab.pcolor(M)
pylab.colorbar()
plt.show()
