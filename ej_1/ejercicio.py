import numpy as np
import matplotlib.pyplot as plt
import dectobin as db

def f(x):
	if x < 0.5:
		return 2*x
	else:
		return 2*x-1

N=60
x = np.zeros(N+1)

goon=True
while(goon):
	inp = raw_input('Introduzca una semilla entre 0 y 1 (para semilla aleatoria escriba -1): ')
	inp = float(inp)
	if (inp >= 0 and inp <=1):
		x[0]=inp
		goon = False
	elif inp == -1:
		x[0]=np.random.rand(1)[0]
		goon = False

for i in range(N):
	db.printbin(x[i])
	x[i+1] = f(x[i])
db.printbin(x[N])

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

ax1.plot(range(N+1),x, "o", markersize=2)
ax1.set_xlabel("i")
ax1.set_ylabel("x(i)")

for i in range(N-1):
	a = [x[i],x[i+1]]
	b = [x[i+1],x[i+1]]
	ax2.plot(a,b,'r')
	a = [x[i+1],x[i+1]]
	b = [x[i+1],x[i+2]]
	ax2.plot(a,b,'r')

ax2.plot([0,1],[0,1],'g')
ax2.plot([0,0.5],[0,1],'b')
ax2.plot([0.5,1],[0,1],'b')
ax2.plot(x, np.insert( np.delete(x, 0, 0) , N, 0, axis=0), 'o', markersize=5)
ax2.set_xlabel("x(i)")
ax2.set_ylabel("x(i+1)")
plt.show()
