import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import back_prop as bp

values = np.arange(50,900+1,50)
error_training = []
error_test = []

for i in values:
	print i
	error_training += [bp.back_prop(i,title='Network_4x4blocks.h5')]
	error_test += [bp.evaluate(title='Network_4x4blocks.h5')]

fout = h5.File('Error.h5','w')
fout['error_training'] = error_training
fout['error_test'] = error_test
fout.close()

plt.figure()
for err in error_training:
	plt.plot(err)
plt.figure()
plt.plot(error_test)
plt.show()
