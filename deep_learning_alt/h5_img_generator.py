import numpy as np
import h5py as h5
from scipy import misc

img = []
for i in range(10):
	for j in range(1000):
		img += [misc.imread('data/img'+str(i)+'{0:03}'.format(j)+'.bmp',flatten=1)]

fout = h5.File('images.h5','w')
fout.attrs['shape'] = np.array(img).shape
fout['images'] = img
fout.close()