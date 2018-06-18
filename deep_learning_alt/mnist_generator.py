import mnist
import h5py as h5

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

fout = h5.File('mnist.h5','w')

fout['train_images'] = train_images
fout['train_labels'] = train_labels
fout['test_images'] = test_images
fout['test_labels'] = test_labels

fout.close()