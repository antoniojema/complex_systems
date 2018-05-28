import back_prop as bp
import matplotlib.pyplot as plt

error_training = bp.back_prop(100,title='Network.h5',verbose=True)
error_test = bp.evaluate(title='Network.h5')
'''
print error_test
plt.plot(error_training)
plt.show
'''
