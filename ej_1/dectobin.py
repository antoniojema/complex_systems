import numpy as np

def printbin(x):
	if x > 0:
		n = int(np.log2(x))
		a = ""
		if n < 0:
			n = 0
		
		while(True):
			if n != -1:
				if x >= 2**n:
					a = a + "1"
					x = x - 2**n
				else:
					a = a + "0"
			else:
				a = a + "."
				if x >= 2**n:
					a = a + "1"
					x = x - 2**n
				else:
					a = a + "0"
			n = n - 1
			if (x <= 0 and n<0):
				print a
				return
	elif x == 0:
		print "0"
	else:
		x = -1.*x
		n = int(np.log2(x))
		a = ""
		if n < 0:
			n = 0
		
		while(True):
			if n != -1:
				if x >= 2**n:
					a = a + "1"
					x = x - 2**n
				else:
					a = a + "0"
			else:
				a = a + "."
				if x >= 2**n:
					a = a + "1"
					x = x - 2**n
				else:
					a = a + "0"
			n = n - 1
			if (x <= 0 and n<0):
				print "-" + a
				return
