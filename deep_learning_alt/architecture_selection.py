import numpy as np
import random

def architecture(dimension):

	if dimension == '14x14':
		#######################################################################################
		#                                                                                     #
		#                         ARQUITECTURA 1: 2*2 partes de 14x14                         #
		#                                                                                     #
		#######################################################################################
		
		dim=28
		N = np.array([dim*dim,2*2,16,10])
		#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
		w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
		w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
		w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
		#Alive matrix. If TRUE it means weigh exists
		a01 = np.array([[False for j in range(N[0])] for i in range(N[1])])
		a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
		a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
		#Thresholds
		th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
		th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
		th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])
		
		for n in range(2):
			for m in range(2):
				for i in range(14):
					for j in range(14):
						a01[2*n+m][28*(14*n+i)+(14*m+j)] = True
	
	
	elif dimension == '7x7':
		#######################################################################################
		#                                                                                     #
		#                          ARQUITECTURA 2: 4*4 partes de 7x7                          #
		#                                                                                     #
		#######################################################################################
		
		dim=28
		N = np.array([dim*dim,4*4,16,10])
		#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
		w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
		w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
		w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
		#Alive matrix. If TRUE it means weigh exists
		a01 = np.array([[False for j in range(N[0])] for i in range(N[1])])
		a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
		a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
		#Thresholds
		th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
		th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
		th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])
	
		for n in range(4):
			for m in range(4):
				for i in range(7):
					for j in range(7):
						a01[4*n+m][28*(7*n+i)+(7*m+j)] = True
	
	
	elif dimension == '4x4':
		#######################################################################################
		#                                                                                     #
		#                          ARQUITECTURA 3: 7*7 partes de 4x4                          #
		#                                                                                     #
		#######################################################################################
		
		dim=28
		N = np.array([dim*dim,7*7,16,10])
		#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
		w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
		w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
		w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
		#Alive matrix. If TRUE it means weigh exists
		a01 = np.array([[False for j in range(N[0])] for i in range(N[1])])
		a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
		a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
		#Thresholds
		th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
		th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
		th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])
	
		for n in range(7):
			for m in range(7):
				for i in range(4):
					for j in range(4):
						a01[7*n+m][28*(4*n+i)+(4*m+j)] = True
	
	
	elif dimension == '2x2':
		#######################################################################################
		#                                                                                     #
		#                         ARQUITECTURA 4: 14*14 partes de 2x2                         #
		#                                                                                     #
		#######################################################################################
		
		dim=28
		N = np.array([dim*dim,14*14,16,10])
		#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
		w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
		w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
		w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
		#Alive matrix. If TRUE it means weigh exists
		a01 = np.array([[False for j in range(N[0])] for i in range(N[1])])
		a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
		a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
		#Thresholds
		th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
		th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
		th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])
	
		for n in range(14):
			for m in range(14):
				for i in range(2):
					for j in range(2):
						a01[14*n+m][28*(2*n+i)+(2*m+j)] = True
	
	
	elif dimension == 'solapado':
		#######################################################################################
		#                                                                                     #
		#                          ARQUITECTURA 5: Con solapamientos                          #
		#                                                                                     #
		#######################################################################################
		
		dim=28
		N = np.array([dim*dim,7*7+6*7*2,16,10])
		#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
		w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
		w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
		w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
		#Alive matrix. If TRUE it means weigh exists
		a01 = np.array([[False for j in range(N[0])] for i in range(N[1])])
		a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
		a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
		#Thresholds
		th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
		th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
		th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])
	
		for n in range(7):
			for m in range(7):
				for i in range(4):
					for j in range(4):
						a01[7*n+m][28*(4*n+i)+(4*m+j)] = True
		
		for n in range(7):
			for m in range(6):
				for i in range(4):
					for j in range(4):
						a01[7*7+6*n+m][28*(4*n+i)+(4*m+i+2)] = True
						a01[7*7+7*6+7*m+n][28*(4*m+i+2)+(4*n+i)] = True
	
	
	elif dimension == 'solapado+':
		#######################################################################################
		#                                                                                     #
		#                        ARQUITECTURA 6: Con mas solapamientos                        #
		#                                                                                     #
		#######################################################################################
		
		dim=28
		N = np.array([dim*dim,7*7+6*7*2+6*6,16,10])
		#Weighs matrix. wab[i][j] is the weigh of a[j] for b[i]
		w01 = np.array([[2.*(random.random()-0.5) for j in range(N[0])] for i in range(N[1])])
		w12 = np.array([[2.*(random.random()-0.5) for j in range(N[1])] for i in range(N[2])])
		w23 = np.array([[2.*(random.random()-0.5) for j in range(N[2])] for i in range(N[3])])
		#Alive matrix. If TRUE it means weigh exists
		a01 = np.array([[False for j in range(N[0])] for i in range(N[1])])
		a12 = np.array([[True for j in range(N[1])] for i in range(N[2])])
		a23 = np.array([[True for j in range(N[2])] for i in range(N[3])])
		#Thresholds
		th1 = np.array([2.*(random.random()-0.5) for i in range(N[1])])
		th2 = np.array([2.*(random.random()-0.5) for i in range(N[2])])
		th3 = np.array([2.*(random.random()-0.5) for i in range(N[3])])
	
		for n in range(7):
			for m in range(7):
				for i in range(4):
					for j in range(4):
						a01[7*n+m][28*(4*n+i)+(4*m+j)] = True
		
		for n in range(7):
			for m in range(6):
				for i in range(4):
					for j in range(4):
						a01[7*7+6*n+m][28*(4*n+i)+(4*m+i+2)] = True
						a01[7*7+7*6+7*m+n][28*(4*m+i+2)+(4*n+i)] = True
		
		for n in range(6):
			for m in range(6):
				for i in range(4):
					for j in range(4):
						a01[7*7+7*6*2+6*n+m][28*(4*n+i+2)+(4*m+i+2)] = True
	
	
	return N, [[w01,w12,w23],[a01,a12,a23],[th1,th2,th3]]
