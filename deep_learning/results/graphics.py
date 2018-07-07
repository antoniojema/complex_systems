import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import subprocess
import sys

ARG = sys.argv[1]

if ARG == 'hits':
	files = (subprocess.check_output(['ls'])).splitlines()
	omegas_file = open('omegas_hits.txt','w')
	
	for filename in files:
	#for filename in [files[0]]:
		if filename[len(filename)-3:] == '.h5' and not 'prediction' in filename:
			print '\n',filename
			fin = h5.File(filename,'r')
			omegas_file.write('\n'+filename+'	'+str(fin.attrs['n_omegas']))
			print 'Omegas:',fin.attrs['n_omegas']
			
			hits = 10 * (np.array( [(0.1*(np.random.random(7)-0.5)+1).tolist()] + (0.1*fin['hits_percent'][:]).T.tolist() )).T
			
			x = np.array([np.arange(i*len(hits[0])-i , (i+1)*len(hits[0])-i) for i in range(7)]) * 10*fin.attrs['n_fonts_error']
			
			### PLOT ###
			
			plt.figure()
			plt.title(filename.replace('_',' ').replace('.h5',''))
			plt.xlim(0,fin.attrs['max_iterations']*7)
			plt.ylim(0,100)
			plt.xlabel('Iterations')
			plt.ylabel('Hits (%)')
			for i in range(7):
				plt.plot(x[i], hits[i],color='blue')
				plt.axvline(x=x[i][len(hits[0])-1],color='black',linewidth=1.0)
			for i in range(7):
				plt.text(2500+i*20000,95,'T='+str((i+1)*1000))
			###
			
			fin.close()
			plt.savefig('images/'+filename.replace('.h5','_hits.png'))
			#plt.show()
			plt.clf()
	
	omegas_file.close()

elif ARG == 'error':
	files = (subprocess.check_output(['ls'])).splitlines()
	omegas_file = open('omegas.txt','w')
	
	for filename in files:
	#for filename in [files[0]]:
		if filename[len(filename)-3:] == '.h5':
			print '\n',filename
			fin = h5.File(filename,'r')
			omegas_file.write('\n'+filename+'	'+str(fin.attrs['n_omegas']))
			print 'Omegas:',fin.attrs['n_omegas']
			
			test_error = (np.array( [(0.1*(np.random.random(7)-0.5)+0.9).tolist()] + (0.1*fin['test_error'][:]).T.tolist() )).T
			train_error = (np.array( [(0.1*(np.random.random(7)-0.5)+0.9).tolist()] + (0.1*fin['training_error'][:]).T.tolist() )).T
			if 'prediction' in filename:
				pred_error = (np.array( [(0.1*(np.random.random(7)-0.5)+0.9).tolist()] + (0.1*fin['prediction_error'][:]).T.tolist() )).T
			
			x = np.array([np.arange(i*len(test_error[0])-i , (i+1)*len(test_error[0])-i) for i in range(7)]) * 10*fin.attrs['n_fonts_error']
			
			### PLOT ###
			
			plt.figure()
			plt.title(filename.replace('_',' ').replace('.h5',''))
			plt.xlim(0,fin.attrs['max_iterations']*7)
			plt.ylim(0,0.1)
			plt.xlabel('Iterations')
			plt.ylabel('Error')
			for i in range(6):
				plt.plot(x[i], test_error[i],color='blue')
				plt.plot(x[i], train_error[i],color='red')
				if 'prediction' in filename:
					plt.plot(x[i],pred_error[i],color='green')
				plt.axvline(x=x[i][len(test_error[0])-1],color='black',linewidth=1.0)
			plt.plot(x[6], test_error[6],color='blue',label='Test error')
			plt.plot(x[6], train_error[6],color='red',label='Train error')
			if 'prediction' in filename:
				plt.plot(x[6],pred_error[6],color='green',label='Prediction error')
			#plt.legend(loc=(100400,0.08))
			if 'prediction' in filename:
				plt.legend(loc=(0.665,0.75),framealpha=1)
			else:
				plt.legend(loc=(0.735,0.8),framealpha=1)
			for i in range(7):
				if 'prediction' in filename:
					plt.text(3500+i*20000,0.095,'T='+str((i+1)*100))
				else:
					plt.text(2500+i*20000,0.095,'T='+str((i+1)*1000))
			###
			
			fin.close()
			plt.savefig('images/'+filename.replace('.h5','.png'))
			#plt.show()
			plt.clf()
	
	omegas_file.close()

elif ARG == 'lyapunov':
	fin = h5.File('lyapunov.h5','r')
	data = fin['lyapunov'][:]
	fin.close()
	
	plt.hist(data,50)
	plt.show()

else:
	print 'Argumento erroneo'