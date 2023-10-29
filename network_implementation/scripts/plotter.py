
import random

from copy import deepcopy

import matplotlib.pyplot as plt

from numpy import linalg as LA
import numpy as np
from copy import copy
from random import sample
import itertools

import operator


epochs = 100








selective_acc = list() 


for i in range(6):
	temp = list()
	with open('sresults/sacc'+ str(i+1) + '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp.append(float(current_number))  

	selective_acc.append(temp)   

selective_time = list()

for i in range(6):
	temp = list()
	with open('sresults/st'+ str(i+1) + '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp.append(float(current_number))  

	selective_time.append(temp)   



total_selective_time = list()

for i in range(6):	
	total_selective_time.append(np.cumsum(selective_time[i]))


random_acc = list() 

for i in range(2):
	temp = list()
	with open('random/rsacc'+ str(i+1) + '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp.append(float(current_number))  

	random_acc.append(temp)   



random_time = list()

for i in range(2):
	temp = list()
	with open('random/rst'+ str(i+1) + '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp.append(float(current_number))  

	random_time.append(temp) 


total_random_time = list()

for i in range(2):	
	total_random_time.append(np.cumsum(random_time[i]))



standard_acc = list() 

for i in range(2):
	temp = list()
	with open('standard/facc'+ str(i+1) + '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp.append(float(current_number))  

	standard_acc.append(temp)   



standard_time = list()

for i in range(2):
	temp = list()
	with open('standard/ft'+ str(i+1) + '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp.append(float(current_number))  

	standard_time.append(temp) 


total_standard_time = list()

for i in range(2):	
	total_standard_time.append(np.cumsum(standard_time[i]))




for i in range(6):
	plt.plot(total_selective_time[i], selective_acc[i], label = 'Coefficients Set ' + str(i+1))



# for i in range(2):
# 	plt.plot(total_random_time[i], random_acc[i], label = 'random ' + str(i+1))


# for i in range(2):
# 	plt.plot(total_standard_time[i], standard_acc[i], label = 'standard ' + str(i+1))



# plt.plot(total_selective_time[5], label = 'intelligent sampling' )
# plt.plot(total_random_time[0], label = 'random sampling')
# plt.plot(total_standard_time[0], label = 'standard FL')

#plt.xlim(0, 840)


plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Runtime(s)')
plt.title('Coefficients Comparison for 100 Epochs')
plt.show()





















