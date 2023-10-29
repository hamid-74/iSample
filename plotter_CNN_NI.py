import numpy as np
import random
from matplotlib import pyplot as plt
import statistics
import math

def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]



cnn_standard_NI_times = []
with open('standard_NI/cnn_epoch_times0.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        cnn_standard_NI_times.append(float(current_number))

cnn_standard_NI_times_print = Cumulative(cnn_standard_NI_times)

cnn_standard_NI_accuracy = []
with open('standard_NI/cnn_standard_results0.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        cnn_standard_NI_accuracy.append(float(current_number))



cnn_random_NI_times = []
with open('random_NI/cnn_random_epoch_times.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        cnn_random_NI_times.append(float(current_number))

cnn_random_NI_times_print = Cumulative(cnn_random_NI_times)

cnn_random_NI_accuracy = []
with open('random_NI/cnn_random_results.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        cnn_random_NI_accuracy.append(float(current_number))



cnn_selective_P_NI_times_print = list()
cnn_selective_P_NI_accuracy = list()

for i in range (3):	

	temp_time = list()
	with open('P:NI/epoch_times_cnn_P_NI' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_time.append(float(current_number))

	temp_time_print = Cumulative(temp_time)
	cnn_selective_P_NI_times_print.append(temp_time_print)


	temp_accuracy = list()
	with open('P:NI/selective_results_cnn_P_NI' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_accuracy.append(float(current_number))

	cnn_selective_P_NI_accuracy.append(temp_accuracy)


cnn_selective_NP_NI_times_print = list()
cnn_selective_NP_NI_accuracy = list()

for i in range (3):	

	temp_time = list()
	with open('NP:NI/epoch_times_cnn_NP_NI' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_time.append(float(current_number))

	temp_time_print = Cumulative(temp_time)
	cnn_selective_NP_NI_times_print.append(temp_time_print)


	temp_accuracy = list()
	with open('NP:NI/selective_results_cnn_NP_NI' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_accuracy.append(float(current_number))

	cnn_selective_NP_NI_accuracy.append(temp_accuracy)

for i in range(3):
	print(cnn_selective_NP_NI_times_print[i][99])
for i in range(3):
	print(cnn_selective_P_NI_times_print[i][99])
print(cnn_standard_NI_times_print[99])
print(cnn_random_NI_times_print[99])


# plt.plot(cnn_selective_P_NI_times_print[0], cnn_selective_P_NI_accuracy[0], label = 'selective_cnn_cfg1_P')
# plt.plot(cnn_selective_P_NI_times_print[1], cnn_selective_P_NI_accuracy[1], label = 'selective_cnn_cfg2_P')
# plt.plot(cnn_selective_P_NI_times_print[2], cnn_selective_P_NI_accuracy[2], label = 'Sampling_CNN_cfg3_P')

plt.plot(cnn_selective_NP_NI_times_print[0], cnn_selective_NP_NI_accuracy[0], label = 'Sampling_CNN_cfg1_NP')
plt.plot(cnn_selective_NP_NI_times_print[1], cnn_selective_NP_NI_accuracy[1], label = 'Sampling_CNN_cfg2_NP')
plt.plot(cnn_selective_NP_NI_times_print[2], cnn_selective_NP_NI_accuracy[2], label = 'Sampling_CNN_cfg3_NP')

# plt.plot(cnn_standard_NI_times_print, cnn_standard_NI_accuracy,label = 'Standard_CNN')
# plt.plot(cnn_random_NI_times_print, cnn_random_NI_accuracy,label = 'Random_CNN')

plt.xlim([0,800])
plt.title('CNN with Non-IID Training Data')

plt.xlabel('Time(s)')
plt.ylabel('Accuracy')

plt.legend()
#plt.show()