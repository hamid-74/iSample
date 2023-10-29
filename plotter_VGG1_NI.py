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



vgg1_standard_NI_times = []
with open('standard_NI/vgg1_epoch_times.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        vgg1_standard_NI_times.append(float(current_number))

vgg1_standard_NI_times_print = Cumulative(vgg1_standard_NI_times)

vgg1_standard_NI_accuracy = []
with open('standard_NI/vgg1_standard_results.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        vgg1_standard_NI_accuracy.append(float(current_number))



vgg1_random_NI_times = []
with open('random_NI/vgg1_random_epoch_times.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        vgg1_random_NI_times.append(float(current_number))

vgg1_random_NI_times_print = Cumulative(vgg1_random_NI_times)

vgg1_random_NI_accuracy = []
with open('random_NI/vgg1_random_results.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        vgg1_random_NI_accuracy.append(float(current_number))



vgg1_selective_P_NI_times_print = list()
vgg1_selective_P_NI_accuracy = list()

for i in range (2):	

	temp_time = list()
	with open('P:NI/epoch_times_vgg1_P_NI_' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_time.append(float(current_number))

	temp_time_print = Cumulative(temp_time)
	vgg1_selective_P_NI_times_print.append(temp_time_print)


	temp_accuracy = list()
	with open('P:NI/selective_results_vgg1_P_NI_' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_accuracy.append(float(current_number))

	vgg1_selective_P_NI_accuracy.append(temp_accuracy)


vgg1_selective_NP_NI_times_print = list()
vgg1_selective_NP_NI_accuracy = list()

for i in range (3):	

	temp_time = list()
	with open('NP:NI/epoch_times_vgg1_NP_NI' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_time.append(float(current_number))

	temp_time_print = Cumulative(temp_time)
	vgg1_selective_NP_NI_times_print.append(temp_time_print)


	temp_accuracy = list()
	with open('NP:NI/selective_results_vgg1_NP_NI' + str(i)+ '.txt', 'r') as filehandle:
	    for line in filehandle:
	        
	        current_number = line[:-1]

	        temp_accuracy.append(float(current_number))

	vgg1_selective_NP_NI_accuracy.append(temp_accuracy)


for i in range(3):
	print(vgg1_selective_NP_NI_times_print[i][99])
for i in range(2):
	print(vgg1_selective_P_NI_times_print[i][99])
print(vgg1_standard_NI_times_print[99])
print(vgg1_random_NI_times_print[99])

# plt.plot(vgg1_selective_P_NI_times_print[0], vgg1_selective_P_NI_accuracy[0], label = 'selective_vgg1_cfg1_P')
# plt.plot(vgg1_selective_P_NI_times_print[1], vgg1_selective_P_NI_accuracy[1], label = 'Sampling_VGG1_cfg2_P')
# plt.plot(vgg1_selective_P_NI_times_print[2], vgg1_selective_P_NI_accuracy[2], label = 'selective_vgg1_cfg3_P')

plt.plot(vgg1_selective_NP_NI_times_print[0], vgg1_selective_NP_NI_accuracy[0], label = 'Sampling_VGG1_cfg1_NP')
plt.plot(vgg1_selective_NP_NI_times_print[1], vgg1_selective_NP_NI_accuracy[1], label = 'Sampling_VGG1_cfg2_NP')
plt.plot(vgg1_selective_NP_NI_times_print[2], vgg1_selective_NP_NI_accuracy[2], label = 'Sampling_VGG1_cfg3_NP')

# plt.plot(vgg1_standard_NI_times_print, vgg1_standard_NI_accuracy,label = 'Standard_VGG1')
# plt.plot(vgg1_random_NI_times_print, vgg1_random_NI_accuracy,label = 'Random_VGG1')


plt.xlim([0,3800])
plt.title('VGG1 with Non-IID Training Data')

plt.xlabel('Time(s)')
plt.ylabel('Accuracy')

plt.legend()
#plt.show()