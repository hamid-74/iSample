import tensorflow as tf
import random

from copy import deepcopy

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from numpy import linalg as LA
import numpy as np
from copy import copy
from random import sample

import json


from matplotlib import pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'






no_of_clients = 64
epochs = 100
dataset_size = 50000
training_dataset_size = 2000

size_factor = 1060130/122570

latencies = [0.006, 0.0247, 0.043, 0.050]






bandwidths = list()
with open('../bandwidths.txt', 'r') as filehandle:
    for line in filehandle:
        
        current_number = line[:-1]

        bandwidths.append(float(current_number))



print(bandwidths)



      
def averaging_time (n):
  return n * 0.0002 * size_factor

training_time = 5.33

receive_parameters_time = 0.04
sending_request_time = 0.099
sending_global_models_time = 0.405 * size_factor #different for different models


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0


def model_norm_calculator(model_old, model):
  norm0 = LA.norm(model_old.layers[0].get_weights()[0] - model.layers[0].get_weights()[0])
  norm1 = LA.norm(model_old.layers[1].get_weights()[0] - model.layers[1].get_weights()[0])
  norm4 = LA.norm(model_old.layers[4].get_weights()[0] - model.layers[4].get_weights()[0])
  norm5 = LA.norm(model_old.layers[5].get_weights()[0] - model.layers[5].get_weights()[0])
  
  return norm0 + norm1 + norm4 + norm5


def createRandomSortedList(num, start = 1, end = 50000):
    arr = []
    tmp = random.randint(start, end)
      
    for x in range(num):
          
        while tmp in arr:
            tmp = random.randint(start, end)
              
        arr.append(tmp)
          
    arr.sort()
      
    return arr






def create_indexes(clients_training_images, clients_training_labels, no_of_clients):
  
  for i in range(no_of_clients): 
    index = list()
    #index = createRandomSortedList(training_dataset_size, 1, 50000)

    with open('../indexes/index' + str(i+1) + '.txt', 'r') as filehandle:
      for line in filehandle:
        current_number = line[:-1]
        index.append(int(current_number))
  
    temp_images = list()
    temp_labels = list()

    for j in range(training_dataset_size):
      temp_images.append(train_images[index[j]-1])
      temp_labels.append(train_labels[index[j]-1])

    clients_training_images.append(np.array(temp_images))
    clients_training_labels.append(np.array(temp_labels))


clients_training_images = list()
clients_training_labels = list()


create_indexes(clients_training_images, clients_training_labels, no_of_clients)
  






coefficients = list()

coefficients.append([1, 0.5, 0.1])
coefficients.append([0.5, 1, 0.1])
coefficients.append([0.75, 0.75, 0.1])


for k in range(2,3):

  a = coefficients[k][0]
  b = coefficients[k][1]
  c = coefficients[k][2]

  clients = list()

  for i in range(no_of_clients):
    clients.append(models.Sequential())
    clients[i].add(layers.Conv2D(32,(3,3), activation = 'relu', padding = 'same', input_shape = (32,32,3)))
    clients[i].add(layers.Conv2D(32,(3,3), activation = 'relu', padding = 'same'))
    clients[i].add(layers.MaxPooling2D((2,2)))
  
    clients[i].add(layers.Flatten())
    clients[i].add(layers.Dense(128, activation = 'relu'))
    clients[i].add(layers.Dense(10, activation = 'softmax'))

      
    clients[i].compile(optimizer = 'adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])



  clients_accuracy = np.zeros((no_of_clients,epochs))
  clients_loss = np.zeros((no_of_clients,epochs))
  global_loss = np.zeros(epochs)
  global_accuracy = np.zeros(epochs)
  epoch_time = np.zeros(epochs)


  for i in range(epochs):

    old_models = list()
    client_grades = list()
    max_comm_time = 0

    bandwidths_for_plot = list()
    accuracies_for_plot = list()

    best_accuracies_for_plot = list()
    best_bandwiths_for_plot = list()
    other_accuracies_for_plot = list()
    other_bandwidths_for_plot = list()

    max_accuracy = 0
    max_norm = 0

    for j in range(no_of_clients):
      old_models.append(tf.keras.models.clone_model(clients[j]))
      old_models[j].set_weights(clients[j].get_weights())

      temp_images = clients_training_images[j]
      temp_labels = clients_training_labels[j]

      

      clients[j].fit(temp_images, temp_labels, epochs=1, validation_data=(test_images, test_labels))
      
      clients_loss[j][i], clients_accuracy[j][i] = clients[j].evaluate(test_images,  test_labels, verbose=2)

      if clients_accuracy[j][i] > max_accuracy:
        max_accuracy = clients_accuracy[j][i]

      if model_norm_calculator(clients[j], old_models[j]) > max_norm:
        max_norm = model_norm_calculator(clients[j], old_models[j])
      

      bandwidths_for_plot.append(5/bandwidths[j]) #mbit/s
      accuracies_for_plot.append(clients_accuracy[j][i])
      

    for j in range(no_of_clients):
      client_grade = a * (clients_accuracy[j][i]/max_accuracy) - b * (bandwidths[j]/max(bandwidths)) + c * (model_norm_calculator(clients[j], old_models[j])/max_norm)
      client_grades.append(client_grade)

    sorted_grades = sorted( [(x,j) for (j,x) in enumerate(client_grades)], reverse=True ) 



    best_models = []
    for j in range (int(no_of_clients/4)):
      best_models.append(clients[sorted_grades[j][1]])

      #change here for different models
      if (bandwidths[sorted_grades[j][1]] * size_factor + latencies[(sorted_grades[j][1])%4]) > max_comm_time:
        max_comm_time = bandwidths[sorted_grades[j][1]]  * size_factor + latencies[(sorted_grades[j][1])%4]

    weights = [mdl.get_weights() for mdl in best_models]
    new_weights = list()

    
    for weights_list_tuple in zip(*weights): 
      new_weights.append(
          np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
      )
    
    for j in range(int(no_of_clients/4)):
      best_accuracies_for_plot.append(clients_accuracy[sorted_grades[j][1]][i])
      best_bandwiths_for_plot.append(5/bandwidths[sorted_grades[j][1]])

    for j in range(int(no_of_clients/4), no_of_clients):
      other_accuracies_for_plot.append(clients_accuracy[sorted_grades[j][1]][i])
      other_bandwidths_for_plot.append(5/bandwidths[sorted_grades[j][1]])

    if i%(int(epochs/4)) == 0:

      plt.plot(other_accuracies_for_plot, other_bandwidths_for_plot, 'bs', best_accuracies_for_plot, best_bandwiths_for_plot, 'g^')
      plt.xlabel("Accuracy")
      plt.ylabel("bandwidth(mbit/s)")
      plt.savefig('vgg1_epoch' + str(i+1) + '_cfg' + str(k+1) + 'NP_I.png', dpi = 600)
      plt.close()
    
    global_model = tf.keras.models.clone_model(clients[0])  
    global_model.set_weights(new_weights)

    
    global_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    global_loss[i], global_accuracy[i] = global_model.evaluate(test_images,  test_labels, verbose=2)
    print("global model accuracy: " + str(global_accuracy[i]) + "Epoch: " + str(i))

    
    for j in range(no_of_clients):
      local_models = [clients[j], global_model]
      weights = [mdl.get_weights() for mdl in local_models]

      new_weights = list()

    
      for weights_list_tuple in zip(*weights): 
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )
      clients[j].set_weights(global_model.get_weights())


    temp_epoch_time = training_time + receive_parameters_time + sending_request_time + max_comm_time + averaging_time(int(no_of_clients/4)) + sending_global_models_time
    epoch_time[i] = temp_epoch_time
    print("temp_epoch_time", temp_epoch_time)


  with open('selective_results_vgg1_NP_I' + str(k) + '.txt', 'w') as filehandle: 
    for listitem in global_accuracy:
      filehandle.write('%s\n' % listitem)

  with open('epoch_times_vgg1_NP_I' + str(k) + '.txt', 'w') as filehandle: 
    for listitem in epoch_time:
      filehandle.write('%s\n' % listitem)


  


  

  

print('bandwidths', bandwidths)
print('coefficients', coefficients)














