import tensorflow as tf

from copy import deepcopy

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from numpy import linalg as LA
import numpy as np
from copy import copy
import random


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

N = 16
E = 100

dataset_size = 50000

def model_norm_calculator(model_old, model):
  norm0 = LA.norm(model_old.layers[0].get_weights()[0] - model.layers[0].get_weights()[0])
  norm2 = LA.norm(model_old.layers[2].get_weights()[0] - model.layers[2].get_weights()[0])
  norm4 = LA.norm(model_old.layers[4].get_weights()[0] - model.layers[4].get_weights()[0])
  norm6 = LA.norm(model_old.layers[6].get_weights()[0] - model.layers[6].get_weights()[0])
  norm7 = LA.norm(model_old.layers[7].get_weights()[0] - model.layers[7].get_weights()[0])
  return norm0 + norm2 + norm4 + norm6 + norm7




  

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()



# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


model = []

for i in range(N):
  model.append(models.Sequential())
  model[i].add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model[i].add(layers.MaxPooling2D((2, 2)))
  model[i].add(layers.Conv2D(64, (3, 3), activation='relu'))
  model[i].add(layers.MaxPooling2D((2, 2)))
  model[i].add(layers.Conv2D(64, (3, 3), activation='relu'))
  model[i].add(layers.Flatten())
  model[i].add(layers.Dense(64, activation='relu'))
  model[i].add(layers.Dense(10))





accuracy_history = np.zeros((N,E))
global_loss = np.zeros(E)
global_accuracy = np.zeros(E)


for i in range(E):
  old_model = []
  norms = []
  for j in range(N):
    old_model.append(tf.keras.models.clone_model(model[j]))
    old_model[j].set_weights(model[j].get_weights())

    model[j].compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    a = int(j*(dataset_size/N))
    b = int((j+1)*(dataset_size/N))

    history = (model[j].fit(train_images[a:b], train_labels[a:b], epochs=1, validation_data=(test_images, test_labels)))
    
    accuracy_history[j][i] = deepcopy(history.history['accuracy'][0])

    norms.append((model_norm_calculator(model[j], old_model[j])))

    

  sorted_norms = sorted( [(x,j) for (j,x) in enumerate(norms)], reverse=True ) 

 

  selected_models = []
  seeds = random.sample(range(1, N), int(N/4))

  
  for j in range (int(N/4)):

    selected_models.append(model[seeds[j]])

  weights = [mdl.get_weights() for mdl in selected_models]
  new_weights = list()

  
  for weights_list_tuple in zip(*weights): 
    new_weights.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
    )
  
  
  global_model = tf.keras.models.clone_model(model[0])
  global_model.set_weights(new_weights)

  
  global_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  global_loss[i], global_accuracy[i] = global_model.evaluate(test_images,  test_labels, verbose=2)
  print("global model accuracy: " + str(global_accuracy[i]) + "Epoch: " + str(i))

  
  for j in range(N):
    local_models = [model[j], global_model]
    weights = [mdl.get_weights() for mdl in local_models]

    new_weights = list()

  
    for weights_list_tuple in zip(*weights): 
      new_weights.append(
          np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
      )
    model[j].set_weights(new_weights)
  
  
with open('random_selective_results' + '.txt', 'w') as filehandle: 
  for listitem in global_accuracy:
    filehandle.write('%s\n' % listitem)




for i in range(N):
  plt.plot(accuracy_history[i], label='accuracy' + str(i))
plt.plot(global_accuracy, label='global_accuracy')
#plt.plot(history[0].history['accuracy'], label='accuracy')
#plt.plot(history[0].history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(loc='lower right')
plt.show()

#test_loss, test_acc = model[0].evaluate(test_images,  test_labels, verbose=2)
#print(test_acc)



