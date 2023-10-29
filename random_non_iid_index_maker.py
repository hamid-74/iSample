

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


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0




specific_indexes = list()

for i in range(10):
  temp = list()
  for j in range (dataset_size):
    if (train_labels[j] == i):
      temp.append(j)

  specific_indexes.append(temp)






for i in range(no_of_clients): 
  index = list()
  

  with open('indexes/index' + str(i+1) + '.txt', 'r') as filehandle:
    for line in filehandle:
      current_number = line[:-1]
      index.append(int(current_number))


  with open('non_iid_indexes/index'+ str(i+1) + '.txt', 'w') as filehandle: 
    for listitem in index[0:(int(training_dataset_size/5))] + random.sample(specific_indexes[i % 10], int(training_dataset_size * 0.8)):
      filehandle.write('%s\n' % listitem)




