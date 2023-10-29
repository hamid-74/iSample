import tensorflow as tf
import random

from copy import deepcopy

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from numpy import linalg as LA
import numpy as np
from copy import copy
from random import sample

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

clients = models.Sequential()
clients.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
clients.add(layers.MaxPooling2D((2, 2)))
clients.add(layers.Conv2D(64, (3, 3), activation='relu'))
clients.add(layers.MaxPooling2D((2, 2)))
clients.add(layers.Conv2D(64, (3, 3), activation='relu'))
clients.add(layers.Flatten())
clients.add(layers.Dense(64, activation='relu'))
clients.add(layers.Dense(10))
clients.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])










clients.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


global_loss, global_accuracy = clients.evaluate(test_images,  test_labels, verbose=2)
print("global model accuracy: " + str(global_accuracy))




