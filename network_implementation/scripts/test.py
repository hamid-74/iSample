import tensorflow as tf
import random

from copy import deepcopy

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from numpy import linalg as LA
import numpy as np
from copy import copy
from random import sample

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0



print(random.uniform(0,1))