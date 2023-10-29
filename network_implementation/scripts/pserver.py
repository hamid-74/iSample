import socket
import pickle
import time

import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from threading import Thread

from _thread import *
import threading

weights_lock = threading.Lock()



no_of_epochs = 10
no_of_clients = 2


# thread function

def threaded_receive(c, weights, addr, parameters):
	
	
	data = b""

	while True:
		packet = c.recv(4096)
		if not packet: break
		data += packet

	time_stamp = pickle.loads(data[0:12])
	client_weights = pickle.loads(data[12:])

	temp = 1/(time.time() - time_stamp)
	weights_lock.acquire()
	weights.append(client_weights)
	parameters.append((addr[0], temp))		
	weights_lock.release()
	print("transfer time:", time.time() - time_stamp)


	
	# connection closed
	c.close()
	

def threaded_send(c, averaged_weights, i):

	
	msg = pickle.dumps(averaged_weights)

	c.send(msg)

	c.close()
	



def sort_parameters(parameters):
	return dict(sorted(parameters.items(), key=lambda item: item[1]))

def create_connection():
	port = 12345
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((socket.gethostname(), port))
	print("socket binded to port", port)


	s.listen(10)
	print("socket1 is listening")


	port = 12346
	s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s2.bind((socket.gethostname(), port))
	print("socket2 binded to port", port)
	s2.listen(10)
	print("socket2 is listening")

	return s, s2

def close_connection(s, s2):
	s.shutdown(socket.SHUT_RDWR)
	s2.shutdown(socket.SHUT_RDWR)
	s.close()
	s2.close()



(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




epoch_time = list()
start_time = time.time()

accuracy_history = list()

s, s2 = create_connection()

for e in range(no_of_epochs):
	
	print("we are at epoch:", e)
	

	weights = list()
	clients = list()
	addresses = list()
	counter = 0
	threads = []
	parameters = list()
	for i in range(no_of_clients):
		

		c, addr = s.accept()
		clients.append(c)
		addresses.append(addr)
		
		print('Connected to :', addr[0], ':', addr[1])


		t = Thread(target=threaded_receive, args=(c, weights, addr, parameters, ))
		t.start()
		threads.append(t)
		
	


	for t in threads:
	    t.join()


	print(parameters)
	parameters.sort(key = lambda x: x[1], reverse = True)

	print(parameters)
	#s.close()


	best_models = list()
	for i in range(len(weights)):
		best_models.append(tf.keras.models.clone_model(model))
		best_models[i].set_weights(weights[i])


	weights_list = [mdl.get_weights() for mdl in best_models]
	averaged_weights = list()

	  
	for weights_list_tuple in zip(*weights_list): 
		averaged_weights.append(
	        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
	    )



	model.set_weights(averaged_weights)
	loss, accuracy = model.evaluate(test_images,  test_labels, verbose=2)


	threads = []

	for i in range(no_of_clients):
		c, addr = s2.accept()
		t = Thread(target=threaded_send, args=(c, averaged_weights, i, ))
		t.start()
		threads.append(t)


	for t in threads:
	    t.join()


	

	epoch_time.append(time.time() - start_time)
	accuracy_history.append(accuracy)


close_connection(s, s2)	


print("epoch times:", epoch_time)
print("accuracy:", accuracy_history)


