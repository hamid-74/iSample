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
parameters_lock = threading.Lock()

no_of_clients = 4
no_of_epochs = 7


def receive_parameters(c, parameters, addr):

	data = b""

	while True:
		packet = c.recv(4096)
		if not packet: break
		data += packet

	time_stamp = pickle.loads(data[0:12])
	client_parameters = pickle.loads(data[12:])


	temp = client_parameters/(time.time() - time_stamp)

	parameters_lock.acquire()
	parameters.append((addr[0], temp))	
	parameters_lock.release()

	print("transfer time:", time.time() - time_stamp)


	c.close()



def request_models(c, parameters, addr):
	
	msg = pickle.dumps(0)

	for i in range(int(no_of_clients/2)):
		if parameters[i][0] == addr[0]:
			msg = pickle.dumps(1)

	c.send(msg)

	c.close()


def receive_weights(c, weights):
	
	data = b""

	while True:
		packet = c.recv(4096)
		if not packet: break
		data += packet

	time_stamp = pickle.loads(data[0:12])
	client_weights = pickle.loads(data[12:])
	weights_lock.acquire()
	weights.append(client_weights)	
	weights_lock.release()
	print("transfer time:", time.time() - time_stamp)


	
	# connection closed
	c.close()
	

def send_global_model(c, averaged_weights):

	msg = pickle.dumps(averaged_weights)

	c.send(msg)

	c.close()
	







(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#train_images, test_images = train_images / 255.0, test_images / 255.0

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



port = 12345
parameters_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
parameters_socket.bind((socket.gethostname(), port))
print("parameters_socket binded to port", port)
parameters_socket.listen(10)
print("parameters_socket is listening")

port = 12346
requests_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
requests_socket.bind((socket.gethostname(), port))
print("requests_socket binded to port", port)
requests_socket.listen(10)
print("requests_socket is listening")

port = 12347
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), port))
print("receiving weights socket binded to port", port)
s.listen(10)
print("eceiving weights socket is listening")


port = 12348
s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s2.bind((socket.gethostname(), port))
print("sending global model socket binded to port", port)
s2.listen(10)
print("sending global model socket is listening")



epoch_time = list()
start_time = time.time()

accuracy_history = list()


for e in range(no_of_epochs):
	
	#receiving parameters
	parameters = list()
	threads = []
	for i in range(no_of_clients):
		c, addr = parameters_socket.accept()

		t = Thread(target=receive_parameters, args=(c, parameters, addr, ))
		t.start()
		threads.append(t)


	for t in threads:
	    t.join()

	
	parameters.sort(key = lambda x: x[1], reverse = True)

	print(parameters)
	print("receiving parameters done!")

	#sending requests
	threads = []
	for i in range(no_of_clients):
		c, addr = requests_socket.accept()

		t = Thread(target=request_models, args=(c, parameters, addr, ))
		t.start()
		threads.append(t)


	for t in threads:
	    t.join()

	print("sending requests done!")

	#receiveing models
	weights = list()
	threads = []

	for i in range(int(no_of_clients/2)):

		c, addr = s.accept()

		t = Thread(target=receive_weights, args=(c, weights, ))
		t.start()
		threads.append(t)

	for t in threads:
	    t.join()

	print("receiving models done!")

	#averaging
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

	#sending global model
	threads = []

	for i in range(int(no_of_clients)):
		c, addr = s2.accept()
		t = Thread(target=send_global_model, args=(c, averaged_weights, ))
		t.start()
		threads.append(t)


	for t in threads:
	    t.join()

	print("sending global model done!")

	epoch_time.append(time.time() - start_time)
	accuracy_history.append(accuracy)

s.close()
s2.close()
parameters_socket.close()
requests_socket.close()	

print("epoch times:", epoch_time)
print("accuracy:", accuracy_history)



