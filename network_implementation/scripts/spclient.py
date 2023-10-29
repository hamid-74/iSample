import socket
import pickle
import time


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

no_of_epochs = 3

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

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


for e in range(no_of_epochs):

	model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))
	print("before averaging:")
	loss, accuracy = model.evaluate(test_images,  test_labels, verbose=2)


	#sending parameters
	port = 12345

	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	host = socket.gethostname()
	s.connect((host,port))

	time_stamp = pickle.dumps(time.time())
	
	
	parameters = pickle.dumps(accuracy)

	msg = time_stamp + parameters

	s.send(msg)
	s.close()


	#receiving request
	port = 12346
	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	s.connect((host,port))

	received_message = b""

	while True:
		packet = s.recv(4096)
		if not packet: break
		received_message += packet

	request = pickle.loads(received_message)

	s.close()

	print("request:", request)

	#sending weights
	if request:
		port = 12347

		s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		host = socket.gethostname()
		s.connect((host,port))

		time_stamp = pickle.dumps(time.time())
		weights = pickle.dumps(model.get_weights())

		msg = time_stamp + weights

		s.send(msg)
		s.close()
	

	port = 12348
	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	s.connect((host,port))
	received_message = b""

	while True:
		packet = s.recv(4096)
		if not packet: break
		received_message += packet

	received_weights = pickle.loads(received_message)

	s.close()



	model.set_weights(received_weights)

	print("after averaging:", model.evaluate(test_images,  test_labels, verbose=2))

