#Image Classification of handwritten digits
#usage of the MNIST database for training and test data

import tensorflow as tf

#External Libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#above is are the 4 arrays for testing and training data

#preprocess the data

train_images = train_images/255.0
test_images = test_images/255.0

model =  tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(10)

    #2Fully Connected Layers after input using ReLu activation f(x)=max(0,x)
])

model.compile(optimizer='adam', #optimizer (uses loss function data)
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) #metrics measured during training

#fit model

model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest Accuracy:', test_acc) 

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest Accuracy:', test_acc)























