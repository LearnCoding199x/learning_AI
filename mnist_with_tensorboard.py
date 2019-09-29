import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras as keras
from time import time
from keras.datasets import mnist
import numpy as np
(x_train , y_train),(x_test,y_test) = mnist.load_data()

x_train = np.reshape(x_train,(x_train.shape[0],28,28,1))
x_test = np.reshape(x_test,(x_test.shape[0],28,28,1))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test /  255.0

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))


model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[28,28,1]))
model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='tanh'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dense(10,activation = 'softmax'))
model.compile(optimizer='adam',metrics=['acc'],loss='sparse_categorical_crossentropy')
model.fit(x_train,y_train,epochs=5,callbacks=[tensorboard])

test_loss , test_acc = model.evaluate(x_test,y_test)

print('ACCURACY : ',test_acc)

