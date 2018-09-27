#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: MingZ
# @Date created: 05 April 2018
# @Date last modified: 24 May 2018
# Python Version: 3.6
# Description: Construct a classification model on the MNIST data set in the form of a deep neural network
#              Perform and compare two or more methods of hyperparameter optimization on this model, and comment on the comparison.
# cnn layers: (2 convolutional + 1 pooling + dropout) * 2 + flatten + dense + dropout + dense
# others: learning rate reduction, enlarge the data sets size, divide original training sets into training and validating sets 

# import packages
import numpy as np
import pickle as pk
import gzip
import pandas as pd

from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
# from keras.utils import print_summary
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
# optimizers: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, TFOptimizer
from keras.optimizers import RMSprop, Adadelta
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------Data Preparation---------------------------------
# load the dataset
file = gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set = pk.load(file,encoding='latin1')
file.close()

x_train = train_set[0]
# print(x_train.shape) #(50000, 784)
y_train = train_set[1]
del train_set # free some space

x_test = test_set[0]
# print(x_test.shape) #(10000, 784)
y_test = test_set[1]
del test_set

x_val = valid_set[0]
y_val = valid_set[1]
del valid_set

# reshape (sample population=-1, 28*28, chanel=1(black&white))
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
x_val = x_val.reshape(-1,28,28,1)

# map labels to on hot vectors in order to be applied into 
# model with categorical_crossentropy objective funtion
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test,num_classes=10)
y_val = to_categorical(y_val,num_classes=10)

# data augmentation (avoid overfitting)
data_generator = ImageDataGenerator(
    featurewise_center = False,  # set input mean to 0 over the dataset，使输入数据集去中心化（均值为0）
    samplewise_center = False,  # set each sample mean to 0，使输入数据的每个样本均值为0
    featurewise_std_normalization = False,  # divide inputs by std of the dataset
    samplewise_std_normalization = False,  # divide each input by its std
    zca_whitening = False,  # apply ZCA whitening
    rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range = 0.10,  # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.10,  # randomly shift images vertically (fraction of total height)
    horizontal_flip = False,  # randomly flip images
    vertical_flip = False)  # randomly flip images
print (data_generator.fit(x_train))

# # ---------------------------Convolutional Neural Network--------------------------
# # model construction
# model = Sequential()
# model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
#                  activation ='relu', input_shape = (28,28,1)))
# model.add(BatchNormalization())
# model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
#                  activation ='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.2))


# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
#                  activation ='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
#                  activation ='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.2))

# model.add(Flatten())
# model.add(Dense(512, activation = "relu"))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation = "softmax"))

# # parameter
# learning_rate = 1
# epochs = 40
# batch_size = 82
# step_size = 100

# # optimizer
# optimizer = Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)

# # compile
# model.compile(
#     optimizer = optimizer,
#     loss = 'categorical_crossentropy',
#     metrics = ['accuracy'])
# learning_rate_reduction = ReduceLROnPlateau(
#     monitor = 'val_acc', 
#     patience = 3, 
#     verbose = 1, 
#     factor = 0.5, 
#     min_lr = 0.00001)

# # fit
# history = model.fit_generator(
#     data_generator.flow(x_train,y_train, batch_size=batch_size),
#     epochs = epochs, 
#     validation_data = (x_val,y_val),
#     verbose = 1, 
#     steps_per_epoch = x_train.shape[0] // step_size, 
#     callbacks = [learning_rate_reduction])

# model.save('my_model.h5')
