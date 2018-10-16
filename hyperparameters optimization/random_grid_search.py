#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: MingZ
# @Date created: 05 April 2018
# @Date last modified: 24 May 2018
# Python Version: 3.6
# Description: Construct a classification model on the MNIST data set in the form of a deep neural network
#              Perform and compare two or more methods of hyperparameter optimization on this model, and comment on the comparison.
# hyperparameter: learning_rate, epochs, batch_size, activation function, hidden layers and neurons, weight initialization, dropout for regularization
# hyperparameter tuning:grid search, randomized search

# import packages
import numpy as np
import pickle as pk
import gzip
import time

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
# from sklearn.metrics import confusion_matrix
# from keras.datasets import mnist
from keras.backend import clear_session
from keras.utils.np_utils import to_categorical
from keras.utils import print_summary
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
# optimizers: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, TFOptimizer
from keras.optimizers import *
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------Convolutional Neural Network--------------------------
# model building
def build_model(filters,kernel_size,dropout_rate,layer_size,num_layer):
    clear_session() # clear backend model

    # model construction
    model = Sequential()

    model.add(Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu', input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu'))
    model.add(MaxPool2D(pool_size = 2))
    model.add(Dropout(dropout_rate))

    for n in range(num_layer):
        model.add(Conv2D(filters = (2**(n+1))*filters, kernel_size = kernel_size, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters = (2**(n+1))*filters, kernel_size = kernel_size, activation = 'relu'))
        model.add(MaxPool2D(pool_size = 2))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(layer_size, activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation = "softmax"))

    # compile
    model.compile(
    optimizer = Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 0.0),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])
    return model

if __name__ == "__main__": 
    # ---------------------------Data Preparation---------------------------------
    # load the dataset
    file = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = pk.load(file)
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

    start = time.time()
    # -----------------------Grid Search------------------------------------
    dropout_rate = [0.25, 0.5]
    layer_size = [256]
    kernel_size = [5, 3]
    filters = [32]
    epochs = [1]
    batch_size = [100, 80, 60]
    num_layer = [1]

    # pass parameters to GridSearchCV
    param_grid = dict(
        num_layer = num_layer,
        dropout_rate = dropout_rate,
        layer_size = layer_size,
        kernel_size = kernel_size,
        filters = filters,
        epochs = epochs,
        batch_size = batch_size)
    model = KerasClassifier(build_fn = build_model, verbose = 2)
    models = RandomizedSearchCV(model, param_grid, cv = StratifiedKFold(n_splits = 2), n_jobs = -1)

    # fit
    result = models.fit(x_train,y_train)
    print('best model:')
    print(result.best_params_)
    print(time.time() - start)

    # -----------------------predict & score-------------------------------
    print(grid.score(x_test, y_test))
