#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: MingZ
# @Date created: 28 mars 2018
# @Date last modified: 28 mars 2018
# Python Version: 2.7
# Description: tensorflow & softmax regression $ cnn

import numpy as np
import tensorflow as tf
import os
import csv

# import tensorflow.examples.tutorials.mnist.input_data as input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print "images:",len(mnist.train.images[0]),mnist.train.images
# print "labels:",len(mnist.train.labels)

def load_train_set():
    l = []
    i = 0
    with open("train.csv") as file:
        lines = csv.reader(file)
        for line in lines:
            if i == 0:
                i += 1
                continue
            l.append(map(eval, line))
    l = np.array(l)
    labels_list = l[:,0]
    features = l[:,1:]
    labels = []
    dict_label = dict.fromkeys(range(10),0)
    for label in labels_list:
        dict_label[label] = 1
        labels.append(dict_label.values())
        dict_label[label] = 0
    return normalizing(features),np.array(labels)

def normalizing(array):
    m, n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j] != 0:
                array[i,j] = 1
    return array

index_in_epoch = 0
num_examples = x_train.shape[0]
def next_batch(batch_size):
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > num_examples:
      # Finished epoch
      epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(num_examples)
      numpy.random.shuffle(perm)
      images = images[perm]
      labels = labels[perm]
      # Start next epoch
      start = 0
      index_in_epoch = batch_size
      assert batch_size <= num_examples
    end = index_in_epoch
    return images[start:end], labels[start:end]

if __name__ == "__main__":
    train_set = load_train_set()
    x_train = train_set[0]
    y_train = train_set[1]









