#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: MingZ
# @Date created: 28 mars 2018
# @Date last modified: 29 mars 2018
# Python Version: 2.7
# Description: tensorflow & softmax regression $ cnn

import numpy as np
import tensorflow as tf
import os
import csv
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 读取数据，返回xtrainset和ytrainset
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

def load_test_set():
    l = []
    i = 0
    with open("test.csv") as file:
        lines = csv.reader(file)
        for line in lines:
            if i == 0:
                i = 1
                continue
            l.append(map(eval, line))
    l = np.array(l)
    return normalizing(l)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def next_batch(images,labels,batch_size,index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size
    end = index_in_epoch
    return images[start:end], labels[start:end], index_in_epoch

def hiden_layer(x, keep_prob):
    #第一层卷积，卷积在每个5x5的patch中算出32个特征
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    #把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    #(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #密集连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #减少拟合
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #输出
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


if __name__ == "__main__":
    train_set = load_train_set()
    x_train = train_set[0]
    y_train = train_set[1]
    x_test = load_test_set()

    index_in_epoch = 0
    num_examples = x_train.shape[0]

    #计算图
    sess = tf.InteractiveSession()

    #实际输入输出站位
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    # #softmax regression weights and bias
    # W = tf.Variable(tf.zeros([784,10]))
    # b = tf.Variable(tf.zeros([10]))

    # #初始化变量，只有初始过的变量才能在sess里使用
    # sess.run(tf.initialize_all_variables())

    # #交叉熵
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # y = tf.nn.softmax(tf.matmul(x,W) + b)
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    keep_prob = tf.placeholder("float")
    y_conv = hiden_layer(x,keep_prob)

    #train
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    for i in range(201):
        batch = next_batch(x_train, y_train, 50, index_in_epoch)
        index_in_epoch = batch[2]
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # print "test accuracy %g"%accuracy.eval(feed_dict={
    #     x: x_test, y_: y_train, keep_prob: 1.0})
    x_test_image = tf.reshape(x_test, [-1,28,28,1])
    # result = sess.run([y], feed_dict={x: x_test_image})
    # print result
    y_test = sess.run(hiden_layer(x_test_image,keep_prob),feed_dict={x: x_test_image})













