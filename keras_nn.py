#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: MingZ
# @Date created:
# @Date last modified:
# Python Version: 2.7
# Description:

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras import optimizers
import numpy as np
import csv
import pandas as pd

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

train_set = load_train_set()
x_train = train_set[0]
y_train = train_set[1]
x_test = load_test_set()

model = Sequential()
# 第一层
model.add(Dense(input_dim=28*28,
                output_dim=500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 第二层
model.add(Dense(output_dim=500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 输出层
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

# 编译学习过程
# 优化器optimizer
# 损失函数loss:模型试图最小化的目标函数mse：均方误差函数
# 指标列表metrics:分类问题一般将该列表设置为metrics=['accuracy']
# sgd = optimizers.SGD(lr=0.1)
adm = optimizers.Adam()
model.compile(loss='categorical_crossentropy',
                optimizer=adm,
                metrics=['accuracy'])

# 训练
model.fit(x_train, y_train,batch_size=100,nb_epoch=150)
result_list = model.predict(x_test)
# np.savetxt("result.txt",result)

data = pd.read_csv("sample_submission.csv")
for i in data.index:
    result = list(result_list[i])
    data.loc[i,"Label"] = result.index(max(result))
result_list.to_csv("submission.csv",index=None)
