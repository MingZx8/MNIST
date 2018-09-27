#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: MingZ
# @Date created:
# @Date last modified:
# Python Version: 2.7
# Description:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.utils import print_summary
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

# 读取数据
train = pd.read_csv("train.csv")
x_train = train.drop(columns=["label"])
y_train = train['label']
del train
test = pd.read_csv("test.csv")

# 归一化
x_train = x_train/255.0
test = test/255.0

# reshape: -1表示样本数量，28*28的尺寸，1是黑白图片
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# sns.distplot(y_train)

# print y_train.value_counts()
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
# 用于应用到以categorical_crossentropy为目标函数的模型中
y_train = to_categorical(y_train, num_classes=10)

# Split the train and the validation set for the fitting一部分用来训练一部分用来评估
# 只适用于分布平均的数据集，否则要加参数stratify = True
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = 3, activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = 3, activation ='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = 3, activation ='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = RMSprop() , loss = "categorical_crossentropy", metrics=["accuracy"])
# 回调学习率，monitor：被监测的是accuracy，patience：2个epoch后性能不能提升就开始优化学习率
# verbose:日志进度，factor：每次减少的学习率lr *= factor， min_lr：学习率的下限
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                            factor=0.5, min_lr=0.00001)
epoch = 20
batch_size = 86

# data augmentation增加样本: 图片生成器
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset，使输入数据集去中心化（均值为0）
        samplewise_center=False,  # set each sample mean to 0，使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

# 自定义训练模型
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epoch, validation_data = (x_val,y_val),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // 60, 
                              callbacks=[learning_rate_reduction])

#
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# plt.imshow(x_train[0][:,:,0])

plt.show()

# predict results
result_list = model.predict(test)

# select the indix with the maximum probability
result_list = np.argmax(result_list,axis = 1)

result_list = pd.Series(result_list,name="Label")

# # 构造模型
# model = Sequential()

# # 第一层卷积层cov2d
# model.add(Conv2D(input_shape=(28,28,1),filters=32, kernel_size=3, activation='relu'))
# # model.add(Dropout(0.25))

# # 第二层卷积层，池化层
# model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
# model.add(MaxPool2D(pool_size=4))

# # 输出层
# model.add(Flatten())
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10,activation='softmax'))

# # 编译，训练模型
# model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=["accuracy"])
# # learning_rate_reduction = ReduceLROnPlateau(monitor="val_acc",patience=3,verbose=1,factor=0.5,min_lr=0.00001)

# print_summary(model)
# model.fit(x_train,y_train,batch_size=100,epochs=5)

# # test
# result_list = model.predict(test)

#
data = pd.DataFrame(["ImageID","Label"])
row = 1
for result in result_list:
    result = list(result)
    data.insert(row,row,[row,result.index(max(result))])
    row += 1
data.T.to_csv("submission.csv",index=None,header=None)
