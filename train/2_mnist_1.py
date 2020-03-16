#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 中文注释需要添加 UTF-8


# MNIST 的训练识别例子
# MNIST ： Mixed National Institute of Standards and Technology
# 手写数字样本，了解整个运行过程，有整体的印象和概念


import tensorflow as tf 
from tensorflow import keras

#获取训练数据
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#将样本从整数转换为浮点数                                                   ？？？？为什么浮点数
x_train = x_train / 255.0
x_test = x_test / 255.0

#将模型各层叠加， 搭建tf.keras.Sequential模型，
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

#为模型选择优化器和损失函数
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#训练
model.fit(x_train, y_train, epochs=5)

#验证
model.evaluate(x_test, y_test, verbose=2)


