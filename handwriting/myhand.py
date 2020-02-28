#!/usr/bin/python
# -*- coding: UTF-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import scipy.misc

print(tf.__version__)



mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[1]
print(img.shape)
print(type(img))#图片类型
print(img.dtype)#数据类型
print(img.shape)#图片大小
scipy.misc.imsave('im.jpg', img)#保存图片

plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
plt.grid(False)
plt.show()


img = scipy.misc.imread('test.jpeg',mode='RGB')
 
print(img.shape)
print(type(img))#图片类型
print(img.dtype)#数据类型
print(img.shape)#图片大小
 
plt.figure(1)
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()

#img_tensor = tf.image.decode_image(img)
#img_tensor = tf.image.decode_jpeg(img_tensor, channels=3)
#img_tensor = tf.image.resize(img_tensor, [28, 28])
#img_tensor /= 255.0  # normalize to [0,1] range
#
#
#scipy.misc.imsave('process.jpg', img_tensor)#保存图片