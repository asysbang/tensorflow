#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

import scipy.misc

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
  
#注意不要太多空白，否则缩放之后，文字信息不够大
#注意要黑底白字，否则识别不行的  
#需要二值化，切黑色是0-1的浮点数
#识别：5  0  2  3  6
#未识别：1->6   4->3    7->2    8->3   9->1
img = cv2.imread('test_data/9_2.png')  # 手写数字图像所在位置

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换图像为单通道(灰度图)

ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) #二值化  必须要有ret 两个参数来接受返回值

re_img = cv2.resize(thresh_img, (28,28))

cv2.imshow('image', re_img) #图片很小，在左上角。。。。注意仔细找

re_img = re_img / 255.0

scipy.misc.imsave('xx.jpg', re_img)#保存图片

print(re_img.shape)

cv2.waitKey(0)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.figure()
plt.imshow(x_train[7])
plt.colorbar()
plt.grid(False)
plt.show()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=6)

model.evaluate(x_test,  y_test, verbose=2)

# 将整个模型保存为HDF5文件
model.save('my_model.h5')




# 重新创建完全相同的模型，包括其权重和优化程序
new_model = keras.models.load_model('my_model.h5')

# 显示网络结构
new_model.summary()

loss, acc = new_model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

print(x_test[0].shape)

predictions = model.predict(re_img.reshape(1,28,28), batch_size=1)

print(predictions[0])

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#用柱状图显示预测结果，第一个参数表示“索引”，在评估时根据对错来显示颜色，这里做预测，所以这个参数没有意义
plot_value_array(4, predictions[0], y_test)
_ = plt.xticks(range(10), class_names, rotation=0)
plt.show()



