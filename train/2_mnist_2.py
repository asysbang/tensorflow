#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

import tensorflow as tf

from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]

test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0 #注意这里不是28,28  而是28*28

test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
    )

    return model

model = create_model()

model.summary()

checkoutpoint_path = 'checkout/cp.ckpt'

checkoutpoint_dir = os.path.dirname(checkoutpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkoutpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[cp_callback])

!ls {checkpoint_dir}