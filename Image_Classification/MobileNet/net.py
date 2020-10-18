# -*- coding: utf-8 -*-
# @File : Net.py
# @Author:Riky
# @Time : 2020/10/18
# @Software: PyCharm
# @Brief: 网络结构
import tensorflow as tf
from tensorflow import keras
import config as cfg

def Net(num_class,dropoutRate = 0.3):
    MobileNet = tf.keras.applications.MobileNet(input_shape=(224,224,3),
                                                alpha=1.0,
                                                depth_multiplier=1,
                                                include_top=False,
                                                weights='imagenet'
                                          )
    width,height = cfg.input_shape
    inputs = keras.Input(shape=(width,height,3))
    x = MobileNet(inputs,training = True)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropoutRate)(x)
    outputs = keras.layers.Dense(int(num_class),activation= "softmax")(x)
    model = keras.Model(inputs,outputs)
    model.summary()
    return model

if __name__ == "__main__":
    model = Net(5)
    print("Code is running at the end.")