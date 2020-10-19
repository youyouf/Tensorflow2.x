# -*- coding: utf-8 -*-
# @File : VGG16.py
# @Author: riky
# @Time : 2020/10/19
# @Software: PyCharm
# @Brief: 网络构建

import tensorflow as tf
import config.config as cfg

def VGG16():
    image_input = tf.keras.Input(shape = cfg.input_shape)
    #第一个卷积部分
    x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(image_input)
    x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)

    # 第二个卷积部分
    x = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)

    # 第三个卷积部分
    x = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)

    # 第四个卷积部分
    x = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)

    # 第五个卷积部分
    x = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)

    # 分类部分
    x = tf.keras.layers.Conv2D(256,(7,7),activation = 'relu',padding = 'valid', name = 'block6_conv4')(x)
    x = tf.keras.layers.Flatten(name = 'flatten')(x)
    x = tf.keras.layers.Dense(256,activation = 'relu',name = 'fullc1')(x)
    x = tf.keras.layers.Dense(256,activation = 'relu',name = 'fullc2')(x)
    x = tf.keras.layers.Dense(cfg.num_classes,activation = 'softmax',name = 'fullc3')(x)
    model = tf.keras.Model(image_input,x,name = 'vgg16')

    model.summary()

    # 注意要开启skip_mismatch和by_name
    try:
        model.load_weights(cfg.pretain_weights_path,by_name=True,skip_mismatch=True)
        print("The model of weights is existing,and is loaded.")
    except:
        print("The weights is not existing.")
    try:
        for i in range(0,len(model.layers) + cfg.frozeing_layers_num):
            model.layers[i].trainable = False
        print("Frozeing layers is successful")
    except:
        print("Frozeing layers occurs errors")

    return model


if __name__ == "__main__":
    Net = VGG16()
    print("Code is running at the end.")