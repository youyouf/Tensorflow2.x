# -*- coding: utf-8 -*-
# @File : mobilenet_v2.py
# @Author: riky
# @Time : 2020/10/18
# @Software: PyCharm
# @Brief: mobilenet v2网络构建

import tensorflow as tf
from tensorflow.keras import layers

def MobileNet(input_shape = (224,224,3),alpha = 1,dropout = 0.5,num_classes = 1000):
    input = layers.Input(input_shape)

    #第一层
    x = layers.Conv2D(filters=int(32* alpha),kernel_size=(3,3),strides=(2,2),padding="same")(input)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    #第二层
    x = layers.DepthwiseConv2D(kernel_size=(3,3),strides=(1,1),padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(64* alpha),kernel_size=(1,1),strides=(1,1),padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    #第二层
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(128 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    #第三层
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(128 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    #第四层
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(256 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    #第五层
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(256 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    #第六层
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(512 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    #第七层
    for _ in range(5):
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Activation("relu")(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.Conv2D(filters=int(512 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Activation("relu")(x)
        x = layers.ReLU(max_value=6.0)(x)

    #第八层
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(1024 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters=int(1024 * alpha), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    x = layers.ReLU(max_value=6.0)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Flatten()(x)

    # x = layers.Dense(1024,activation= None,kernel_initializer= tf.keras.initializers.he_normal(),# W的初始化
    #                                         bias_initializer= tf.keras.initializers.zeros())(x)     # B的初始化
    #
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(dropout)(x)
    out = layers.Dense(units=num_classes,activation="softmax")(x)

    model = tf.keras.Model(input,out,name="MobileNet")
    return model


def main():
    print(tf.__version__)
    model = MobileNet()
    model.summary()



if __name__ == '__main__':
    main()



