# -*- coding: utf-8 -*-
# @File : model.py
# @Author: Runist
# @Time : 2020/3/30 13:44
# @Software: PyCharm
# @Brief: YOLO3模型实现

from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
import config.config as cfg


def conv_bn_leaky(inputs, num_filter, kernel_size, strides=(1, 1), bn=True):
    """
    卷积 + 批归一化 + leaky激活，因为大量用到这样的结构，所以这样写
    :param inputs: 输入
    :param num_filter: 卷积个数
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param bn: 是否使用批归一化
    :return: x
    """
    if "conv2d" not in conv_bn_leaky.__dict__:
        if cfg.pretrain:
            conv_bn_leaky.conv2d = 53
        else:
            conv_bn_leaky.conv2d = 1

    if "bn" not in conv_bn_leaky.__dict__:
        if cfg.pretrain:
            conv_bn_leaky.bn = 53
        else:
            conv_bn_leaky.bn = 1

    if "leaky" not in conv_bn_leaky.__dict__:
        if cfg.pretrain:
            conv_bn_leaky.leaky = 53
        else:
            conv_bn_leaky.leaky = 1

    if strides == (1, 1) or strides == 1:
        padding = 'same'
        x = inputs
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)  # top left half-padding
        padding = 'valid'

    x = Conv2D(num_filter, kernel_size=kernel_size,
               strides=strides, padding=padding,                    # 这里的参数是只l2求和之后所乘上的系数
               use_bias=not bn, kernel_regularizer=l2(0.0005),      # 只有添加正则化参数，才能调用model.losses方法
               name="conv2d_{}".format(conv_bn_leaky.conv2d)
               )(x)    # 加上序号，名字一致，方便debug
    conv_bn_leaky.conv2d += 1

    if bn:
        x = BatchNormalization(name="batch_normalization_{}".format(conv_bn_leaky.bn))(x)
        # alpha是x < 0时，变量系数
        x = LeakyReLU(alpha=0.1, name="leaky_re_lu_{}".format(conv_bn_leaky.leaky))(x)

        conv_bn_leaky.bn += 1
        conv_bn_leaky.leaky += 1

    return x


def darknet_block(inputs, num_filters):
    """
    darknet53残差基本单元
    （conv + bn + leaky） x 2 然后相加
    :param inputs: 上一层输入
    :param num_filters: 残差块中的两个conv 对应的卷积核个数，传入列表
    :return:
    """
    if "add" not in darknet_block.__dict__:
        darknet_block.add = 1

    filter_1, filter_2 = num_filters
    x = conv_bn_leaky(inputs, filter_1, kernel_size=1, strides=(1, 1))
    x = conv_bn_leaky(x, filter_2, kernel_size=3, strides=(1, 1))

    output = Add(name="add_{}".format(darknet_block.add))([inputs, x])
    darknet_block.add += 1

    return output


def resiual_block(inputs, filters, num_blocks):
    """
    残差块
    ZeroPadding + conv + nums_filters 次 darknet_block
    :param inputs: 上一层输出
    :param filters: conv的卷积核个数，每次残差块是不一样的
    :param num_blocks: 有几个这样的残差块
    :return: 卷积结果
    """
    x = conv_bn_leaky(inputs, filters, 3, strides=(2, 2))

    for i in range(num_blocks):
        # 传入残差块 卷积核个数，第一个conv的个数是上面的conv的一半
        # 第二个是和上面的conv的个数一样
        x = darknet_block(x, [filters // 2, filters])

    return x


def darknet53(inputs):
    # input_shape [b, 416, 416, 3]
    x = conv_bn_leaky(inputs, 32, 3)  # [b, 416, 416, 32]

    # -----------------------------------------
    x = resiual_block(x, 64, 1)  # [b, 208, 208, 64]

    # -----------------------------------------x2
    # 注释的原操作，没有可以用resiual_block替代
    # x = conv_bn_leaky(x, 128, 3, strides=(2, 2))
    # x = darknet_block(x, [64, 128])
    # x = darknet_block(x, [64, 128])
    x = resiual_block(x, 128, 2)  # [b, 104, 104, 128]

    # -----------------------------------------x8
    x = resiual_block(x, 256, 8)  # [b, 52, 52, 256]
    feat52x52 = x

    # -----------------------------------------x8
    x = resiual_block(x, 512, 8)  # [b, 26, 26, 512]
    feat26x26 = x

    # -----------------------------------------x4
    x = resiual_block(x, 1024, 4)  # [b, 13, 13, 1024]
    feat13x13 = x

    # 下面是完整的darknet53，但是yolov3只是用他的卷积层，不用全连接层和激活层
    # x = keras.layers.AvgPool2D()(x)
    # x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(1000)(x)
    # predict = keras.layers.Softmax()(x)
    # import tensorflow as tf
    # model = tf.keras.Model(inputs=inputs, outputs=[feat13x13, feat26x26, feat52x52])
    # model.summary()
    # tf.keras.utils.plot_model(model,"./model.png")
    # return model
    return feat52x52, feat26x26, feat13x13

if __name__ == "__main__":
    pass