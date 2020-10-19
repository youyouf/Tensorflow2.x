# -*- coding: utf-8 -*-
# @File : config.py
# @Author: riky
# @Time : 2020/10/19
# @Software: PyCharm
# @Brief: 配置文件

import numpy as np

#分类数目
num_classes = 2
#输入网络的图片大小
input_shape = (224,224,3)
#输入网络的高度、宽度
height, width = input_shape[0], input_shape[1]

#是否加载预训练权重(注意要开启skip_mismatch和by_name)(绝对路径)
pretain_weights_path = r"C:\Users\xx\Desktop\VGG\model\vgg16_weights_tf_dim_ordering_tf_kernels.h5"
#从后面开始计算冻结网络
frozeing_layers_num = -5

#图片存放的位置（该文件夹应该直接包含图片而不是其子文件夹包含图片）（绝对路径）
img_dir = r"C:\Users\xx\Desktop\VGG\train"
#该文件是图片存放的位置与标签的关系对应表（绝对路径）
img_label_TXT_dir = r"C:\Users\xx\Desktop\VGG\config\train.txt"

#训练信息
log_dir = "./logs"
batch_size = 3
epochs = 2
learning_rate = (1e-4)*batch_size

#训练集合与验证集的划分
train_rate = 0.9
val_rate = 1 - train_rate

