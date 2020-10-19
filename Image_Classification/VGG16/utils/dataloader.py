# -*- coding: utf-8 -*-
# @File : dataloader.py
# @Author: riky
# @Time : 2020/10/19
# @Software: PyCharm
# @Brief: 数据加载
import cv2
import numpy as np
import tensorflow as tf
import config.config as cfg

class dataloader:
    def __init__(self,
                 txtdir = cfg.img_label_TXT_dir,
                 mode ="train",
                 train_rate = cfg.train_rate):
        self.txtdir = txtdir
        self.mode = mode
        self.train_rate = train_rate

    def data_split(self):
        # 打开数据集的txt
        with open(self.txtdir, "r") as f:
            lines = f.readlines()
        # 打乱行，这个txt主要用于帮助读取数据来训练
        # 打乱的数据更有利于训练
        np.random.seed(12345)
        np.random.shuffle(lines)
        np.random.seed()

        # 90%用于训练，10%用于估计
        num_train = int(len(lines) * self.train_rate)
        num_val = len(lines) - num_train

        lines_train = lines[:num_train]
        lines_val = lines[num_train:]
        print("Example of lines: lines[0]:",lines[0])

        return lines_train,lines_val

    def data_loader(self,lines, batch_size):
        # 获取总长度
        n = len(lines)
        i = 0

        np.random.seed()
        np.random.shuffle(lines) #先对lines进行打乱
        while True:
            X_train = []
            Y_train = []
            # 获取一个batch_size大小的数据
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(lines)

                img_dir = lines[i].split(';')[0]
                # 从文件中读取图像
                img = cv2.imread(img_dir) #可能有错
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=(cfg.height, cfg.width), interpolation=cv2.INTER_AREA)
                img = img / 255.
                X_train.append(img)

                Y_train.append(lines[i].split(';')[1]) #获取标签信息
                # 读完一个周期后重新开始
                i = (i + 1) % n

            # 处理图像
            #X_train = tf.convert_to_tensor(X_train)
            X_train = tf.reshape(X_train, (-1, cfg.input_shape[0],cfg.input_shape[2], cfg.input_shape[3]))
            Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=cfg.num_classes)  # 进行归一化
            yield (X_train, Y_train)

#将图片修剪成中心的正方形
def cut_image(path):
    #读取图片，rgp
    img = cv2.imread(path)
    #将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])#选取图片最短的边长
    yy = int((img.shape[0] - short_edge) / 2) #取出左右两边需要删减的图片
    xx = int((img.shape[1] - short_edge) / 2) #去除上下两边需要删减的图片

    crop_img = img[yy:yy + short_edge, xx:xx+short_edge] #对图片进行裁剪，获取ROI区域

    return crop_img, crop_img.shape

#将图片的四周补灰，并且进行缩放
def image_resize(image,dsize):
    pass



