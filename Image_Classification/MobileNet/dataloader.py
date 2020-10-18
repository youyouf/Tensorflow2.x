# -*- coding: utf-8 -*-
# @File : DataLoader.py
# @Author: Riky
# @Time : 2020/10/18
# @Software: PyCharm
# @Brief: 数据加载文件

import os, glob
import random, csv
import config as cfg
import tensorflow as tf
import numpy as np

class DataLoader():
    def __init__(self,root = cfg.images_path, mode = "train"):
        self.root = root
        self.mode = mode

        # 创建数字编码表
        self.name2label = {}  # "sq...":0    #文件夹与标签的对应关系
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 给每个类别编码一个数字
            self.name2label[name] = len(self.name2label.keys())

    def getName2Label(self):
        return self.name2label

    def getNumClass(self):
        return len(self.name2label)

    def load_csv(self,filename = 'images.csv'):
        if not os.path.exists(os.path.join(self.root, filename)):
            # 如果csv文件不存在，则创建
            images = []
            for name in self.name2label.keys():  # 遍历所有子目录，获得所有的图片
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            print(len(images), images)
            random.shuffle(images)  # 随机打散顺序
            # 创建csv文件，并存储图片路径及其label信息
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('written into csv file:', filename)

        # 此时已经有csv文件，直接读取
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
                # 返回图片路径list和标签list
        assert len(images) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(images), len(labels))
        return images, labels

    def load_data(self,mode ='train'):
        self.mode = mode
        # 读取Label信息
        images, labels = self.load_csv()
        print("Total images num:",len(images))

        if self.mode == 'train':  # 60%
            images = images[:int(cfg.train_rate * len(images))]
            labels = labels[:int(cfg.train_rate * len(labels))]
            print("----------Train images num:",len(images))
        elif self.mode == 'val':  # 20% = 60%->80%
            images = images[int(cfg.train_rate * len(images)):int((cfg.train_rate + cfg.valid_rate) * len(images))]
            labels = labels[int(cfg.train_rate * len(labels)):int((cfg.train_rate + cfg.valid_rate) * len(labels))]
            print("----------Val images num:",len(images))
        else:  # 20% = 80%->100%
            images = images[int((cfg.train_rate + cfg.valid_rate) * len(images)):]
            labels = labels[int((cfg.train_rate + cfg.valid_rate) * len(labels)):]
            print("----------Test images num:",len(images))

        return images, labels

    def normalize(self,x, mean=cfg.img_mean, std=cfg.img_std):
        # 标准化
        x = (x - mean) / std
        return x

    def denormalize(self,x, mean=cfg.img_mean, std=cfg.img_std):
        # 标准化的逆过程
        x = x * std + mean
        return x

    def preprocess(self,x,y):

        width,height = cfg.input_shape
        # x: 图片的路径，y：图片的数字编码
        x = tf.io.read_file(x)
        x = tf.image.decode_jpeg(x, channels=3)  # RGBA
        x = tf.image.resize(x, [width, height])

        if self.mode == 'train': #测试集合的时候进行数据变换
            if np.random.rand() < cfg.img_enhanceRate:
                if np.random.rand() > 0.5:
                    x = tf.image.random_flip_left_right(x)
                if np.random.rand() > 0.5:
                    x = tf.image.random_flip_up_down(x)
                if np.random.rand() > 0.5:
                    x = tf.image.random_brightness(x, max_delta=10)
                if np.random.rand() > 0.5:
                    x = tf.image.random_contrast(x, lower=0.7, upper=1.3)
                if np.random.rand() > 0.5:
                    x = tf.image.random_saturation(x, lower=0.618, upper=1.382)
                if np.random.rand() > 0.5:
                    x = tf.image.random_crop(x, [width, height, 3])
        # x: [0,255]=> -1~1
        x = tf.cast(x, dtype=tf.float32) / 255.
        x = self.normalize(x)

        y = tf.convert_to_tensor(y)
        y = tf.one_hot(y, depth=int(self.getNumClass()))

        return x, y
    def make_datasets(self,images, labels):
        db = tf.data.Dataset.from_tensor_slices((images, labels))
        db = db.shuffle(cfg.shuffle_size).map(self.preprocess).batch(cfg.batch_size)
        return db

def main():
    #加载pokemon数据集，指定加载训练
    images,labels,name2label = DataLoader(root="./pokeman",mode="train").load_data()
    print('images:', len(images), images)
    print('labels:', len(labels), labels)
    print('name2label:', name2label)
    print("Code is running at the end!")



if __name__ == "__main__":
    main()

