# -*- coding: utf-8 -*-
# @File : train.py
# @Author: riky
# @Time : 2020/10/18
# @Software: PyCharm
# @Brief: 训练文件
import tensorflow as tf
from dataloader import DataLoader
import config as cfg
from net import Net
import numpy as np
import os
import shutil

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

def main():
    log_dir = r"./logs"
    # 删除之前tensorboard创建时遗留下来的文件
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    if not os.path.exists(log_dir):  # 判断logs的文件夹是否存在，如果不存在则创建logs
        os.mkdir(log_dir)

    # 创建训练集Datset对象
    Dataset = DataLoader(root=cfg.images_path)
    images, labels= Dataset.load_data(mode='train')
    db_train = Dataset.make_datasets(images,labels)

    # 创建验证集Datset对象
    images, labels= Dataset.load_data(mode='val')
    db_val = Dataset.make_datasets(images,labels)

    # 创建测试集Datset对象
    images, labels= Dataset.load_data(mode='test')
    db_test = Dataset.make_datasets(images,labels)

    print("num_class",Dataset.getNumClass())
    print("类型与标签对应表：",Dataset.getName2Label())
    model = Net(num_class=int(Dataset.getNumClass()))
    # ----------------------------------------学习率以及提前停止设置--------------------------------------------------------
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=15)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=3, min_lr=1e-7)
    modelsave_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + '/Model-Weight-ep{epoch:03d}-loss{loss:.3f}-accuracy{accuracy:.3f}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}.ckpt',
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True,
        period=1
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=cfg.learn_rating),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', "Recall", "AUC"])
    history = model.fit(db_train, validation_data=db_val, validation_freq=1, epochs=cfg.epochs,
                        callbacks=[reduce_lr])#, early_stopping, modelsave_checkpoint

   # ----------------------------------------输出结果图片-------------------------------------
    history = history.history
    print(history.keys())
    print(history['val_accuracy'])
    print(history['accuracy'])
    test_acc = model.evaluate(db_test)
    print("test acc:", test_acc)

    plt.figure()
    returns = history['val_accuracy']
    plt.plot(np.arange(len(returns)), returns, label='验证准确率')
    plt.plot(np.arange(len(returns)), returns, 's')
    returns = history['accuracy']
    plt.plot(np.arange(len(returns)), returns, label='训练准确率')
    plt.plot(np.arange(len(returns)), returns, 's')

    plt.plot([len(returns) - 1], [test_acc[-1]], 'D', label='测试准确率')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.savefig('scratch.svg')

    plt.show()
if __name__ == "__main__":
    main()
    print("Code is running at the end!")