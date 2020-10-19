# -*- coding: utf-8 -*-
# @File : train.py
# @Author: riky
# @Time : 2020/10/19
# @Software: PyCharm
# @Brief: 训练文件
import os
import shutil
import tensorflow as tf
from model.VGG16 import VGG16
from utils.dataloader import dataloader
import config.config as cfg

def main():
    #模型保存的位置
    #删除遗留下来的文件
    if os.path.exists(cfg.log_dir):
        shutil.rmtree(cfg.log_dir)

    if not os.path.exists(cfg.log_dir):  # 判断logs的文件夹是否存在，如果不存在则创建logs
        os.mkdir(cfg.log_dir)

    #打开数据集的txt
    dataHandler = dataloader()
    train_db,val_db = dataHandler.data_split()

    #建立VGG16模型
    model = VGG16()

    # 保存的方式，3世代保存一次
    checkpoint_period1 = tf.keras.callbacks.ModelCheckpoint(
                                    cfg.log_dir +"/" +'Model-Weight-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}.h5',
                                    monitor='accuracy', #一般使用val_accuracy
                                    save_weights_only=False,
                                    save_best_only=True,
                                    period=3
                                )

    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='accuracy', #训练的精度
                            factor=0.5,
                            patience=3,
                            verbose=1
                        )

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1
                        )

    # 交叉熵
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=cfg.learning_rate),
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy', "Recall", "AUC"]
                  )


    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_db), len(val_db), cfg.batch_size))

    # 开始训练
    model.fit_generator(dataHandler.data_loader(train_db, cfg.batch_size),
                        steps_per_epoch=max(1, int(len(train_db) // cfg.batch_size)),
                        validation_data=dataHandler.data_loader(val_db, cfg.batch_size),
                        validation_steps=max(1, int(len(val_db) // cfg.batch_size)),
                        epochs=cfg.epochs,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr,early_stopping]
                        )

    model.save_weights(cfg.log_dir + '/last1.h5')


    #-------------------------------------Test--------------------------------------------



if __name__ == '__main__':
    main()
    print("Code is running at the end.")