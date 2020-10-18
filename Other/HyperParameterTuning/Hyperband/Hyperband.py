# -*- coding: utf-8 -*-
# @File : Hyperband.py
# @Author: Runist
# @Time : 2020/4/22 15:24
# @Software: PyCharm
# @Brief: Hyperband 超带搜索

import kerastuner as kt
import tensorflow as tf
import tensorflow_datasets as tfds

def build_model(hp):
    inputs = tf.keras.Input(shape=(32,32,3))
    x = inputs
    for i in range(hp.Int("conv_blocks",3,5,default=3)):
        filters = hp.Int("filters_" + str(i),32,256,step=32)
        for _ in range(2):
            x = tf.keras.layers.Convolution2D(filters,kernel_size=(3,3), padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = tf.keras.layers.AvgPool2D()(x)
    x  = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(hp.Int('hidden_size', 30, 100, step=10, default=50),activation='relu')(x)
    x = tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    return model

print("Load data ....")
data = tfds.load('cifar10')
print("load data is ok!")

train_ds, test_ds = data['train'], data['test']

def standardize_record(record):
    return tf.cast(record['image'], tf.float32) / 255., record['label']

train_ds = train_ds.map(standardize_record).cache().batch(200).shuffle(10000)
test_ds = test_ds.map(standardize_record).cache().batch(200)

#依旧需要对该类的一些参数进行调参（还有一些参数使用，例如：仅对设置的参数进行调参、数据保存路径等）
tuner = kt.Hyperband(build_model,   #设置超参数的模型，函数返回的是模型
                     objective = "val_accuracy", #观测对象为验证集的准确度
                     max_epochs=2,    #最大的epochs
                     hyperband_iterations=2)

print("Tuner is running.")
tuner.search(train_ds,                  #训练集
             validation_data=test_ds,   #测试集
             epochs=3,                  #训练次数
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)]) #Callbacks

print("Tuner search is ok!")
best_model = tuner.get_best_models(1)[0] #获取最好的模型
best_hyperparameters = tuner.get_best_hyperparameters(1)[0] #获取最好模型的参数

print("------------------------------------------------------------------------------------------")
print("best model:",best_model)
print("Summary:",best_model.summary)
print("------------------------------------------------------------------------------------------")
print("best_hyperparameters:",best_hyperparameters)
print("------------------------------------------------------------------------------------------")
print("tuner.results_summary():",tuner.results_summary())

#保存最好的模型
print("save best model...")
best_model.save("best_model.h5")

#加载该模型
print("load model..")
model = tf.keras.models.load_model("best_model.h5")

#测试该模型
test_acc = model.evaluate(test_ds)
print("test acc:", test_acc)
print("Code is running at the end.")

