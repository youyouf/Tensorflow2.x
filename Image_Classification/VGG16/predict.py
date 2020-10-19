# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: riky
# @Time : 2020/10/19
# @Software: PyCharm
# @Brief: 测试文件

import numpy as np
import cv2
import tensorflow as tf

def test(model,img_path):
    #读取图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    img = img / 255.
    img = np.expand_dims(img,axis=0)

    #进行预测
    result = model.predict(img)
    print("model output is :",result)
    if np.argmax(result) == 0:
        print("The result is cat.")
    else:
        print("The result is dog.")

if __name__ == "__main__":
    #-------------------------------加载只带有权重的模型-------------------------------
    # model = VGG16()
    # try:
    #     model.load_weights("./logs/last1.h5")
    #     print("load weights is ok.")
    # except:
    #     print("load weights occur error.")

    #-------------------------------加载不只是带有权重的模型-------------------------------
    try:
        model = tf.keras.models.load_model("./logs/Model-Weight-ep002-loss0.146-val_loss0.163.h5")
        print("load model is successful.")
    except:
        print("load model occurs error.")

    #-------------------------------测试-------------------------------
    test(model = model, img_path = r"./train/cat.5.jpg")

    print("Code is running at the end.")