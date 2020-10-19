# -*- coding: utf-8 -*-
# @File : model.py
# @Author: riky
# @Time : 2020/3/30 13:44
# @Software: PyCharm
# @Brief: YOLOv2模型实现
import tensorflow as tf
from tensorflow import keras
import config.config as cfg

class Yolov2:
    def __init__(self,
                 input_shape = cfg.input_shape,
                 num_classes = cfg.num_classes,
                 num_bbox = cfg.num_bbox
                 ):
        self.height, self.width = input_shape
        self.num_classes = num_classes
        self.num_bbox = num_bbox
    def conv2d(self,
               input,
               filters,
               kernel_size,
               padding_size=0,
               strides=1,
               batch_normalize=True,
               activation = tf.nn.leaky_relu,
               use_bias=False,
               name='conv2d'
               ):
        if padding_size > 0:
            input=tf.pad(input, paddings=[[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]])
        #进行卷积操作
        input=tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding="valid",
                                     activation=None,
                                     use_bias=use_bias,
                                     name=name
                                     )(input)
        # BN应该在卷积层conv和激活函数activation之间,
        # 后面有BN层的conv就不用偏置bias，并激活函数activation在后
        # 如果需要标准化则进行标准化
        if batch_normalize:
            input = tf.keras.layers.BatchNormalization()(input)
        if activation:
            input = activation(features=input,alpha=0.1)#leaky_relu(input)
        return input

    def yolo_boady(self):
        inputs = keras.layers.Input(shape=(self.height,self.width,3), dtype='float32')
        # 416,416,3 -> 416,416,32
        x = self.conv2d(inputs, filters=32, kernel_size=3, padding_size=1,name='conv1')

        # 416,416,32 -> 208,208,32
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool1')(x)
        # 208,208,32 -> 208,208,64
        x = self.conv2d(x, filters=64, kernel_size=3, padding_size=1, name='conv2')
        # 208,208,64 -> 104,104,64
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool2')(x)

        # 104,104,64 -> 104,104,128
        x = self.conv2d(x, filters=128, kernel_size=3, padding_size=1, name='conv3_1')
        x = self.conv2d(x, filters=64, kernel_size=1, padding_size=0, name='conv3_2')
        x = self.conv2d(x, filters=128, kernel_size=3, padding_size=1, name='conv3_3')
        # 104,104,128 -> 52,52,128
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool3')(x)

        x = self.conv2d(x, filters=256, kernel_size=3, padding_size=1, name='conv4_1')
        x = self.conv2d(x, filters=128, kernel_size=1, padding_size=0, name='conv4_2')
        x = self.conv2d(x, filters=256, kernel_size=3, padding_size=1, name='conv4_3')
        # 52,52,128 -> 26,26,256
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='pool4')(x)

        # 26,26,256-> 26,26,512
        x = self.conv2d(x, filters=512, kernel_size=3, padding_size=1, name='conv5_1')
        x = self.conv2d(x, filters=256, kernel_size=1, padding_size=0, name='conv5_2')
        x = self.conv2d(x, filters=512, kernel_size=3, padding_size=1, name='conv5_3')
        x = self.conv2d(x, filters=256, kernel_size=1, padding_size=0, name='conv5_4')
        x = self.conv2d(x, filters=512, kernel_size=3, padding_size=1, name='conv5_5')

        # 这一层特征图，要进行后面passthrough，保留一层特征层
        shortcut = x
        # 26,26,512-> 13,13,512
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),name='pool5')(x)

        # 13,13,512-> 13,13,1024
        x = self.conv2d(x, filters=1024, kernel_size=3, padding_size=1, name='conv6_1')
        x = self.conv2d(x, filters=512, kernel_size=1, padding_size=0, name='conv6_2')
        x = self.conv2d(x, filters=1024, kernel_size=3, padding_size=1, name='conv6_3')
        x = self.conv2d(x, filters=512, kernel_size=1, padding_size=0, name='conv6_4')
        x = self.conv2d(x, filters=1024, kernel_size=3, padding_size=1, name='conv6_5')

        # 下面这部分主要是training for detection
        x = self.conv2d(x, filters=1024, kernel_size=3, padding_size=1, name='conv7_1')
        # 13,13,1024-> 13,13,1024
        x = self.conv2d(x, filters=1024, kernel_size=3, padding_size=1, name='conv7_2')

        # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
        # 得到了26*26*512 -> 26*26*64 -> 13*13*256的特征图
        shortcut = self.conv2d(shortcut, filters=64, kernel_size=1, padding_size=0, name='conv_shortcut')
        shortcut = tf.reshape(shortcut,shape=[-1,13,13,256])

        # 连接之后，变成13*13*（1024+256）
        x = keras.layers.concatenate(inputs=[shortcut, x], axis=-1)
        # channel整合到一起，concatenated with the original features，passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上，
        x = self.conv2d(x, filters=1024, kernel_size=3, padding_size=1, name='conv8')
        # detection layer: 最后用一个1*1卷积去调整channel，该层没有BN层和激活函数，变成: S*S*(B*(5+C))，在这里为：13*13*425
        x = self.conv2d(x,
                        filters=(self.num_classes + 5) * self.num_bbox,
                        kernel_size=1,
                        batch_normalize=False,
                        activation=None,
                        use_bias=True,
                        name='conv_dec')
        # 实际上13x13的感受野是比较大的，对应的是大的先验框
        # 相应的52x52感受野是比较小的，检测小物体，先验框也比较小
        outputs = tf.keras.layers.Lambda(lambda x: yolo_feat_reshape(x,self.num_bbox,self.num_classes), name='reshapeTo13x13')(x)

        model = tf.keras.Model(inputs,outputs)
        model.summary()
        return model

def yolo_feat_reshape(feat,num_bbox,num_class):  #把模型的输出形状进行reshape
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    :param feat:
    :return:
    """
    grid_size = tf.shape(feat)[1]
    reshape_feat = tf.reshape(feat, [-1, grid_size, grid_size, num_bbox, num_class + 5])

    return reshape_feat

if __name__ == '__main__':
    import numpy as np
    ##---------------------------------test model ------------------------------
    model_body = Yolov2().yolo_boady()
    try:
        print("loading weights...")
        model_body.load_weights("C:/Users/xxxxx/Desktop/yolo/model/yolo2_coco.ckpt")
    except:
        print("load weights is error..")
    inputs = np.random.random([2,416,416,3]).astype("float32")
    ouputs = model_body(inputs)
    print("model outpus:",ouputs.shape) #[2,13,13,7,12]
    tf.keras.utils.plot_model(model_body,"./model_body.png",show_shapes=True)
    print("Code is running at the end.")

