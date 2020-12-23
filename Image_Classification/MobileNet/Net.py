# -*- coding: utf-8 -*-
# @File : Net.py
# @Author:Riky
# @Time : 2020/10/18
# @Software: PyCharm
# @Brief: 网络结构
import tensorflow as tf

def Net(num_class, #类别
        input_shape=(300,300,3),#输入图片的大小
        dropoutRate0=0.35,      #dropout的机率值
        dropoutRate1=0.30,      #dropout的机率值
        unfreeze_layers_num = 3,  #解冻层数
        chocie = 1              #网络构建形式，建议选择1,因为可以选择1生成的模型可以进行热力图输出
        ):

    if chocie == 0:
        inputs = tf.keras.Input(shape=input_shape)
        MobileNetV2 = tf.keras.applications.MobileNetV2(input_shape=input_shape,  #不一样
                                                        include_top=False,
                                                        weights='imagenet'
                                                        )
        MobileNetV2.trainable = True
        MobileNetV2.summary()
        print("-------------------------Before freeze freMobileNetV2------------------------")
        print("weights:", len(MobileNetV2.weights))
        print("trainable_weights:", len(MobileNetV2.trainable_weights))
        print("non_trainable_weights:", len(MobileNetV2.non_trainable_weights))

        print("\r\nfreeze layers.....")
        # ------------------------------------------------------#
        #   主干特征提取网络特征通用，冻结训练可以加快训练速度
        #   也可以在训练初期防止权值被破坏。
        #   提示OOM或者显存不足请调小Batch_size
        # ------------------------------------------------------#
        freeze_layers = len(MobileNetV2.layers) - unfreeze_layers_num
        for i in range(freeze_layers):
            MobileNetV2.layers[i].trainable = False

        print("-------------------------After freeze freMobileNetV2------------------------")
        print("weights:", len(MobileNetV2.weights))
        print("trainable_weights:", len(MobileNetV2.trainable_weights))
        print("non_trainable_weights:", len(MobileNetV2.non_trainable_weights))
        x = MobileNetV2(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropoutRate0)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Dropout(dropoutRate1)(x)
        outputs = tf.keras.layers.Dense(int(num_class), activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.summary()
        return model
    else:
        inputs = tf.keras.layers.Input(shape=input_shape)  #输入改成用layer层
        MobileNetV2 = tf.keras.applications.MobileNetV2(input_tensor=inputs,  #这个与之前的不一样
                                                        include_top=False,
                                                        weights='imagenet',
                                                        )
        #先允许网络进行训练
        MobileNetV2.trainable = True
        MobileNetV2.summary()

        print("-------------------------Before freeze freMobileNetV2------------------------")
        print("weights:", len(MobileNetV2.weights))
        print("trainable_weights:", len(MobileNetV2.trainable_weights))
        print("non_trainable_weights:", len(MobileNetV2.non_trainable_weights))

        print("\r\nfreeze layers.....")
        # ------------------------------------------------------#
        #   主干特征提取网络特征通用，冻结训练可以加快训练速度
        #   也可以在训练初期防止权值被破坏。
        #   提示OOM或者显存不足请调小Batch_size
        # ------------------------------------------------------#
        freeze_layers = len(MobileNetV2.layers) - unfreeze_layers_num
        for i in range(freeze_layers):
            MobileNetV2.layers[i].trainable = False

        print("-------------------------After freeze freMobileNetV2------------------------")
        print("weights:", len(MobileNetV2.weights))
        print("trainable_weights:", len(MobileNetV2.trainable_weights))
        print("non_trainable_weights:", len(MobileNetV2.non_trainable_weights))

        x = tf.keras.layers.GlobalAveragePooling2D()(MobileNetV2.output)
        x = tf.keras.layers.Dropout(dropoutRate0)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Dropout(dropoutRate1)(x)
        outputs = tf.keras.layers.Dense(int(num_class),activation= "softmax")(x)
        model = tf.keras.Model(inputs,outputs)
        model.summary()

        print("-------------------------After freeze model------------------------")
        print("weights:", len(model.weights))
        print("trainable_weights:", len(model.trainable_weights))
        print("non_trainable_weights:", len(model.non_trainable_weights))

        return model

if __name__ == "__main__":
    model = Net(5)
    print("Code is running at the end.")