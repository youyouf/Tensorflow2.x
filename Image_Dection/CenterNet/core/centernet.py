import tensorflow as tf
import numpy as np

from configuration import Config
from core.models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from core.models.dla import dla_34, dla_60, dla_102, dla_169
from core.models.efficientdet import d0, d1, d2, d3, d4, d5, d6, d7

backbone_zoo = {"resnet_18": resnet_18(),
                "resnet_34": resnet_34(),
                "resnet_50": resnet_50(),
                "resnet_101": resnet_101(),
                "resnet_152": resnet_152(),
                "dla_34": dla_34(),
                "dla_60": dla_60(),
                "dla_102": dla_102(),
                "dla_169": dla_169(),
                "D0": d0(),
                "D1": d1(),
                "D2": d2(),
                "D3": d3(),
                "D4": d4(),
                "D5": d5(),
                "D6": d6(),
                "D7": d7()
                }

class CenterNet(tf.keras.Model):  #CenterNet的网络架构
    def __init__(self):
        super(CenterNet, self).__init__()
        self.backbone = backbone_zoo[Config.backbone_name]

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs, training=training)
        # print("CenterNet x[0].shape:",x[0].shape," x[1].shape:",x[1].shape," x[2].shape:",x[2].shape)
        x = tf.concat(values=x, axis=-1)
        # print("CenterNet Out x.shape:",x.shape)
        return x


if __name__ == "__main__":
    print("Code is start.....")
    centerNet = CenterNet()

    sample_inputs = tf.random.normal(shape=(Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    sample_outputs = centerNet(sample_inputs, training=True)
    centerNet.summary()
    print("inputs.shape:",sample_inputs.shape,"outputs.shape:",sample_outputs.shape)
    print("Code is running at the end.")