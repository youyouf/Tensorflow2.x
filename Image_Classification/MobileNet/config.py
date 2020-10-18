
# @File : config.py
# @Author:
# @Time : 2020/10/18
# @Software: PyCharm
# @Brief: 配置文件
import tensorflow as tf

#图片存放总路径
images_path = r"./pokeman"

# 网络输入层信息
input_shape = (224, 224)

# 训练集、验证集、测试集的比例
train_rate = 0.6
valid_rate = 0.2
test_rate = (1-train_rate - valid_rate) #需要大于零

# batch的大小
batch_size = 2
shuffle_size = 2
# 训练轮数
epochs = 5
# 学习率
learn_rating = (1e-4)*batch_size

#图片归一化的操作
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])
img_enhanceRate = 0.8 #图片进行增强的概率
