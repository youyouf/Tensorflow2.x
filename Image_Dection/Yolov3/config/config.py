# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/4/3 10:59
# @Software: PyCharm
# @Brief: 配置文件
import numpy as np


# 标签的位置
annotation_path = "C:/Users/x/Desktop/MyYolov3/config/train.txt"
# 获取classes和anchor的位置
classes_path = 'classes.txt'
#用于kmeans.py 运行时存放的先验框数据
anchors_path = 'C:/Users/x/Desktop/MyYolov3/config/anchors.txt'
# 预训练模型的位置
pretrain_weights_path = r"C:/Users/x/Desktop/MyYolov3/config/convert_yolov3.h5"
# 是否预训练
pretrain = False
# 是否微调
fine_tune = True
# 训练的方式
train_mode = "fit"
# train_mode = "eager"

# 训练集和测试集的比例
valid_rate = 0.1
batch_size = 1
shuffle_size = 2

# 网络输入层信息
input_shape = (416, 416)
# 预测框的数量
num_bbox = 3

# 训练信息
epochs = 50
# 学习率
learn_rating = 1e-4

# 获得分类名
class_names =  ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']
# 类别总数
num_classes = len(class_names)

# iou忽略阈值
ignore_thresh = 0.7
iou_threshold = 0.3
# 分数的阈值（只留下高过这个阈值的box）
score_threshold = 0.5

# 先验框信息
anchors = np.array([(51, 46), (41, 65), (77, 50),
                    (59, 89), (86, 66), (125, 84),
                    (97, 130), (149, 115), (149, 115)],
                   np.float32)

# 先验框对应索引
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# tensorboard存储路径
log_dir = r"C:\Users\x\Desktop\MyYolov3\logs\summary"
# 模型路径
model_path = r"C:/Users/x/Desktop/MyYolov3/logs/yolo_test.h5"
best_model = r"C:/Users/x/Desktop/MyYolov3/logs/best_model.h5"
