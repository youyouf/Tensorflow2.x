# -*- coding: utf-8 -*-
# @File : config.py
# @Software: PyCharm
# @Brief: 配置文件
import numpy as np

# 标签的位置(绝对路径)
annotation_path = "C:/Users/xxxxx/Desktop/Yolov2/config/train.txt"
#用于kmeans.py 运行时存放的先验框数据(绝对路径)
anchors_path = 'C:/Users/xxxxx/Desktop/Yolov2/config/anchors.txt'
# 预训练模型的位置(绝对路径)
pretrain_weights_path = r"C:/Users/xxxxx/Desktop/Yolov2/config/convert_yolov3.h5"
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
img_enhance_rate = 0.8 #图片进行数据增强的概率

# 网络输入层信息
input_shape = (416, 416)
#下采样的数量(网络输出栅格大小)
GRID_H, GRID_W = 13, 13 # GRID size = IMAGE size / 32
# 训练信息
epochs = 30
# 学习率
learn_rating = (1e-4) * batch_size

# 获得分类名
class_names = ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']
# 类别总数
num_classes = len(class_names)

# iou忽略阈值
ignore_thresh = 0.7
iou_threshold = 0.3
# 分数的阈值（只留下高过这个阈值的box）
score_threshold = 0.5

# 先验框信息
anchors = np.array([(1.07463884, 1.41780199), (0.80732484, 0.79700272), (0.42992126, 0.63745981),
                    (0.61904762, 0.4310145), (0.91246057, 0.58912387), (0.59815951, 0.93003145),
                    (1.41990291, 0.99027553)],np.float32)
# 预测框的数量
num_bbox = int(len(anchors))
# 先验框对应索引
anchor_masks = np.array(range(len(anchors)))

# tensorboard存储路径(绝对路径)
log_dir = r"C:\Users\xxxxx\Desktop\Yolov2\logs\summary"
# 模型路径(绝对路径)
model_path = r"C:/Users/xxxxx/Desktop/Yolov2/logs/best_model.h5"#r"C:/Users/xxxxx/Desktop/Yolov2/logs/yolo_test.h5"
best_model = r"C:/Users/xxxxx/Desktop/Yolov2/logs/best_model.h5"
