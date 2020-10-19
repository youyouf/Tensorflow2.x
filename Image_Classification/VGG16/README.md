# 使用Tensorflow2.x实现VGG16

### 进度

- [x] 模型搭建（VGG16）
- [x] 使用model.fit_generator方式训练
- [x] 迁移学习（加载了imageNet的权重进行冻结训练）
- [x] GPU加速
- [x] 数据增强（加载的图片没有进行数据增强了处理，例如图片不变性缩放等）

### 迁移学习与微调

​		加载权重进行训练，冻结网络的几层进行训练

### config - 配置文件

​		配置文件

### utils - 数据加载

​		将图片和标签进行联合生成txt文件
		采用yield方式对图片和标签进行加载
		数据集合采用猫狗数据集

### Reference

- https://blog.csdn.net/weixin_44791964/article/details/102779878
	- 模型大体结构的构建
	- 数据加载
	