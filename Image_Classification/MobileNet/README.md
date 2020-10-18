# 使用MobileNet实现分类任务

### 进度

- [x] 使用tf.data读取数据
- [x] 模型搭建（MobileNet）
- [x] 使用model.fit方式训练
- [x] 迁移学习
- [x] 数据增强
- [x] GPU加速

### 关于迁移学习

​		使用的是tensorflow提供的预训练模型(另外构建了自己的MobileNet模型)进行训练，图片输入大小为224x224x3(没有进行微调)

### config -配置文件

​		config采用的是较为常用的py配置文件，方便进行使用

### Reference

- https://github.com/dragen1860
	- 数据加载
