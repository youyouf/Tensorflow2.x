import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

class MyGrad_CAM:
    #显示层输出的数据
    def visualize_LayerOutput(self,layer_data, num_filter = 8):
        for i in range(num_filter):
            plt.subplot(2, 4, i + 1)
            plt.imshow(layer_data[0, :, :, i])
            plt.title(str("layer")+str(i))
        plt.show()

    #获取多层网络输出的数据
    def get_Model_MultiOutput(self,model, img, begin=0, end=1):#多层输出的结果

        def _build_multi_output(model = model, begin=begin, end=end):  #创建多层输出的模型
            multi_layers_output = [layer.output for layer in model.layers[begin:end]]
            multi_output_model = tf.keras.models.Model(model.input, multi_layers_output)
            return multi_output_model

        multi_output_model = _build_multi_output()
        result = multi_output_model.predict(img)

        print("result[1].shape:",result[1].shape)
        plt.matshow(result[1][0, :, :, 0], cmap='viridis')  # 第1卷积层的第1特征层输出
        plt.matshow(result[1][0, :, :, 1], cmap='viridis')  # 第1卷积层的第0特征层输出
        plt.show()

    #进行grad_cam的热力图制作
    def grad_cam(self,
                 model = None, #输入模型
                 layer_name = None, #需要进行热力图输出的层名字
                 img_dir = None,   #图片存在的路径
                 img_preprocess_input = None#进行图片输入以及处理的函数
                 ):

        if model == None or \
                layer_name == None \
                or img_dir == None or \
                img_preprocess_input == None:
            return

        img = img_preprocess_input(img_dir) #读取模型并进行预处理

        #进行构建新的模型，也就是带有heatmap层输出的模型
        heatmap_layer = model.get_layer(layer_name)
        heatmap_model = tf.keras.models.Model([model.inputs],[heatmap_layer.output, model.output])

        #Grad-CAM算法
        with tf.GradientTape() as tape:
            conv_output, Predictions = heatmap_model(img)
            prob = Predictions[:, np.argmax(Predictions[0])]  # 最大可能性类别的预测概率
            grads = tape.gradient(prob, conv_output)  # 类别与卷积层的梯度
            pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))  # 特征层梯度的全局平均代表每个特征层权重

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1) #权重与特征层相乘，512层求和平均

        # 绘制激活热力图
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat  #归一化
        plt.matshow(heatmap[0], cmap='viridis')  #热力图的显示

        # -----------------与原有图片进行加权叠加---------------------
        orginal_img = cv2.imread(img_dir)
        heatmap1 = cv2.resize(heatmap[0], (orginal_img.shape[1], orginal_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap1 = np.uint8(255 * heatmap1)  #进行限制输出heatmap1的大小
        heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)  # cv2.COLORMAP_JET
        frame_out = cv2.addWeighted(orginal_img, 1, heatmap1, 0.4, 0)   #热力图输出为frame_out，其中1以及0.4为可调参数

        cv2.imwrite('frame_out.jpg', frame_out)
        cv2.imshow("frame_out", cv2.resize(frame_out, dsize=(512, 512)))
        plt.figure()
        plt.imshow(orginal_img)
        plt.figure()
        plt.imshow(frame_out)
        plt.show()

#图片预处理函数
def img_preprocess_input(imgdir, imgSize=(300, 300)):
    # opencv读取图片
    img = cv2.imread(imgdir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #-------------------------将长方形的图片生成正方形的图片----------------------
    img_hight, img_width, img_channel = img.shape
    if img_hight >= img_width:
        dx = (img_hight - img_width) // 2
        dy = 0
    else:
        dy = (img_width - img_hight) // 2
        dx = 0
    img = cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_CONSTANT,
                             value=(128, 128, 128))  # (将图像处理成正方形，其他数据变成灰度图片)

    #-------------------------将图片进行归一化处理[-1,1]----------------------
    img = img / 127.5 - 1.0
    img = cv2.resize(img, imgSize, interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)  # 为图片增加一维batchsize，直接设置为1
    return img

if __name__ == '__main__':
    model = tf.keras.models.load_model("logs1/Model-Weight-ep131-loss0.388-accuracy0.921-val_loss0.510-val_accuracy0.760.h5")
    model.summary()

    print("------------------------------------------------\r\n")
    for i, layer in enumerate(model.layers):  #打印网络的层以及名字
        print(i,":", layer.name)

    print("-----\r\n")
    for i in range(len(model.layers)):
        print(i,model.layers[i].name)
    print("------------------------------------------------\r\n")
    print("\r\n----------热力图计算--")
    grad_cam = MyGrad_CAM()
    grad_cam.grad_cam(model=model,
                      layer_name="Conv_1",
                      img_dir=r"neg_19.png",
                      img_preprocess_input=img_preprocess_input
                      )



