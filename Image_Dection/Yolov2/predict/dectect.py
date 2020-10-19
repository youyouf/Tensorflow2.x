import config.config as cfg
import tensorflow as tf
import numpy as np
import colorsys
import os

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
from utils import parse_yolov_output
from model.model import Yolov2

class Yolov2Dectect(object):
    def __init__(self,model_path,
                 class_names = cfg.class_names,
                 anchors = cfg.anchors,
                 score = cfg.score_threshold
                 ):
        self.model_path = model_path
        self.class_names = class_names
        self.anchors = anchors
        self.score = score
        self.model = None

    def load_model(self):
        """
        加载模型
        :return:
        """
        model = Yolov2(input_shape=cfg.input_shape, num_classes=cfg.num_classes, num_bbox=cfg.num_bbox).yolo_boady()
        print("loading weights...")
        model.load_weights(self.model_path)
        self.model = model

    def process_image(self,image):
        """
        读取图片，填充图片后归一化
        :param image: 图片路径
        :return: 图片的np数据、宽、高
        """
        # 获取原图尺寸 和 网络输入尺寸
        image_w, image_h = image.size
        w, h = cfg.input_shape
        scale = min(w / image_w, h / image_h)
        new_w = int(image_w * scale)
        new_h = int(image_h * scale)

        # 插值变换、填充图片
        image = image.resize((new_w, new_h), Image.BICUBIC)
        new_image = Image.new('RGB', cfg.input_shape, (128, 128, 128))
        new_image.paste(image, ((w - new_w) // 2, (h - new_h) // 2))

        # 归一化
        image_data = np.array(new_image, dtype=np.float32)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # 增加batch的维度

        return image_data

    def dectect(self,image):
        """
        读取模型，做出预测，并处理预测结果。将其变成正常图片下的结果，而非416x416的结果
        :param image: 图片
        :return:
        """
        height,width = image.height,image.width

        image = self.process_image(image)
        output = self.model.predict(image)

        boxes, scores, classes = parse_yolov_output(output, (height, width), self.score, max_boxes=20)
        return boxes, scores, classes

    def detect_image(self, image):
        """
        检测单张图片
        :param image: 图片
        """
        start = timer()

        # 读取预测结果
        out_boxes, out_scores, out_classes = self.dectect(image = image)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 框的厚度
        thickness = (image.size[0] + image.size[1]) // 300
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 获取坐标
            top, left, bottom, right = box
            top -= 5
            left -= 5
            bottom += 5
            right += 5

            # 防止小于0
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框、写上分类
            label = '{} {:.4f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            # 获取文字框的大小
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            # 如果文字框位置 小于 0，就在画面外边，这时候需要画在框上。在里面，就画在框上面
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 给定左上角 和 右下角坐标，画矩形
            draw.rectangle([left, top, right, bottom], outline=self.colors[c], width=thickness)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # 写上分类的文字
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print("use_time:{:.2f}s".format(end - start))

        return image

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPUS:",gpu)


    img_path = "../insects/test/images/1833.jpeg"

    yolo = Yolov2Dectect(cfg.model_path)
    yolo.load_model()

    if not os.path.exists(img_path):
        print("Error,image path is not exists.")
        exit(-1)

    image = Image.open(img_path)

    image = yolo.detect_image(image)
    image.show()
