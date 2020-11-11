
import tensorflow as tf
import numpy as np
import os

from configuration import Config

class DetectionDataset:
    def __init__(self):
        self.txt_file = Config.txt_file_dir
        self.batch_size = Config.batch_size

        self.traintxt_dir = "./data/train.txt"
        self.valtxt_dir = "./data/val.txt"

        if (not os.path.exists(self.traintxt_dir)) or (not os.path.exists(self.valtxt_dir)):
            with open(self.txt_file, "r") as f:  # 打开文件
                data = f.readlines()
            np.random.shuffle(data)

            train_data = data[:int(len(data) * 0.8)]
            val_data = data[int(len(data) * 0.8):]
            with open(file=self.traintxt_dir, mode="a+", encoding="utf-8") as f:
                print("First time we will create traintxt..")
                for line_text in train_data:
                    f.write(line_text)

            with open(file=self.valtxt_dir, mode="a+", encoding="utf-8") as f:
                print("First time we will create valtxt..")
                for line_text in val_data:
                    f.write(line_text)

    @staticmethod
    def __get_length_of_dataset(dataset):
        length = 0
        for _ in enumerate(dataset):
            length += 1
        return length

    def generate_datatset(self,mode="train"):
        if mode == "train":
            dataset = tf.data.TextLineDataset(filenames=self.traintxt_dir)
            length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
            train_dataset = dataset.shuffle(length_of_dataset).batch(batch_size=self.batch_size)
            return train_dataset, length_of_dataset
        else:
            dataset = tf.data.TextLineDataset(filenames=self.valtxt_dir)
            length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
            val_dataset = dataset.shuffle(length_of_dataset).batch(batch_size=self.batch_size)
            return val_dataset, length_of_dataset




class DataLoader:

    input_image_height = Config.get_image_size()[0] #获取输入图片的高度
    input_image_width = Config.get_image_size()[1]  #获取输入图片的宽度
    input_image_channels = Config.image_channels    #图片的通道数量

    def __init__(self):
        self.max_boxes_per_image = Config.max_boxes_per_image   #最大的boxes数量

    def read_batch_data(self, batch_data):     #读取batch大小的数据，返回加载后的图片以及boxes[batch, max_boxes_per_image, xmin, ymin, xmax, ymax, class_id]

        batch_size = batch_data.shape[0]
        image_file_list = []
        boxes_list = []

        for n in range(batch_size):
            image_file, boxes = self.__get_image_information(single_line=batch_data[n])
            image_file_list.append(image_file)
            boxes_list.append(boxes)
        boxes = np.stack(boxes_list, axis=0)
        image_tensor_list = []
        for image in image_file_list:
            image_tensor = DataLoader.image_preprocess(is_training=True, image_dir=image)
            image_tensor_list.append(image_tensor)
        images = tf.stack(values=image_tensor_list, axis=0)
        return images, boxes      #images为加载后的图片，boxes为[batch, max_boxes_per_image, xmin, ymin, xmax, ymax, class_id]

    def __get_image_information(self, single_line):
        """
        :param single_line: tensor  #image_file_dir + image_height + image_width + class_ids + bbox
        :return:
        image_file: string, image file dir
        boxes_array: numpy array, shape = (max_boxes_per_image, 5(xmin, ymin, xmax, ymax, class_id))
        """
        line_string = bytes.decode(single_line.numpy(), encoding="utf-8")
        line_list = line_string.strip().split(" ")
        image_file, image_height, image_width = line_list[:3]  #获取前三个数据量
        image_height, image_width = int(float(image_height)), int(float(image_width))
        boxes = []
        num_of_boxes = (len(line_list) - 3) / 5  #boxes的数量
        if int(num_of_boxes) == num_of_boxes:
            num_of_boxes = int(num_of_boxes)
        else:
            raise ValueError("num_of_boxes must be type 'int'.")
        for index in range(num_of_boxes):
            if index < self.max_boxes_per_image:
                xmin = int(float(line_list[3 + index * 5]))        #获取xmin
                ymin = int(float(line_list[3 + index * 5 + 1]))    #获取ymin
                xmax = int(float(line_list[3 + index * 5 + 2]))    #获取xmax
                ymax = int(float(line_list[3 + index * 5 + 3]))    #获取ymax
                class_id = int(line_list[3 + index * 5 + 4])       #获取类id
                xmin, ymin, xmax, ymax = DataLoader.box_preprocess(image_height, image_width, xmin, ymin, xmax, ymax)
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        num_padding_boxes = self.max_boxes_per_image - num_of_boxes
        if num_padding_boxes > 0:
            for i in range(num_padding_boxes):
                boxes.append([0, 0, 0, 0, -1])
        boxes_array = np.array(boxes, dtype=np.float32)
        return image_file, boxes_array

    @classmethod
    def box_preprocess(cls, h, w, xmin, ymin, xmax, ymax):  #将实际box输出为网络输出的大小
        resize_ratio = [DataLoader.input_image_height / h, DataLoader.input_image_width / w]  #是度量变化率，但是会有量化误差
        xmin = int(resize_ratio[1] * xmin)
        xmax = int(resize_ratio[1] * xmax)
        ymin = int(resize_ratio[0] * ymin)
        ymax = int(resize_ratio[0] * ymax)
        return xmin, ymin, xmax, ymax

    @classmethod
    def image_preprocess(cls, is_training, image_dir): #对图片进行处理以及加载
        image_raw = tf.io.read_file(filename=image_dir)
        decoded_image = tf.io.decode_image(contents=image_raw, channels=DataLoader.input_image_channels, dtype=tf.dtypes.float32)
        decoded_image = tf.image.resize(images=decoded_image, size=(DataLoader.input_image_height, DataLoader.input_image_width))
        return decoded_image


if __name__ == "__main__":
    dataloader = DetectionDataset()
    print("Code is running at the end.")