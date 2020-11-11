# -*- coding: utf-8 -*-
# @File : voc_annotation.py
# @Author: riky
# @Time : 2020/11/11
# @Software: PyCharm
# @Brief: voc转换为yolo3读取的格式


import xml.etree.ElementTree as ET
import config.config as cfg
import os

class AnnotationToTxt():
    def convert_annotation(self, imgfilepath, xmlfilepath, xml_id, list_file,picType = ".jpeg"):
        """
        把单个xml转换成annotation格式
        :param imgfilepath: 图片文件的路径
        :param xmlfilepath: xml文件的路径
        :param xml_id: xml的id
        :param list_file: 写入的文件句柄
        :return: None
        """
        # print(xmlfilepath + xml_id)
        tree = ET.parse(xmlfilepath + xml_id)
        root = tree.getroot()

        xmlImgNme = tree.find("filename").text
        if xmlImgNme[:-len(picType)] == xml_id[:-len(".xml")]:  # 判断图片id是否和xml读出来的一致
            list_file.write(imgfilepath + xmlImgNme)
        else:
            print("xmlImgNme:", xmlImgNme, "is not equal to the ", xml_id)
            return -1

        for obj in root.iter('object'):

            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # 不在分类内部的不要，难度为1的也不要
            if cls not in cfg.class_names or int(difficult) == 1:
                print(xml_id, " 's class names is :", cls, " int(difficult):", int(difficult))
                continue

            cls_id = cfg.class_names.index(cls)
            xmlbox = obj.find('bndbox')

            b = (int(xmlbox.find('xmin').text),
                 int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))

            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        list_file.write('\n')

    def ToTxt(self,
              xmlfilepath = '../insects/val/annotations/xmls/',
              WriteImgfilepath = "C:/Users/1/Desktop/MyYolov3/insects/val/images/",  # 必须是绝对路径
              txtdir = '../config/test.txt',
              WriteTxtMode = "w",
              picType=".jpeg"
              ):
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        print("oslist file:",total_xml)

        with open(txtdir, WriteTxtMode) as files:
            for xml_id in total_xml:
                try:
                    self.convert_annotation(WriteImgfilepath, xmlfilepath, xml_id, files,picType = ".jpeg")
                except (ValueError, ArithmeticError):
                    print(ValueError, ":", ArithmeticError)
        print("write annotation to ", txtdir," file is ok!")

if __name__ == '__main__':
   print("Code is stating....")
   WriteTxt = AnnotationToTxt()

   WriteTxt.ToTxt(xmlfilepath='../insects/train/annotations/xmls/',
                  WriteImgfilepath="C:/Users/1/Desktop/MyYolov3/insects/train/images/",  #必须是绝对路径
                  txtdir = '../config/train.txt',
                  WriteTxtMode="w", #用于控制txt写入的类型 "w"：清空txt文件再写入 "a+":为在文件后面添加
                  picType=".jpeg"   #读入图片的类型，用于判断xml文件与图片的名字是否一致
                  )

   WriteTxt.ToTxt(xmlfilepath='../insects/val/annotations/xmls/',
                  WriteImgfilepath="C:/Users/1/Desktop/MyYolov3/insects/val/images/",  #必须是绝对路径
                  txtdir = '../config/test.txt',
                  WriteTxtMode="w", #用于控制txt写入的类型
                  picType=".jpeg"   #读入图片的类型，用于判断xml文件与图片的名字是否一致
                  )
   print("Code is running at the end.")