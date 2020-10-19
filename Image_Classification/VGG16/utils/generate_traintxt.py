# -*- coding: utf-8 -*-
# @File : generate_traintxt.py
# @Author: riky
# @Time : 2020/10/19
# @Software: PyCharm
# @Brief: 生成图片与标签的对应txt
import os
import config.config as cfg
class GenerateTrainTxT:
    def __init__(self,imgs_dir = r"../train",txt_dir="train.txt"):
        self.imgs_dir =imgs_dir
        self.txt_dir = txt_dir

    def WriteLabel2Txt(self):
        with open(self.txt_dir,"w") as f:
            after_generate = os.listdir(self.imgs_dir)
            for image in after_generate:
                if image.split(".")[0] == "cat":
                    f.write( self.imgs_dir + "\\" + image + ";" + "0" + "\n")
                else:
                    f.write(self.imgs_dir + "\\" + image + ";" + "1" + "\n")
        print("Write label to txt is ok!")

if __name__  == "__main__":
    Handle = GenerateTrainTxT(imgs_dir=cfg.img_dir,txt_dir=cfg.img_label_TXT_dir)
    Handle.WriteLabel2Txt()
    print("Code is running at the end.")