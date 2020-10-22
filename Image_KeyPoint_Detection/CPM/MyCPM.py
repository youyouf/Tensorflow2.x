import tensorflow as tf
import numpy as np
import itertools
import json
import glob
import cv2

class CPM:
    def __init__(
            self,
            keypoints_num,
            stage_num,
            padding,
            batch_size,
            epochs,
            lr,
            lr_decay,
            save_path,
            train_images,
            train_labels,
            val_images,
            val_labels
    ):
        self.keypoints_num = keypoints_num
        self.stage_num = stage_num
        self.padding = padding
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.save_path = save_path
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels

    def stage_1(self, inputs):
        conv_1 = tf.keras.layers.Conv2D(128, (9,9), activation='relu', padding=self.padding)(inputs)
        pooling_1 = tf.keras.layers.MaxPooling2D((2,2), strides=(2, 2), padding=self.padding)(conv_1)
        bn_1 = tf.keras.layers.BatchNormalization()(pooling_1)

        conv_2 = tf.keras.layers.Conv2D(128, (9, 9), activation='relu', padding=self.padding)(bn_1)
        pooling_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding=self.padding)(conv_2)
        bn_2 = tf.keras.layers.BatchNormalization()(pooling_2)

        conv_3 = tf.keras.layers.Conv2D(256, (9, 9), activation='relu', padding=self.padding)(bn_2)
        pooling_3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding=self.padding)(conv_3)
        bn_3 = tf.keras.layers.BatchNormalization()(pooling_3)

        conv_4 = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding=self.padding)(bn_3)
        bn_4 = tf.keras.layers.BatchNormalization()(conv_4)

        conv_5 = tf.keras.layers.Conv2D(256, (9, 9), activation='relu', padding=self.padding)(bn_4)
        bn_5 = tf.keras.layers.BatchNormalization()(conv_5)

        conv_6 = tf.keras.layers.Conv2D(256, (1, 1), activation=None, padding=self.padding)(bn_5)
        bn_6 = tf.keras.layers.BatchNormalization()(conv_6)

        conv_7 = tf.keras.layers.Conv2D(self.keypoints_num, (1, 1), activation=None, padding=self.padding)(bn_6)

        return conv_7

    def stage_t(self, feature_org, feature_before):
        inputs = tf.keras.layers.concatenate([feature_before, feature_org])

        conv_1 = tf.keras.layers.Conv2D(64, (11, 11), activation='relu', padding=self.padding)(inputs)
        bn_1 = tf.keras.layers.BatchNormalization()(conv_1)

        conv_2 = tf.keras.layers.Conv2D(64, (11, 11), activation='relu', padding=self.padding)(bn_1)
        bn_2 = tf.keras.layers.BatchNormalization()(conv_2)

        conv_3 = tf.keras.layers.Conv2D(64, (11, 11), activation='relu', padding=self.padding)(bn_2)
        bn_3 = tf.keras.layers.BatchNormalization()(conv_3)

        conv_4 = tf.keras.layers.Conv2D(64, (1, 1), activation=None, padding=self.padding)(bn_3)
        bn_4 = tf.keras.layers.BatchNormalization()(conv_4)

        conv_5 = tf.keras.layers.Conv2D(self.keypoints_num, (1, 1), activation=None, padding=self.padding)(bn_4)

        return conv_5

    def orginal_feature_extractor(self, inputs):
        conv_1 = tf.keras.layers.Conv2D(128, (9,9), activation='relu', padding=self.padding)(inputs)
        pooling_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding=self.padding)(conv_1)
        bn_1 = tf.keras.layers.BatchNormalization()(pooling_1)

        conv_2 = tf.keras.layers.Conv2D(128, (9, 9), activation='relu', padding=self.padding)(bn_1)
        pooling_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_2)
        bn_2 = tf.keras.layers.BatchNormalization()(pooling_2)

        conv_3 = tf.keras.layers.Conv2D(256, (9, 9), activation='relu', padding=self.padding)(bn_2)
        pooling_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)(conv_3)
        bn_3 = tf.keras.layers.BatchNormalization()(pooling_3)

        conv_4 = tf.keras.layers.Conv2D(self.keypoints_num, (5,5), activation=None, padding=self.padding)(bn_3)

        return conv_4

    def forward_model(self, inputs):
        feature_org = self.orginal_feature_extractor(inputs)

        score_map = []

        for i in range(self.stage_num):
            if i == 0:
                score_map.append(self.stage_1(inputs))
            else:
                score_map.append(self.stage_t(feature_org=feature_org, feature_before=score_map[i-1]))

        return score_map
        # x = self.stage_1(inputs)
        #
        # x = self.stage_t(feature_org=feature_org, feature_before=x)
        # x = self.stage_t(feature_org=feature_org, feature_before=x)
        # x = self.stage_t(feature_org=feature_org, feature_before=x)
        # return x


    def build_model(self):
        inputs = tf.keras.Input(shape=[368, 368, 3])
        outputs = self.forward_model(inputs)
        model = tf.keras.Model(inputs = inputs, outputs = outputs)

        return model

    def train(self):
        model = self.build_model()
        model.summary()

        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(self.lr, self.lr_decay),
            metrics=['mse']
        )

        model.fit_generator(
            self.dataGenerator('training'),
            steps_per_epoch=10,
            epochs=self.epochs,
            validation_data=self.dataGenerator('validation'),
            validation_steps=10
        )

        model.save(self.save_path)

    def dataGenerator(self, mode):

        if mode == 'training':
            images = glob.glob(self.train_images + '*.jpg')
            images.sort()
            labels = glob.glob((self.train_labels + '*.json'))
            labels.sort()
            zipped = itertools.cycle(zip(images, labels))
            while True:
                x_train = []
                y_train = []
                for _ in range(self.batch_size):
                    img, label = next(zipped)
                    img = cv2.resize(cv2.imread(img, 1), (368, 368)) / 255.0

                    labels_data = json.load(open(label, 'r'))['shapes']
                    p_list = []
                    p_array = []

                    for item in labels_data:
                        p_list.append(item['points'][0])
                    for item in p_list:
                        p_array.append(self.generateHeatmap(item[0] * (46.0/400), item[1] * (46.0/600.0), variance=1.0))

                    p_array = np.array(p_array)
                    p_array = np.transpose(p_array, (1, 2, 0))

                    x_train.append(img)
                    y_train.append(p_array)
                    result = []
                    result.extend(y_train)
                    result.extend(y_train)
                    result.extend(y_train)
                    result.extend(y_train)
                yield np.array(x_train),result
        if mode == 'validation':
            images = glob.glob(self.val_images + '*.jpg')
            images.sort()
            labels = glob.glob((self.val_labels + '*.json'))
            labels.sort()
            zipped = itertools.cycle(zip(images, labels))
            while True:
                x_eval = []
                y_eval = []
                for _ in range(self.batch_size):
                    img, label = next(zipped)
                    img = cv2.resize(cv2.imread(img, 1), (368, 368)) / 255.0

                    labels_data = json.load(open(label, 'r'))['shapes']
                    p_list = []
                    p_array = []

                    for item in labels_data:
                        p_list.append(item['points'][0])
                    for item in p_list:
                        p_array.append(self.generateHeatmap(item[0] *  (46.0/400), item[1] * (46.0/600.0), variance=1.0))

                    p_array = np.array(p_array)
                    p_array = np.transpose(p_array, (1, 2, 0))

                    x_eval.append(img)
                    y_eval.append(p_array)
                    result = []
                    result.extend(y_eval)
                    result.extend(y_eval)
                    result.extend(y_eval)
                    result.extend(y_eval)
                yield np.array(x_eval), result


    def generateHeatmap(self, c_x, c_y, variance):
        gaussian_map = np.zeros((46, 46))
        for x_p in range(46):
            for y_p in range(46):
                dist_sq = (x_p - c_x) * (x_p - c_x) + \
                          (y_p - c_y) * (y_p - c_y)
                exponent = dist_sq / 2.0 / variance / variance
                gaussian_map[y_p, x_p] = np.exp(-exponent)

        return gaussian_map


cpm = CPM(
    keypoints_num=7,
    stage_num=4,
    padding='same',
    batch_size=1,
    epochs=100,
    lr=0.0001,
    lr_decay=0.0000001,
    save_path='./snapshots/model_cpm.h5',
    train_images='./datasets/training/images/',
    train_labels='./datasets/training/labels/',
    val_images='./datasets/validation/images/',
    val_labels='./datasets/validation/labels/'
)
cpm.train()
# cpm.dataGenerator('training')
# model = cpm.build_model()
# model.summary()










