import tensorflow as tf
import numpy as np
import cv2
import glob
import itertools

class UNet:
    def __init__(
            self,
            input_width,
            input_height,
            num_classes,
            train_images,
            train_instances,
            val_images,
            val_instances,
            epochs,
            lr,
            lr_decay,
            batch_size,
            save_path
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.num_classes = num_classes
        self.train_images = train_images
        self.train_instances = train_instances
        self.val_images = val_images
        self.val_instances = val_instances
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.save_path = save_path

    def leftNetwork(self, inputs):
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')(inputs)
        o_1 = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2))(o_1)

        x = tf.keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
        o_2 = tf.keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(o_2)

        x = tf.keras.layers.Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
        o_3 = tf.keras.layers.Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(o_3)

        x = tf.keras.layers.Conv2D(512, (3, 3), padding='valid', activation='relu')(x)
        o_4 = tf.keras.layers.Conv2D(512, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(o_4)

        x = tf.keras.layers.Conv2D(1024, (3, 3), padding='valid', activation='relu')(x)
        o_5 = tf.keras.layers.Conv2D(1024, (3, 3), padding='valid', activation='relu')(x)

        return [o_1, o_2, o_3, o_4, o_5]

    def rightNetwork(self, inputs):
        c_1, c_2, c_3, c_4, o_5 = inputs



        o_5 = tf.keras.layers.UpSampling2D((2, 2))(o_5)
        x = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(4)(c_4), o_5], axis=3)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(16)(c_3), x], axis=3)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(40)(c_2), x], axis=3)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(88)(c_1), x], axis=3)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.Conv2D(self.num_classes, (1, 1), padding='valid')(x)
        x = tf.keras.layers.Activation('softmax')(x)

        return x

    def build_model(self):
        inputs = tf.keras.Input(shape=[self.input_height, self.input_width, 3])
        left_output = self.leftNetwork(inputs)
        right_output = self.rightNetwork(left_output)

        model = tf.keras.Model(inputs=inputs, outputs=right_output)

        return model

    def train(self):
        G_train = self.dataGenerator(model='training')
        G_eval = self.dataGenerator(model='validation')

        model = self.build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr, self.lr_decay),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy', 'Recall', 'AUC']
        )
        model.fit_generator(
            G_train, 5, validation_data=G_eval, validation_steps=5, epochs=self.epochs
        )
        model.save(self.save_path)

    def dataGenerator(self, model):
        if model == 'training':
            images = glob.glob(self.train_images + '*.jpg')
            images.sort()
            instances = glob.glob(self.train_instances + '*.png')
            instances.sort()
            zipped = itertools.cycle(zip(images, instances))
            while True:
                x_train = []
                y_train = []
                for _ in range(self.batch_size):
                    img, seg = next(zipped)
                    img = cv2.resize(cv2.imread(img, 1), (self.input_width, self.input_height)) / 255.0
                    seg = tf.keras.utils.to_categorical(cv2.imread(seg, 0), self.num_classes)
                    x_train.append(img)
                    y_train.append(seg)

                yield np.array(x_train), np.array(y_train)

        if model == 'validation':
            images = glob.glob(self.val_images + '*.jpg')
            images.sort()
            instances = glob.glob(self.val_instances + '*.png')
            instances.sort()
            zipped = itertools.cycle(zip(images, instances))

            while True:
                x_eval = []
                y_eval = []
                for _ in range(self.batch_size):
                    img, seg = next(zipped)
                    img = cv2.resize(cv2.imread(img, 1), (self.input_width, self.input_height)) / 255.0
                    seg = tf.keras.utils.to_categorical(cv2.imread(seg, 0), self.num_classes)
                    x_eval.append(img)
                    y_eval.append(seg)

                yield np.array(x_eval), np.array(y_eval)


unet = UNet(
    input_width=572,
    input_height=572,
    num_classes=3,
    train_images='./datasets/training/images/',
    train_instances='./datasets/training/instances/',
    val_images='./datasets/validation/images/',
    val_instances='./datasets/validation/instances/',
    epochs=200,
    lr=0.0001,
    lr_decay=0.00001,
    batch_size=5,
    save_path='./snapshots/model.h5'
)

unet.train()

# import cv2
# import numpy as np
# img = cv2.imread('./datasets/training/instances/1.png', 0)
# print(np.unique(img))
