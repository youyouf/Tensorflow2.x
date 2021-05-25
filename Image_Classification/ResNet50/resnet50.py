# -*- coding: utf-8 -*-
# @File : VGG16(1).py
# @Author: riky
# @Time : 2020/10/20
# @Software: PyCharm
# @Brief: 网络构建

from __future__ import print_function
import tensorflow as tf

'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
'''
#模型输入图片格式(用于与训练的时候，图片处理可以按照自己的需求进行，例如img/255.--> 0,1 或者img/127.5 - 1.0-->-1,1)
"""
def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'tf', 'th'}
    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x
"""

#th输入的格式为[n,channel,height,width]
TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_th_dim_ordering_th_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'

#tf的输入格式为[n,height,width,channel]
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

#只选择tf格式的图片输入
def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last': #tf格式
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters=nb_filter1, kernel_size=(1,1),strides=(1,1), name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=nb_filter2, kernel_size=(kernel_size,kernel_size), strides=(1,1), padding='same', name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=nb_filter3, kernel_size=(1,1), strides=(1,1), name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # x = merge([x, input_tensor], mode='sum')
    x = tf.keras.layers.Add()([input_tensor, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters=nb_filter1, kernel_size = (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters = nb_filter2, kernel_size = (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=nb_filter3, kernel_size = (1, 1), name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = tf.keras.layers.Conv2D(filters = nb_filter3, kernel_size = (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # x = merge([x, shortcut], mode='sum')
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if tf.keras.backend.image_data_format() == 'channels_first':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = tf.keras.Input(shape=input_shape)
    else:
        if not tf.backen.is_keras_tensor(input_tensor):
            img_input = tf.keras.Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = tf.keras.layers.ZeroPadding2D(padding = (3, 3))(img_input)
    x = tf.keras.layers.Conv2D(filters=64,  kernel_size = (7, 7), strides=(2, 2), name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = tf.keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1000, activation='softmax', name='fc1000')(x)

    model = tf.keras.Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', tf.keras.backend.image_data_format())
        if tf.keras.backend.image_data_format() == 'channels_first':
            if include_top:
                weights_path = tf.keras.utils.get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        cache_dir=".\\")
            else:
                weights_path = tf.keras.utils.get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        cache_dir=".\\")
            model.load_weights(weights_path,by_name=True,skip_mismatch=True)
            if tf.keras.backend.backend() == 'tensorflow':
                import warnings
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                tf.keras.utils.convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = tf.keras.utils.get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        cache_dir=".\\")
            else:
                weights_path = tf.keras.utils.get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        cache_dir=".\\")
            model.load_weights(weights_path,by_name=True,skip_mismatch=True)
            if tf.keras.backend.backend() == 'theano':
                try:
                    tf.keras.utils.convert_all_kernels_in_model(model)
                except:
                    raise ("Convert all kernels in model occurs errrors..")
    return model

if __name__ == "__main__":
    model = ResNet50(include_top=True, weights='imagenet')
    model.summary()

    #test

    from tensorflow.keras.preprocessing import image
    import numpy as np
    from imagenet_utils import decode_predictions, preprocess_input
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

