# -*- coding: utf-8 -*-

# @File : VGG16(1).py
# @Author: riky
# @Time : 2020/10/20
# @Software: PyCharm
# @Brief: 网络构建

'''Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790.
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.

# Reference:

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

'''
from __future__ import print_function
import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Xception(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained
    on ImageNet. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    You should set `image_data_format="channels_last"` in your Keras config
    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if tf.keras.backend.backend() != 'tensorflow':
        raise RuntimeError('The Xception model is only available with '
                           'the TensorFlow backend.')
    if tf.keras.backend.image_data_format() != 'channels_last':
        import warnings
        warnings.warn('The Xception model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        tf.keras.backend.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=tf.keras.backend.image_data_format(),
                                      require_flatten=include_top)


    if input_tensor is None:
        img_input = tf.keras.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = tf.keras.layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block1_conv1_act')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, name='block1_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block1_conv2_act')(x)

    residual = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2),
                                      padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = tf.keras.layers.Add()([x, residual])

    residual = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = tf.keras.layers.Add()([x, residual])

    residual = tf.keras.layers.Conv2D(filters=728, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = tf.keras.layers.Add()([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = tf.keras.layers.Add()([x, residual])

    residual = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2),
                                      padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block13_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block13_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = tf.keras.layers.Add()([x, residual])

    x = tf.keras.layers.SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block14_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = tf.keras.layers.SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block14_sepconv2_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        # inputs = tf.keras.engine.topology.get_source_inputs(input_tensor)
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = tf.keras.Model(inputs, x, name='xception')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                                   TF_WEIGHTS_PATH,
                                                   cache_subdir='models',
                                                   cache_dir=".\\")
        else:
            weights_path = tf.keras.utils.get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                                   TF_WEIGHTS_PATH_NO_TOP,
                                                   cache_subdir='models',
                                                   cache_dir=".\\")
        #model.load_weights(weights_path,by_name=True,skip_mismatch=True)
        model.load_weights(weights_path)

    if old_data_format:
        tf.keras.backend.set_image_data_format(old_data_format)
    return model

if __name__ == '__main__':
    model = Xception(include_top=True, weights='imagenet')
    model.summary()

'''#测试使用
   from tensorflow.keras.preprocessing import image
   import numpy as np

   def preprocess_input(x):
       x /= 255.
       x -= 0.5
       x *= 2.
       return x

   CLASS_INDEX = None
   CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

   def decode_predictions(preds, top=5):
       import json
       global CLASS_INDEX
       if len(preds.shape) != 2 or preds.shape[1] != 1000:
           raise ValueError('`decode_predictions` expects '
                            'a batch of predictions '
                            '(i.e. a 2D array of shape (samples, 1000)). '
                            'Found array with shape: ' + str(preds.shape))
       if CLASS_INDEX is None:
           fpath = tf.keras.utils.get_file('imagenet_class_index.json',
                                           CLASS_INDEX_PATH,
                                           cache_subdir='models',
                                           cache_dir=".\\")
           CLASS_INDEX = json.load(open(fpath))
       results = []
       for pred in preds:
           top_indices = pred.argsort()[-top:][::-1]
           result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
           results.append(result)
       return results

   img_path = 'elephant.jpg'
   img = image.load_img(img_path, target_size=(299, 299))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)
   print('Input image shape:', x.shape)

   preds = model.predict(x)
   print(np.argmax(preds))
   print('Predicted:', decode_predictions(preds, 1))
'''