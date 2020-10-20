
'''VGG16 model for tensorflow.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

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
from __future__ import print_function
import tensorflow as tf

#th输入的格式为[n,channel,height,width]
# TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
# TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'

#tf的输入格式为[n,height,width,channel]
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

#只选择tf格式的图片输入
def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    #Determine proper input shape
    if include_top:
        input_shape = (224, 224, 3)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = tf.keras.Input(shape=input_shape)
    else:
        img_input = tf.keras.Input(shape=input_tensor)

    # Block 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1')(img_input)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size = (3, 3), padding='same', activation='relu', name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(filters=512, kernel_size = (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size = (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size = (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, kernel_size = (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = tf.keras.Model(img_input, x)
    model.summary()
    """
    --------#cache_subdir="./",
            cache_dir="./"  ---会创建一个dataset文件夹用于保存模型权重
            weights_path -->./datasets\vgg16_weights_tf_dim_ordering_tf_kernels.h5

    --------#cache_subdir="./", (容易出错)
            cache_dir="./"  ---会创建一个dataset文件夹用于保存模型权重       
            print(weights_path)-->././vgg16_weights_tf_dim_ordering_tf_kernels.h5

   --------cache_subdir="./",
            #cache_dir="./"  ---会创建一个dataset文件夹用于保存模型权重       
            print(weights_path)--> 保存在c盘的其他某个位置
    """
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file(fname="vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                                                   origin=TF_WEIGHTS_PATH,
                                                   #cache_subdir="./",
                                                   cache_dir=".\\")
            print("weights_path:",weights_path)
        else:
            weights_path = tf.keras.utils.get_file(fname="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                                   origin=TF_WEIGHTS_PATH_NO_TOP,
                                                   #cache_subdir="./",
                                                   cache_dir=".\\")
            print("weights_path:",weights_path)
        try:
            model.load_weights(weights_path,by_name=True,skip_mismatch=True)
            print("Loading weights is sucessful.")
        except:
            print("Model load weights occurs error. This may be due to an incomplete model weight download error")

    return model

if __name__ == "__main__":

    model = VGG16()


