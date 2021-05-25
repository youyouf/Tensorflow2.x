import numpy as np
import json

import tensorflow as tf

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = tf.keras.backend.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
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


def decode_predictions(preds):
    global CLASS_INDEX
    assert len(preds.shape) == 2 and preds.shape[1] == 1000
    if CLASS_INDEX is None:
        fpath = tf.keras.utils.get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models',
                         cache_dir=".\\")
        CLASS_INDEX = json.load(open(fpath))
    indices = np.argmax(preds, axis=-1)
    results = []
    for i in indices:
        results.append(CLASS_INDEX[str(i)])
    return results
