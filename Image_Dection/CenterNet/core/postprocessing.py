import tensorflow as tf
import numpy as np

from configuration import Config
from utils.gaussian import gaussian_radius, draw_umich_gaussian
from core.loss import CombinedLoss, RegL1Loss

class PostProcessing:
    @staticmethod
    def training_procedure(batch_labels, pred):
        gt = EnCoder(batch_labels)  #也就是GT
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices = gt.get_gt_values()
        loss_object = CombinedLoss()
        loss = loss_object(y_pred=pred, heatmap_true=gt_heatmap, reg_true=gt_reg, wh_true=gt_wh, reg_mask=gt_reg_mask, indices=gt_indices)
        return loss

    @staticmethod
    def testing_procedure(pred, original_image_size):
        decoder = Decoder(original_image_size)
        detections = decoder(pred)
        bboxes = detections[:, 0:4]
        scores = detections[:, 4]
        clses = detections[:, 5]
        return bboxes, scores, clses

#可以说是对数据集进行编码
class EnCoder:#返回的是网络的真实框转到网络大小输出时候的编码
    def __init__(self, batch_labels):
        self.downsampling_ratio = Config.downsampling_ratio   #获取网络输出的大小与网络输入的下降率
        self.features_shape = np.array(Config.get_image_size(), dtype=np.int32) // self.downsampling_ratio #变成网络输出大小
        self.batch_labels = batch_labels                 #batch 标签信息
        self.batch_size = batch_labels.shape[0]          #batch的大小

    def get_gt_values(self):
        gt_heatmap = np.zeros(shape=(self.batch_size, self.features_shape[0], self.features_shape[1], Config.num_classes), dtype=np.float32) #gt_heatmap
        gt_reg = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32) #
        gt_wh = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32)
        gt_reg_mask = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
        gt_indices = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)

        for i, label in enumerate(self.batch_labels):
            label = label[label[:, 4] != -1]  #获取正常的数据出来
            hm, reg, wh, reg_mask, ind = self.__decode_label(label)
            gt_heatmap[i, :, :, :] = hm
            gt_reg[i, :, :] = reg
            gt_wh[i, :, :] = wh
            gt_reg_mask[i, :] = reg_mask
            gt_indices[i, :] = ind
        return gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices

    def __decode_label(self, label):
        hm = np.zeros(shape=(self.features_shape[0], self.features_shape[1], Config.num_classes), dtype=np.float32)
        reg = np.zeros(shape=(Config.max_boxes_per_image, 2), dtype=np.float32)    #每个标签的偏移量
        wh = np.zeros(shape=(Config.max_boxes_per_image, 2), dtype=np.float32)     #每个标签的框与高
        reg_mask = np.zeros(shape=(Config.max_boxes_per_image), dtype=np.float32)  #那个标签有物体
        ind = np.zeros(shape=(Config.max_boxes_per_image), dtype=np.float32)       #

        for j, item in enumerate(label):
            item[:4] = item[:4] / self.downsampling_ratio   #变成输出网络模型大小
            xmin, ymin, xmax, ymax, class_id = item         #
            class_id = class_id.astype(np.int32)            #进行class id的大小
            h, w = int(ymax - ymin), int(xmax - xmin)       #获取高度，宽度
            radius = gaussian_radius((h, w))                #获取高斯半径
            radius = max(0, int(radius))                    #
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2        #获取中心点
            center_point = np.array([ctr_x, ctr_y], dtype=np.float32)  #组合层中心点
            center_point_int = center_point.astype(np.int32)           #变成中心点 整形

            draw_umich_gaussian(hm[:, :, class_id], center_point_int, radius)  #进行高斯绘点

            reg[j] = center_point - center_point_int                           #偏移值，真实边框的偏移量
            wh[j] = 1. * w, 1. * h                                             #是框的宽与高
            reg_mask[j] = 1                                                    #说明在这点是有物体的
            ind[j] = center_point_int[1] * self.features_shape[1] + center_point_int[0]
        return hm, reg, wh, reg_mask, ind


#对网络的输出进行解码
class Decoder:
    def __init__(self, original_image_size):

        self.K = Config.max_boxes_per_image
        self.original_image_size = np.array(original_image_size, dtype=np.float32)
        self.input_image_size = np.array(Config.get_image_size(), dtype=np.float32)
        self.downsampling_ratio = Config.downsampling_ratio
        self.score_threshold = Config.score_threshold

    def __call__(self, pred, *args, **kwargs):
        heatmap, reg, wh = tf.split(value=pred, num_or_size_splits=[Config.num_classes, 2, 2], axis=-1)
        heatmap = tf.math.sigmoid(heatmap)
        batch_size = heatmap.shape[0]
        heatmap = Decoder.__nms(heatmap)
        scores, inds, clses, ys, xs = Decoder.__topK(scores=heatmap, K=self.K)

        if reg is not None:

            reg = RegL1Loss.gather_feat(feat=reg, idx=inds)
            xs = tf.reshape(xs, shape=(batch_size, self.K, 1)) + reg[:, :, 0:1]
            ys = tf.reshape(ys, shape=(batch_size, self.K, 1)) + reg[:, :, 1:2]

        else:

            xs = tf.reshape(xs, shape=(batch_size, self.K, 1)) + 0.5
            ys = tf.reshape(ys, shape=(batch_size, self.K, 1)) + 0.5

        wh = RegL1Loss.gather_feat(feat=wh, idx=inds)
        clses = tf.cast(tf.reshape(clses, (batch_size, self.K, 1)), dtype=tf.float32)
        scores = tf.reshape(scores, (batch_size, self.K, 1))
        bboxes = tf.concat(values=[xs - wh[..., 0:1] / 2,
                                   ys - wh[..., 1:2] / 2,
                                   xs + wh[..., 0:1] / 2,
                                   ys + wh[..., 1:2] / 2], axis=2)
        detections = tf.concat(values=[bboxes, scores, clses], axis=2)
        return self.__map_to_original(detections)

    def __map_to_original(self, detections):
        bboxes, scores, clses = tf.split(value=detections, num_or_size_splits=[4, 1, 1], axis=2)
        bboxes, scores, clses = bboxes.numpy()[0], scores.numpy()[0], clses.numpy()[0]

        resize_ratio = self.original_image_size / self.input_image_size
        bboxes[:, 0::2] = bboxes[:, 0::2] * self.downsampling_ratio * resize_ratio[1]
        bboxes[:, 1::2] = bboxes[:, 1::2] * self.downsampling_ratio * resize_ratio[0]
        bboxes[:, 0::2] = np.clip(a=bboxes[:, 0::2], a_min=0, a_max=self.original_image_size[1])
        bboxes[:, 1::2] = np.clip(a=bboxes[:, 1::2], a_min=0, a_max=self.original_image_size[0])
        score_mask = scores >= self.score_threshold
        bboxes, scores, clses = Decoder.__numpy_mask(bboxes, np.tile(score_mask, (1, 4))), Decoder.__numpy_mask(scores, score_mask), Decoder.__numpy_mask(clses, score_mask)
        detections = np.concatenate([bboxes, scores, clses], axis=-1)
        return detections

    @staticmethod
    def __numpy_mask(a, mask):
        return a[mask].reshape(-1, a.shape[-1])

    @staticmethod
    def __nms(heatmap, pool_size=3):
        hmax = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(heatmap)
        keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
        return hmax * keep

    @staticmethod
    def __topK(scores, K):
        B, H, W, C = scores.shape #B，H，W，C
        scores = tf.reshape(scores, shape=(B, -1)) # (B, H*W*C)
        topk_scores, topk_inds = tf.math.top_k(input=scores, k=K, sorted=True)
        topk_clses = topk_inds % C
        topk_xs = tf.cast(topk_inds // C % W, tf.float32)
        topk_ys = tf.cast(topk_inds // C // W, tf.float32)
        topk_inds = tf.cast(topk_ys * tf.cast(W, tf.float32) + topk_xs, tf.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
