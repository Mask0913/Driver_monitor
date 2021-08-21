import cv2
import time
import argparse

import numpy as np
from PIL import Image
from keras.models import model_from_json
from processor.utils_for_mask import generate_anchors
from processor.utils_for_mask import decode_bbox
from processor.utils_for_mask import single_class_non_max_suppression
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import warnings
warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
ktf.set_session(session)




class mask:

    def __init__(self):
        # 初始化
        self.model = model_from_json(open('models/face_mask_detection.json').read())
        self.model.load_weights('models/face_mask_detection.hdf5')
        self.feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        self.anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        self.anchor_ratios = [[1, 0.62, 0.42]] * 5
        self.anchors = generate_anchors(self.feature_map_sizes, self.anchor_sizes, self.anchor_ratios)
        self.anchors_exp = np.expand_dims(self.anchors, axis=0)
        self.id2class = {0: 'Mask', 1: 'NoMask'}


    def keras_inference(self, model, img_arr):
        result = model.predict(img_arr)
        y_bboxes = result[0]
        y_scores = result[1]
        return y_bboxes, y_scores


    def inference(self, image,
                  conf_thresh=0.5,
                  iou_thresh=0.6,
                  target_shape=(260, 260),
                  draw_result=True,
                  show_result=True
                  ):
        # image = np.copy(image)
        output_info = []
        height, width, _ = image.shape
        image_resized = cv2.resize(image, target_shape)
        image_np = image_resized / 255.0  # 归一化到0~1
        image_exp = np.expand_dims(image_np, axis=0)
        y_bboxes_output, y_cls_output = self.keras_inference(self.model, image_exp)
        y_bboxes = decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                     bbox_max_scores,
                                                     conf_thresh=conf_thresh,
                                                     iou_thresh=iou_thresh,
                                                     )
        mask_boxs = []
        class_id = 0
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)
            mask = [xmin, ymin, xmax, ymax]
            mask_boxs.append(mask)
            output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        return class_id


