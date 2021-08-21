import logging as log
import os
import pathlib
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import torch
import torchvision
from .tracker import update_tracker
from .StatusDetector import FaceDet


def xywh2xyxy(x):

    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y


def non_max_suppression(prediction, conf_thres=0.05, iou_thres=0.4):

    prediction = torch.from_numpy(prediction)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output


class Detector(object):

    def __init__(self):

        self.device = 'GPU'
        self.threshold = 0.1
        self.stride = 10
        self.size = 640
        self.faceDet = FaceDet()
        log.basicConfig(level=log.DEBUG)

        # For objection detection task, replace your target labels here.
        self.label_id_map = ["face", "normal", "phone",
                             "write", "smoke", "eat", "camputer", "sleep"]
        self.model_xml = 'weights/openvo_fp16_yolov5/last.xml'
        self.net = self.init_model()
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        self.build_config()

    def init_model(self):
        if not os.path.isfile(self.model_xml):
            log.error('model_xml does not exist')
            return None
        model_bin = pathlib.Path(self.model_xml).with_suffix('.bin').as_posix()
        net = IENetwork(model=self.model_xml, weights=model_bin)

        ie = IECore()
        self.exec_net = ie.load_network(network=net, device_name=self.device)
        input_blob = next(iter(net.inputs))
        n, c, h, w = net.inputs[input_blob].shape
        self.input_h, self.input_w = h, w
        self.input_c, self.input_n = c, n

        return net

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def preprocess(self, img):

        img0 = img.copy()
        img = cv2.resize(img, (self.size, self.size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float) / 255.0  # 图像归一化
        img = np.expand_dims(img, axis=0)

        return img0, img

    def detect(self, input_image):
        if not self.net or input_image is None:
            log.error('Invalid input args')
            return None
        ih, iw, _ = input_image.shape

        input_image, images = self.preprocess(input_image)
        res = self.exec_net.infer(inputs={self.input_blob: images})

        data = res[self.out_blob]

        data = non_max_suppression(data, 0.4, 0.5)
        detect_objs = []
        if data[0] == None:
            return detect_objs
        else:
            data = data[0].numpy()
            for proposal in data:
                if proposal[4] > self.threshold:
                    xmin = int(iw * (proposal[0]/self.size))
                    ymin = int(ih * (proposal[1]/self.size))
                    xmax = int(iw * (proposal[2]/self.size))
                    ymax = int(ih * (proposal[3]/self.size))
                    # xmin = int(iw * (proposal[0]))
                    # ymin = int(ih * (proposal[1]))
                    # xmax = int(iw * (proposal[2]))
                    # ymax = int(ih * (proposal[3]))
                    detect_objs.append((
                        xmin, ymin, xmax,
                        ymax, self.label_id_map[int(proposal[5])]
                    ))

            return input_image, detect_objs

    def feedCap(self, im, func_status):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1
        with_headpose = func_status['headpose']

        im, faces, face_bboxes = update_tracker(self, im, with_headpose)

        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes

        return retDict


if __name__ == '__main__':

    predictor = Detector()
    base = 'images'

    for p in os.listdir(base):

        img = cv2.imread(os.path.join(base, p))
        result = predictor.feedCap(img)
        cv2.imshow('result', img)
        cv2.waitKey(0)
