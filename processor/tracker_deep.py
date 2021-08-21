# coding:utf-8

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from processor.mask import mask
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import datetime





palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
mask = mask()


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'smoke':
            cls_id = '抽烟'
        if cls_id == 'eat':
            cls_id = '吃喝'
        if cls_id == 'phone':
            cls_id = '接电话'
        if cls_id == 'No_mask':
            cls_id = '无口罩'
        if cls_id == "normal":
            continue
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        if cls_id in ['抽烟', '吃喝', '接电话', '无口罩']:
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            t_size = (int(t_size[0]/2), t_size[1])
        else:
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        if cls_id == '无口罩':
            # cv2.putText(image, '{}'.format(cls_id), (int((c1[0] + c2[0])/2), int((c1[1] + c2[1])/2)), 0, tl / 3,
            #             [0, 0, 255], thickness=tf+1, lineType=cv2.LINE_AA)
            image = cv2ImgAddText(image, cls_id, int(c1[0]), int(c1[1] - 2), (255, 0, 0), 18)
            continue
        cv2.rectangle(image, (int(c1[0]), int(c1[1])), (c2[0], c2[1] - 5), color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(image, '{}'.format(cls_id), (c1[0], c1[1] - 2), 0, tl / 3,
        #             [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        image = cv2ImgAddText(image, cls_id, int(c1[0]), int(c1[1] - 22), (255, 255, 255), 18)
    now = str(datetime.datetime.now())
    now = '时间： ' + now[:-7]
    image = cv2ImgAddText(image, now, 230, 30, (255, 255, 255), 18)
    return image

def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "Font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



def update_tracker(target_detector, image):
    status_face = None
    new_faces = []
    ismask = mask.inference(image)
    _, bboxes = target_detector.detect(image)

    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)
    try:
        outputs = deepsort.update(xywhs, confss, clss, image)
    except:
        return image, new_faces, None, None, None
    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        if cls_ == 'face':
            if ismask:
                bboxes2draw.append((x1, y1, x2, y2, 'No_mask', 100)
        )
        current_ids.append(track_id)
        if cls_ == 'face':
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )
            status_face = image[y1:y2, x1:x2]
    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
    image = plot_bboxes(image, bboxes2draw)
    return image, new_faces, face_bboxes, str(bboxes2draw), status_face
