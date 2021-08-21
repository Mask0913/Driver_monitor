# coding:utf-8

from .tracker_deep import update_tracker
from .StatusDetector import FaceDet
import cv2


class baseDet(object):

    def __init__(self, tracker_type):

        self.img_size = 640
        self.threshold = 0.4
        self.stride = 2
        self.tracker_type = tracker_type
        self.faceDet = FaceDet()
        self.maskflag = 0

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im, func_status=None):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': [],
            'face_statu': [],
            'status_face': None
        }
        self.frameCounter += 1
        im, faces, face_bboxes, bboxes2draw, status_face = update_tracker(self, im)
        if status_face is not None:
            if func_status['smoke']:
                if 'smoke' in bboxes2draw:
                    retDict['face_statu'].append('抽烟')
            if func_status['phone']:
                if 'phone' in bboxes2draw:
                    retDict['face_statu'].append('使用手机')
            if func_status['eat-drink']:
                if 'eat' in bboxes2draw:
                    retDict['face_statu'].append('吃喝')
            if func_status['mask-detector']:
                if 'No_mask' in bboxes2draw:
                    retDict['face_statu'].append('未佩戴口罩')

        retDict['status_face'] = status_face
        #bboxes2draw: [(320, 150, 455, 321, 'face', 1), (342, 271, 406, 364, 'phone', 2)]
        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
