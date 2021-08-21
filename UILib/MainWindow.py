from .Layout import *
import numpy as np
import datetime
from playsound import playsound
import _thread
import cv2
import time
import dlib

class MainWindow(MainWindowLayOut):

    def __init__(self, opt):
        super(MainWindow, self).__init__(opt)
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1("weights/dlib_face_recognition_resnet_model_v1.dat")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)
        self.Recodeface = None
        self.data = []
        self.face_data = {}
        self.pre_time = datetime.datetime.now()
        self.Faceid = 1
        self.reco_face_id = -1
        self.max_face_id = 0
        self.show_face_id = 0
        self.update_face_time = 0
        self.update_face_flag = 0
        self.faces_recorded = None

    def update_graph_values(self, face_status):
        for k, v in face_status.items():
            if not k in self.graph_values:
                continue
            self.graph_values[k].append(float(v))
            if len(self.graph_values[k]) > self.values_max_num:
                self.graph_values[k].pop(0)

            if k == 'Blinks':
                self.curve11.setData(np.arange(len(self.graph_values[k])),
                                     1 - np.array(self.graph_values[k]))
                self.curve12.setData(np.arange(len(self.graph_values[k])),
                                     np.array(self.graph_values[k]))
            elif k == 'Yawning':
                self.curve21.setData(np.arange(len(self.graph_values[k])),
                                     1 - np.array(self.graph_values[k]))
                self.curve22.setData(np.arange(len(self.graph_values[k])),
                                     np.array(self.graph_values[k]))
            else:
                self.curve31.setData(np.arange(len(self.graph_values[k])),
                                     1 - np.array(self.graph_values[k]))
                self.curve32.setData(np.arange(len(self.graph_values[k])),
                                     np.array(self.graph_values[k]))

    def update_image(self, pt='face.jpg'):
        self.data = []
        _, frame = self.vs.read()
        if frame is None:
            return
        frame = imutils.resize(frame, height=500)
        raw = frame.copy()
        packet = self.processor.getProcessedImage(frame, self.func_status)
        now = datetime.datetime.now()
        faces_recorded = packet['faces']
        if len(packet['face_statu']) > 0 and (now - self.pre_time).seconds > 3:
            license_number = 'Driver ' + str(self.show_face_id)
            illegal = packet['face_statu'][0]
            if illegal == '抽烟':
                _thread.start_new_thread(playsound, ('Sound/smoking.mp3',))
            elif illegal == '使用手机':
                _thread.start_new_thread(playsound, ('Sound/use_phone.mp3',))
            elif illegal == '吃喝':
                _thread.start_new_thread(playsound, ('Sound/eat.mp3',))
            elif illegal == '未佩戴口罩':
                _thread.start_new_thread(playsound, ('Sound/no_mask.mp3',))
            location = '未知'
            illegal_path = './illegal_data/' + str(time.time()) + '.jpg'
            cv2.imwrite(illegal_path, packet['status_face'])
            cv2.imwrite('./data/' + pt, packet['status_face'])
            value = {'CARID': 0,
                     'CARIMAGE': QPixmap('./data/' + pt),
                     'CARCOLOR': str(now),
                     'FACE': license_number,
                     'LICENSEIMAGE': None,
                     'LICENSENUMBER': license_number,
                     'LOCATION': location,
                     'RULENAME': illegal}
            headers = [value['FACE'], value['CARCOLOR'], value['RULENAME'], illegal_path]
            with open('data.csv', 'a+', newline='')as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
            self.data.append(value)
            self.update_status_log(self.data)
            self.pre_time = now

        if len(faces_recorded) > 0:
            self.update_face_time = 180
            self.update_face_flag = 1
            self.faces_recorded = faces_recorded.copy()


        if self.update_face_time < 2 and self.update_face_flag == 1:
            for im, faceID in self.faces_recorded:
                license_number = 'Driver ' + str(self.show_face_id)
                illegal = '更换驾驶员\更换监控方式'
                location = '未知'
                cv2.imwrite('./data/'+pt, im)
                value = {'CARID': 0,
                         'CARIMAGE': QPixmap('./data/'+pt),
                         'CARCOLOR': str(now),
                         'FACE' : license_number,
                         'LICENSEIMAGE': None,
                         'LICENSENUMBER': license_number,
                         'LOCATION': location,
                         'RULENAME': illegal}
                self.data.append(value)
            self.Faceid = faceID
            self.Recodeface = im
            self.updateLog(self.data)
            self.update_face_flag = 0
        elif self.update_face_flag == 1:
            you = 0
            for im, faceID in self.faces_recorded:
                if faceID == self.reco_face_id:
                    self.update_face_time = 0
                    you = 1
            if you == 0:
                flag = 0
                features_cap = self.face_reco(frame)
                for key, value in self.face_data.items():
                    if features_cap is False:
                        flag = 1
                        self.show_face_id = 'Unknown_face'
                        break
                    compare = self.return_euclidean_distance(features_cap, value)
                    if compare:
                        self.show_face_id = int(key)
                        flag = 1
                        self.update_face_time = 0
                        self.reco_face_id = faceID
                        break
                if flag == 0 and features_cap is not False:  # 说明没有这张脸 添加
                    self.max_face_id += 1
                    self.show_face_id = self.max_face_id
                    self.face_data[str(self.max_face_id)] = features_cap
                    self.update_face_time = 0
                    self.reco_face_id = faceID
                self.update_face_time -= 1


        qimg0 = self.toQImage(packet['frame'])

        if self.func_status['facestatus']:
            face_bboxes = packet['face_bboxes']
            raw, face_status = self.processor.processor.faceDet.feedCap(
                raw, face_bboxes, face_size=self.face_size)
            self.update_graph_values(face_status)

            if not face_status['face'] is None:
                qimg2 = self.toQImage(face_status['face'], height=self.face_size)
                self.live_preview_keypoints.setPixmap(QPixmap.fromImage(qimg2))
                self.label_X.setText(
                    "头部3D姿态-X: " + "{:.2f}".format(face_status['X']))
                self.label_Y.setText(
                    "头部3D姿态-Y: " + "{:.2f}".format(face_status['Y']))
                self.label_Z.setText(
                    "头部3D姿态-Z: " + "{:.2f}".format(face_status['Z']))
            qimg1 = self.toQImage(raw)
            self.live_preview_2.setPixmap(QPixmap.fromImage(qimg1))

        self.live_preview.setPixmap(QPixmap.fromImage(qimg0))

    def face_reco(self, face_img):
        shape = None
        cv2.imwrite('test.jpg', face_img)
        dets = self.face_detector(face_img, 0)
        for k, d in enumerate(dets):
            shape = self.predictor(face_img, d)
        if shape is None:
            return False
        features_cap = self.facerec.compute_face_descriptor(face_img, shape)
        return features_cap

    def return_euclidean_distance(self, feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        feature_1 = np.around(feature_1, decimals=3)
        feature_2 = np.around(feature_2, decimals=3)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        if dist > 0.45:
            return False
        else:
            return True

