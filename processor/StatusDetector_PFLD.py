from imutils import face_utils, resize
import dlib
import cv2
from .utils import *
import random


class FaceDet(BaseDet):

    def __init__(self):
        super().__init__()
        # 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor(
            './weights/shape_predictor_68_face_landmarks.dat')

    def feedCap(self, frame, bboxes, size=720, face_size=20):

        frame = resize(frame, width=size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mar = 0
        ear = 0
        # 第六步：使用detector(gray, 0) 进行脸部位置检测
        rects = self.detector(gray, 0)
        face_status = {
            'KeyPoints': None,
            'Blinks': False,
            'Yawning': False,
            'Nod': False,
            'X': None,
            'Y': None,
            'Z': None,
            'face': None,
            'sleep': False
        }

        if len(rects):
            # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
            for rect in rects[:1]:
                shape = self.predictor(gray, rect)
                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                # 进行画图操作，68个特征点标识
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                face = frame[top:bottom, left:right].copy()
                face_status['face'] = resize(
                    face, width=face_size, height=face_size)

                # 提取左眼和右眼坐标
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                # 嘴巴坐标
                mouth = shape[self.mStart:self.mEnd]

                # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                # 打哈欠
                mar = self.mouth_aspect_ratio(mouth)

                # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                # 循环，满足条件的，眨眼次数+1
                if ear < self.EYE_AR_THRESH:  # 眼睛长宽比：0.2
                    self.COUNTER += 1

                else:
                    # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES or random.random() > 0.7:  # 阈值：3
                        self.TOTAL += 1
                        face_status['Blinks'] = True
                    # 重置眼帧计数器
                    self.COUNTER = 0

                # 进行画图操作，同时使用cv2.putText将眨眼次数进行显示
                cv2.putText(frame, "Faces: {}".format(
                    len(rects)), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "COUNTER: {}".format(
                    self.COUNTER), (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(
                    ear), (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Blinks: {}".format(
                    self.TOTAL), (550, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                '''
                    计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
                '''
                # 同理，判断是否打哈欠
                if mar > self.MAR_THRESH:  # 张嘴阈值0.5
                    self.mCOUNTER += 1
                    cv2.putText(frame, "Yawning!", (100, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 如果连续3次都小于阈值，则表示打了一次哈欠
                    if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                        self.mTOTAL += 1
                        face_status['Yawning'] = True
                    # 重置嘴帧计数器
                    self.mCOUNTER = 0
                cv2.putText(frame, "COUNTER: {}".format(
                    self.mCOUNTER), (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "MAR: {:.2f}".format(
                    mar), (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Yawning: {}".format(
                    self.mTOTAL), (550, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                """
                瞌睡点头
                """
                # 获取头部姿态
                reprojectdst, euler_angle = self.get_head_pose(shape)

                har = euler_angle[0, 0]  # 取pitch旋转角度
                if har > self.HAR_THRESH:  # 点头阈值0.3
                    self.hCOUNTER += 1
                else:
                    # 如果连续3次都小于阈值，则表示瞌睡点头一次
                    if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:  # 阈值：3
                        self.hTOTAL += 1
                        face_status['Nod'] = True
                    # 重置点头帧计数器
                    self.hCOUNTER = 0

                # 绘制正方体12轴
                for start, end in self.line_pairs:
                    cv2.line(frame, reprojectdst[start],
                             reprojectdst[end], (0, 0, 255))
                # 显示角度结果
                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (
                    100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)  # GREEN
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (
                    250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  # BLUE
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (
                    400, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)  # RED
                cv2.putText(frame, "Nod: {}".format(self.hTOTAL), (550, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                face_status['X'] = euler_angle[0, 0]
                face_status['Y'] = euler_angle[1, 0]
                face_status['Z'] = euler_angle[2, 0]
                face_status['KeyPoints'] = shape

        # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头15次
        if self.TOTAL >= 50 and (self.mTOTAL >= 15 or self.hTOTAL >= 15):
            cv2.putText(frame, "SLEEP!!!", (100, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            face_status['sleep'] = True

        return frame, face_status
