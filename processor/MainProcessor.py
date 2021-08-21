# Copyright © 2021, Yuzu Liu. All Rights Reserved.

# Copyright Notice
# Yuzu Liu copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yuzu Liu.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yuzu Liu assumes no responsibility for any errors contained herein.

import imutils


class VideoPlayer(object):

    def __init__(self):
        self.firstFrame = None

    def feedCap(self, frame):

        self.frame = imutils.resize(frame, height=300)

        pack = {'frame': frame, 'faces': [], 'face_bboxes': []}

        return pack


class MainProcessor(object):

    def __init__(self, model_type='openvino', tracker_type='deep_sort'):
        if model_type == 'openvino':
            from .AIDetector_openvino import Detector as FaceTracker
        else:
            print('-[INFO] Using default model.')
            from .AIDetector_pytorch import Detector as FaceTracker
        self.processor = FaceTracker(tracker_type)
        # self.processor = VideoPlayer()
        self.face_id = 0

    def getProcessedImage(self, frame=None, func_status=None):
        dicti = self.processor.feedCap(frame, func_status)
        return dicti
