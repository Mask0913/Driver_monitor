import dlib
import cv2


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)
        # cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # tf = max(tl - 1, 1)  # font thickness
        # t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
        #             [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(target_detector, image):

    raw = image.copy()

    if target_detector.frameCounter > 2e+4:
        target_detector.frameCounter = 0

    faceIDtoDelete = []

    for faceID in target_detector.faceTracker.keys():
        trackingQuality = target_detector.faceTracker[faceID].update(image)

        if trackingQuality < 8:
            faceIDtoDelete.append(faceID)

    for faceID in faceIDtoDelete:
        target_detector.faceTracker.pop(faceID, None)
        target_detector.faceLocation1.pop(faceID, None)
        target_detector.faceLocation2.pop(faceID, None)
        target_detector.faceClasses.pop(faceID, None)

    new_faces = []

    if not (target_detector.frameCounter % target_detector.stride):

        _, bboxes = target_detector.detect(image)

        for (x1, y1, x2, y2, cls_id, _) in bboxes:
            x = int(x1)
            y = int(y1)
            w = int(x2-x1)
            h = int(y2-y1)

            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            matchCarID = None

            for faceID in target_detector.faceTracker.keys():
                trackedPosition = target_detector.faceTracker[faceID].get_position(
                )

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                if t_x <= x_bar <= (t_x + t_w) and t_y <= y_bar <= (t_y + t_h):
                    if x <= t_x_bar <= (x + w) and y <= t_y_bar <= (y + h):
                        matchCarID = faceID

            if matchCarID is None:
                # 新出现的目标
                tracker = dlib.correlation_tracker()
                tracker.start_track(
                    image, dlib.rectangle(x, y, x + w, y + h))

                target_detector.faceTracker[target_detector.currentCarID] = tracker
                target_detector.faceLocation1[target_detector.currentCarID] = [
                    x, y, w, h]

                matchCarID = target_detector.currentCarID
                target_detector.currentCarID = target_detector.currentCarID + 1

                if cls_id == 'face':
                    pad_x = int(w * 0.15)
                    pad_y = int(h * 0.15)
                    if x > pad_x:
                        x = x-pad_x
                    if y > pad_y:
                        y = y-pad_y
                    face = raw[y:y+h+pad_y*2, x:x+w+pad_x*2]
                    new_faces.append((face, matchCarID))

                target_detector.faceClasses[matchCarID] = cls_id

    bboxes2draw = []
    face_bboxes = []
    for faceID in target_detector.faceTracker.keys():
        trackedPosition = target_detector.faceTracker[faceID].get_position()

        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())
        cls_id = target_detector.faceClasses[faceID]
        target_detector.faceLocation2[faceID] = [t_x, t_y, t_w, t_h]
        bboxes2draw.append(
            (t_x, t_y, t_x+t_w, t_y+t_h, cls_id, faceID)
        )
        if cls_id == 'face':
            face_bboxes.append((t_x, t_y, t_x+t_w, t_y+t_h))

    image = plot_bboxes(image, bboxes2draw)
    print(bboxes2draw)
    return image, new_faces, face_bboxes
