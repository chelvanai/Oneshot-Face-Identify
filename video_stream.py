import os
import shutil

import cv2
import imutils
import numpy as np

from face_compare.face_compare import FaceMatch
from face_detect.face_detector import FaceDetector

from filterpy.kalman import KalmanFilter
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment


@jit
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class VideoCamera:
    def __init__(self, video=None, url=None):
        global fd, fc
        self.max_age = 15
        self.min_hits = 3
        self.trackers = []
        self.frame = 0
        self.ipcam = False

        if url is None:
            self.cap = cv2.VideoCapture(video)

        if video is None:
            self.ipcam = True
            self.url = url

        self.cap = cv2.VideoCapture(video)
        fd = FaceDetector('face_detect/weight/model.pb')
        fc = FaceMatch('face_compare/weight/vgg_face_weights.h5')

        if os.path.isdir('./past_ppl'):
            shutil.rmtree('./past_ppl')
            os.mkdir('./past_ppl')
        else:
            os.mkdir('./past_ppl')

    def __del__(self):
        print("Deleted")

    def find(self, img, box, trk):
        ymin, xmin, ymax, xmax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped_img = img[ymin:ymax, xmin:xmax]
        cv2.imwrite('./temporaryImg.jpg', cropped_img)

        past_ppl = './past_ppl'
        folders = os.listdir(past_ppl)

        for folder in folders:
            files = os.listdir(past_ppl + '/' + folder)
            same = 0
            diff = 0
            numOfFiles = len(files)
            for f in range(numOfFiles):
                if f % 10 is not 0:
                    continue
                ret = fc.comapre('./temporaryImg.jpg', './past_ppl/' + folder + '/' + str(f + 1) + '.jpg')

                if ret:
                    same += 1
                else:
                    diff += 1

            p = 100 * float(same) / float(same + diff)
            if p > 70:
                person_no = len(files) + 1
                cv2.imwrite(past_ppl + '/' + folder + '/' + str(person_no) + '.jpg', cropped_img)
                trk.id = str(folder)
                return

        l = len(folders)
        os.makedirs(past_ppl + '/' + str(l))
        cv2.imwrite(past_ppl + '/' + str(l) + '/1.jpg', cropped_img)
        trk.id = str(l)
        return

    def get_frame(self):
        self.frame += 1

        if self.ipcam:
            self.cap = cv2.VideoCapture(self.url)

        r, img = self.cap.read()

        if r is False:
            return None

        img = imutils.resize(img, width=1200)
        boxes, scores = fd(img)

        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(boxes, trks)

        # tracked
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(boxes[d, :][0])
                ymin, xmin, ymax, xmax = boxes[d, :][0]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, str(trk.id), (int(xmin) - 10, int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)

                cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                person_no = len(os.listdir('past_ppl' + '/' + str(trk.id))) + 1
                cv2.imwrite('past_ppl' + '/' + str(trk.id) + '/' + str(person_no) + '.jpg', cropped_img)

        # new trackers
        for i in unmatched_dets:
            ymin, xmin, ymax, xmax = boxes[i, :]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            trk = KalmanBoxTracker(boxes[i, :])
            self.trackers.append(trk)

            self.find(img, boxes[i, :], trk)

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
