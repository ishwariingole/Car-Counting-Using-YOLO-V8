# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time
import logging

class CentroidTracker():
    def __init__(self, axis, point, maxDisappeared=3, maxDistance=50, minDistance=5):

        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.startTime = OrderedDict()

        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

        self.minDistance = minDistance
        self.startCentroid = OrderedDict()
        self.count = 0

        self.axis = axis
        self.point = point

        if self.axis == 'y':
            self.index = 1
        elif self.axis == 'x':
            self.index = 0

        self.positive_direction_count = 0
        self.negative_direction_count = 0
        self.ignore_count = 0

        self.vsq_logger = logging.getLogger('CT_Event')
        self.vsq_logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')

        VSQ_LOG_PATH = f'centroid_.log'

        self.log_handler = logging.FileHandler(VSQ_LOG_PATH, mode='a')
        self.log_handler.setFormatter(formatter)
        self.vsq_logger.addHandler(self.log_handler)

        self.vsq_logger.info(f'Centroid Started')
        self.vsq_logger.info(
            f'objectID \t startCentroid \t lastCentroid \t axis_dist \t dist. \t +ve dir \t -ve dir \t NA count \t Status')

    def register(self, centroid):

        self.objects[self.nextObjectID] = centroid
        self.startCentroid[self.nextObjectID] = centroid
        self.count += 1
        self.disappeared[self.nextObjectID] = 0
        self.startTime[self.nextObjectID] = int(time.time())
        self.nextObjectID += 1

    def deregister(self, objectID):

        axis_dist = int(self.objects[objectID][self.index] - self.startCentroid[objectID][self.index])

        if self.objects[objectID][self.index] < self.point and self.startCentroid[objectID][self.index] < self.point:
            status = 'ignore'
            self.ignore_count += 1
        elif self.objects[objectID][self.index] > self.point and self.startCentroid[objectID][self.index] > self.point:
            status = 'ignore'
            self.ignore_count += 1
        else:
            status = 'check'

        dist_ = round(np.linalg.norm(np.array(self.startCentroid[objectID]) - np.array(self.objects[objectID])), 2)
        # print('distance = {}'.format(dist_))
        if dist_ < self.minDistance:
            self.count -= 1
            if status == 'check':
                self.ignore_count += 1
            status = 'ignore'
            pass
        # print('LESS THAN MIN DISTANCE')
        elif status == 'check':
            if axis_dist >= 0:
                self.positive_direction_count += 1
                status = 'accepted'
            elif axis_dist < 0:
                self.negative_direction_count += 1
                status = 'accepted'

        self.vsq_logger.info(
            f'{objectID} \t {self.startCentroid[objectID]} \t {self.objects[objectID]} \t {axis_dist} \t {dist_} \t {self.positive_direction_count} \t {self.negative_direction_count} \t {self.ignore_count} \t {status}')

        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.startTime[objectID]
        del self.startCentroid[objectID]

    def update(self, rects):

        if len(rects) == 0:
            try:
                for objectID in self.disappeared.keys():
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            except:
                # camera_vsq.vsq_logger.info('Exception')
                print('..................***********************')

            return self.objects, self.startTime

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    self.register(inputCentroids[col])
                    usedRows.add(row)
                    usedCols.add(col)
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects, self.startTime
# return self.objects