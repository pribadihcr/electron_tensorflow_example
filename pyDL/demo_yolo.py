#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Person Re-ID')
    parser.add_argument('--gpu_id', dest='gpu_id', help='using gpu or not',
                        default='0')
    parser.add_argument('--input_video', dest='input_video', help='input_video',
                        default='/media/rudy/C0E28D29E28D252E/2017_taroko/终点前/YDXJ0101.mp4')
    parser.add_argument('--skip_frame', dest='skip_frame', help='skip_frame for sp',
                        default=5)

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return args

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = './models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    writeVideo_flag = True

    video_capture = cv2.VideoCapture(args.input_video)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('/media/rudy/C0E28D29E28D252E/2017_taroko/终点前/video_YDXJ0101.avi', fourcc, 15, (w, h))
        list_file = open('/media/rudy/C0E28D29E28D252E/2017_taroko/终点前/detection_YDXJ0101.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    n = 0

    id_frame = 0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        if int(args.skip_frame) != n:
            n += 1
            continue
        n = 0

        t1 = time.time()

        image = Image.fromarray(frame)
        boxs, classes = yolo.detect_image(image)

        for idb, box in enumerate(boxs):
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0])+int(box[2]), int(box[1])+int(box[3])),(255,255,255), 2)
            cv2.putText(frame, str(classes[idb]),(int(box[0]), int(box[1])),0, 5e-3 * 100, (0,255,0),2)

        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for idb, det in enumerate(detections):
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            # cv2.putText(frame, str(classes[idb]), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 100, (0, 255, 0), 2)


        # cv2.imshow('gallery', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ';')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        id_frame +=1
        print("idx_frame-%d, fps= %f"%(id_frame, fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
