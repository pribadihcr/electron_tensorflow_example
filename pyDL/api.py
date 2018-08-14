from __future__ import print_function
import sys
import zerorpc
from yolo import YOLO
import cv2
from PIL import Image
import time
import os
import numpy as np
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

class DetectApi(object):
    def detect_yolo(self, input):
        try:
            this_dir = os.path.dirname(__file__)
            yolo = YOLO()

            max_cosine_distance = 0.3
            nn_budget = None
            nms_max_overlap = 1.0

            # deep_sort
            model_filename = os.path.join(this_dir, 'models/mars-small128.pb')
            encoder = gdet.create_box_encoder(model_filename, batch_size=1)

            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            tracker = Tracker(metric)
            writeVideo_flag = True

            video_capture = cv2.VideoCapture(input)
            if writeVideo_flag:
                # Define the codec and create VideoWriter object
                w = int(video_capture.get(3))
                h = int(video_capture.get(4))
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(os.path.join(this_dir, 'data/output.avi'), fourcc, 15, (w, h))
                list_file = open(os.path.join(this_dir, 'data/detection.txt'), 'w')
                frame_index = -1

            fps = 0.0
            n = 0
            skip_frame = 5
            while True:
                ret, frame = video_capture.read()  # frame shape 640*480*3
                if ret != True:
                    break

                if int(skip_frame) != n:
                    n += 1
                    continue
                n = 0

                t1 = time.time()

                image = Image.fromarray(frame)
                boxs, classes = yolo.detect_image(image)

                for idb, box in enumerate(boxs):
                    # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])),
                    #               (255, 255, 255), 2)
                    cv2.putText(frame, str(classes[idb]), (int(box[0]), int(box[1])), 0, 5e-3 * 100, (0, 255, 0), 2)

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
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

                for idb, det in enumerate(detections):
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)


                # cv2.imshow('gallery', frame)

                if writeVideo_flag:
                    # save a frame
                    out.write(frame)
                    frame_index = frame_index + 1
                    list_file.write(str(frame_index) + ' ')
                    if len(boxs) != 0:
                        for i in range(0, len(boxs)):
                            list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(
                                boxs[i][3]) + ';')
                    list_file.write('\n')

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %f" % (fps))

                # Press Q to stop!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            if writeVideo_flag:
                out.release()
                list_file.close()
            cv2.destroyAllWindows()
            msg = "process finished!!!"
        except Exception as e:
            print(e)
            msg = "process error!!!"
        return msg

    def echo(self, input):
        """echo any text"""
        return input
   

def parse_port():
    port = 4242
    try:
        port = int(sys.argv[1])
    except Exception as e:
        pass
    return '{}'.format(port)

def main():
    addr = 'tcp://127.0.0.1:' + parse_port()
    s = zerorpc.Server(DetectApi())
    s.bind(addr)
    print('start running on {}'.format(addr))
    s.run()

if __name__ == '__main__':
    main()

