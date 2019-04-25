#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')


import skimage.io
sys.path.append("Mask_RCNN")
sys.path.append("Mask_RCNN/mrcnn")

from Mask_RCNN import car_detection_train
import model as modellib
import visualize
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cv2
from keras.preprocessing import image
from moviepy.editor import VideoFileClip
import visualize_car_detection


def configure():
    # Root directory of the project
    ROOT_DIR = "./"
    MODEL_DIR = 'model/'
    MY_MODEL_PATH = MODEL_DIR + 'mask_rcnn_car_0030.h5'
    IMAGE_DIR = os.path.join('Mask_RCNN', "images")

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    class InferenceConfig(car_detection_train.BalloonConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.6

    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MY_MODEL_PATH, by_name=True)
    # class_names = ['bg', ' ']

    return model


def detect_cars(img_name):
    # img = 'dataset/train/1.jpg'
    image = skimage.io.imread(img_name)
    results = model.detect([image], verbose=0)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                  class_names, colors= [(0.2, 0.2, 0.95)]*100, figsize=(12, 12))


def convert2output(header, ids, boxes):
    output = ''
    for (box, id) in zip(boxes, ids):
        output += str(header) + ','
        output += str(id)
        for j in box:
            output += ',' + str(j)
        output += '\n'
    return output

class_names = ['bg', ' ']

results_sequence = ''
header = 0

# deep_sort
max_cosine_distance = 0.3
nn_budget = None
# model_filename = 'model_data/mars-small128.pb'
# encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

def process_video(input_img):

    nms_max_overlap = 1.0

    global results_sequence
    global header
    global model
    global tracker
    global encoder

    header += 1
    # img = cv2.resize(input_img, (1024, 1024))
    # img = image.img_to_array(img)
    results = model.detect([input_img], verbose=0)
    r = results[0]
    boxes = r['rois']
    scores = r['scores']

    # features = encoder(input_img, boxes)

    # score to 1.0 here).
    # y1, x1, y2, x2 = boxes[i]
    # x, y, width, height
    feature = np.asarray([-1, -1, -1])
    detections = [Detection(np.asarray([bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]]), score, feature) for bbox, score in zip(boxes, scores)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    # detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)
    tracks = []
    track_ids = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        tracks.append(bbox)
        track_ids.append(track.track_id)
        cv2.rectangle(input_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        cv2.putText(input_img, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 0.3, (255, 255, 255), 1)

    # for det in detections:
    #     bbox = det.to_tlbr()
    #     cv2.rectangle(input_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    results_sequence += convert2output(header, track_ids, tracks)
    return input_img


if __name__ == '__main__':
    model = configure()
    video_folder = 'aic19-track3-train-data/'
    output_folder = 'output_train'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    zoom = 10   # speed up to 10 times

    # anomaly_videos = [2,9,11,14,33,35,49,58,63,66,72,73,74,83,91,93,95,97]
    for file in os.listdir(video_folder):
        filename = os.path.splitext(file)
        results_sequence = ''
        header = 0
        if filename[-1] == '.mp4' :
            video = VideoFileClip(video_folder+ file)
            # This function can reduce frames in the video
            # We shrink each video's length to 1/3 with speeding up
            video = video.fl_time(lambda t: zoom*t).set_duration(video.duration/zoom-0.5)
            new_video = video.fl_image(lambda img: process_video(img))
            new_video.write_videofile(os.path.join(output_folder, file), audio=False)
            # output the box positions (SORT: tracks these boxes)
            with open(os.path.join(output_folder, os.path.basename(file)+'.txt'), 'w') as seq_file:
                seq_file.write(results_sequence)
