import os
import sys
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
    class_names = ['bg', ' ']

    return model, class_names


def detect_cars(img_name):
    # img = 'dataset/train/1.jpg'
    image = skimage.io.imread(img_name)
    results = model.detect([image], verbose=0)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                  class_names, colors= [(0.2, 0.2, 0.95)]*100, figsize=(12, 12))

