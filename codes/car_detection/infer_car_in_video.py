import os
import sys
import skimage.io
import matplotlib.pyplot as plt
import argparse

sys.path.append("cnn")
sys.path.append("cnn/mrcnn")

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


def convert2output(header, box):
    output = ''
    for i in box:
        output += str(header) + ',' + '-1,';
        for j in i:
            output += str(j) + ','
        output += '-1,-1,-1\n'
    return output

class_names = ['bg', ' ']

results_sequence = ''
header = 0

def process_video(input_img):
    global results_sequence
    global header
    global model
    header += 1
    # img = cv2.resize(input_img, (1024, 1024))
    # img = image.img_to_array(img)
    results = model.detect([input_img], verbose=0)
    r = results[0]
    final_img = visualize_car_detection.display_instances2(input_img, r['rois'], r['masks'], r['class_ids'],
                                                           class_names, r['scores'])

    plt.imshow(final_img)
    plt.show()
    results_sequence += convert2output(header, r['rois'])
    return final_img

def parseArgs():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('integer', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    model = configure()
    output = 'output.mp4'
    # get the duration of output video
    params = parseArgs()
    if params:
        duration = params.integer
    else:
        duration = .5
    clip1 = VideoFileClip("aic19-track3-train-data/72.mp4")
    # this function can reduce frames in the video
    # in the demo, we just use 5s duration of the video and two fold faster
    newclip = clip1.fl_time(lambda t: 2*t).set_duration(duration)
    clip = newclip.fl_image(lambda image: process_video(image))
    clip.write_videofile(output, audio=False)

    # output the box positions (SORT: tracks these boxes)
    #print(results_sequence)
    with open("carBoxesOutput.txt", "w") as fp:
        fp.write(results_sequence)


