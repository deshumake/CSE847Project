import os
import sys
import skimage.io
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    video_folder = 'aic19-track3-train-data'
    output_folder = 'speeding_train'
    zoom = 8    # speed up to 3 times
    for file in os.listdir(video_folder):
        if os.path.splitext(file)[-1] == '.mp4':
            video = VideoFileClip(os.path.join(video_folder, file))

            video = VideoFileClip('aic19-track3-train-data/2.mp4')
            # This function can reduce frames in the video
            # We shrink each video's length to 1/3 with speeding up
            clip = video.fl_time(lambda t: zoom*t).set_duration(video.duration/zoom - 1)
            clip.write_videofile(os.path.join(output_folder, file), audio=False)


