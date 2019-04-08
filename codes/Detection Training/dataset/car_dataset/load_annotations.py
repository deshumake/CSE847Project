   
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


dataMap = {}
file = 'annotations.json'
dataList = []
with open(file, 'r') as inFile:
    strBuffer = inFile.read()
    rawStrList = json.loads(strBuffer)

    for key, value in rawStrList.items():
        dataMap[value['filename']] = value['regions']
    # rawStrList = strBuffer.splitlines()
    # if len(rawStrList) >= 2:
    #     rawStrList = rawStrList[1:]
    #     for data in rawStrList:
    #         item_anno = data.split(',')
    #         img_id = item_anno[0]
    #         shape = item_anno[5]
    #         dataList.append(shape)
    #         if int(item_anno[3]) - int(item_anno[4]) <= 1:
    #             dataMap[img_id] = dataList
    #             dataList = []

print(len(dataMap))

temp = dataMap['1.jpg']

image = skimage.io.imread('train/1.jpg')
height, width = image.shape[:2]
print(height, width)
mask = np.zeros((height, width, 3), dtype=np.uint8)
image = image[:][:][1]
for i, item in enumerate(temp):
    # Get indexes of pixels inside the polygon and set them to 1
    anno = item['shape_attributes']
    image
    if anno['name'] == 'rect':
        start = (anno['y'], anno['x'])
        extent = (anno['height'], anno['width'])
        rr, cc = skimage.draw.rectangle(start, extent=extent)

    print(rr)
    print(cc)
    plt.plot(image)
    plt.imshow(image)
    break


# Get indexes of pixels inside the polygon and set them to 1
# anno = temp['shape_attributes']
# if anno['name'] == 'rect':
#     start = (anno['x'], anno['y'])
#     extent = (int(anno['width']), int(anno['height']))
#     print (type(start[0]))

# elif anno['name'] == 'polygon':
#     x = np.asarray(anno['all_points_x'])
#     y= np.asarray(anno['all_points_y'])

#     print (x)
# # print(dataMap['1.jpg'])

