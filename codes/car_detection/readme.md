[TOC]

### Prerequisite 

#### install Mask RCNN

```shell
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
pip3 install -r requirements.txt
python3 setup.py install
```

#### add the lost files

1. Add videos to aic19-track3-train-data/

2. Add our model mask_rcnn_car_0030.h5 to model/

3. ```
   git clone '19spr_ngu....git'
   
   cd 19spr_nguyen_chu_shumaker/codes/car_detection/Mask_RCNN
   pip3 install -r requirements.txt
   python3 setup.py install
   ```

   

### Train the custom model

```
# please ensure you are under the car_detection directory
python3 Mask_RCNN/car_detection_train.py train --dataset=dataset/ --weights=model/mask_rcnn_car_0030.h5 --logs=model/

# train from a pre-trained model COCO
# python3 Mask_RCNN/car_detection_train.py train --dataset=dataset/ --weights=model/mask_rcnn_coco.h5 --logs=model/
python3 infer_car_in_video.py
python3 sort_track.py
# output results into output_train/ or output_test/ in terms of your setup
# make sure there is at least one sequence file under the output_train/
python3 anomalies_detection.py
```


### Train your own model

```shell
# please ensure you are under the car_detection directory
python3 Mask_RCNN/car_detection_train.py train --dataset=dataset/ --weights=model/mask_rcnn_car_0030.h5 --logs=model/

# train from a pre-trained model COCO
# python3 Mask_RCNN/car_detection_train.py train --dataset=dataset/ --weights=model/mask_rcnn_coco.h5 --logs=model/
```

### Detect cars in the video

```shell
python3 infer_car_in_video.py
```

### Track cars in the video

```shell
python3 sort_track.py
# output results into output_train/ or output_test/ in terms of your setup
```

### Find anomalies

```shell
# make sure there is at least one sequence file under the output_train/
python3 anomalies_detection.py
```

