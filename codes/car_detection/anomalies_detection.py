import os, sys
import numpy as np
from moviepy.editor import VideoFileClip


car_seqs = {}
anomaly_candidates = []
SPEED_THRESHOLD = 0.005
MOTION_THRESHOLD = 2.2
CLOSE_POSTION = 30
STOPPING_RATIO = 0.5

RECORD_THRESHOLD = 60


def get_distance(a, b):
    return np.linalg.norm(a-b);


def process(filename):

    with open(filename, 'r') as infile:
        tmp = infile.read()
        tmp = tmp.splitlines()
        total_frames_num = int(tmp[-1].split(',')[0])

        for i in tmp:
            car = i.split(',')
            car = [float(i) for i in car]

            width = car[4] - car[2]
            height = car[5] - car[3]
            x_center = car[2]+width/2
            y_center = car[3]+height/2
            if car[1] not in car_seqs.keys():
                car_seqs[car[1]] = np.array([[car[0], x_center, y_center, 1]])

            else:
                dist = get_distance(car_seqs[car[1]][-1][1:3], np.array([x_center, y_center]))
                dist /= height * 10
                speed = dist / (car[0] - car_seqs[car[1]][-1][0])
                speed = 0 if np.abs(speed) < SPEED_THRESHOLD else np.abs(speed)
                car_seqs[car[1]] = np.vstack([car_seqs[car[1]], [car[0], x_center, y_center, speed]])

    # for key in car_seqs.keys():
    #     if key in [255427, 256312, 265548]:
    #         print(key)
    #         print(car_seqs[key])
    return total_frames_num


def merge_anomalies(anomalies):
    key_list = anomalies.keys()
    remove_list = []
    for i in range(0, len(key_list)-1):
        for j in range(i+1, len(key_list)):
            key1 = key_list[i]
            key2 = key_list[j]

            anomaly1 = anomalies[key1]
            anomaly1 = anomaly1[anomaly1[:, 3] == 0]
            anomaly2 = anomalies[key2]
            anomaly2 = anomaly2[anomaly2[:, 3] == 0]

            x1 = np.mean(anomaly1[:, 1])
            y1 = np.mean(anomaly1[:, 2])
            x2= np.mean(anomaly2[:, 1])
            y2 = np.mean(anomaly2[:, 2])

            dist = get_distance(np.array([x1, y1]), np.array([x2, y2]))
            if dist < CLOSE_POSTION:
                remove_list.append(key2)

    for key in remove_list:
        anomalies.pop(key, None)

    return anomalies


def detect_anomaly(car_seqs, total_frames_num):
    print(len(car_seqs.keys()))
    anomalies = {}
    for key in car_seqs.keys():
        car = car_seqs[key]
        recorder_num = car.shape[0]

        stopping_duration = 300

        x_moving = car[1:-1, 1] - car[0:-2, 1]
        x_stopping = x_moving[np.abs(x_moving) < MOTION_THRESHOLD]
        motion_ratio = x_stopping.shape[0]/float(recorder_num)
        stalling_car = car[car[:, 3] == 0]
        speed_ratio = stalling_car.shape[0]/float(recorder_num)
        k = STOPPING_RATIO
        pre_points = car[1:8, :] if recorder_num > 8 else car
        if motion_ratio > k and speed_ratio > k and car.shape[0] > RECORD_THRESHOLD and \
                pre_points[pre_points[:, 3] == 0].shape[0] <= 0:
            anomalies[key] = car
        # a[np.where((a[:, 0] == 0) * (a[:, 1] == 1))]
        # if key == 4374:
        #     print(motion_ratio)
        #     print(speed_ratio)
        #     print(car.shape[0])


    # print(len(anomalies.keys()))
    for key in anomalies.keys():
        print(key)
        print(anomalies[key])
        anomaly = anomalies[key]
        zero_case = anomaly[anomaly[:, 3] == 0][0, 0]
        print('anomaly time (percent): ')
        print(zero_case  /total_frames_num)

    anomalies = merge_anomalies(anomalies)

    final_results = {}
    for key in anomalies.keys():
        anomaly = anomalies[key]
        zero_case = anomaly[anomaly[:, 3] == 0][0, 0]
        final_results[key] = zero_case / total_frames_num

    return final_results

    # with open('temp_output.txt', 'w') as outfile:
    #     outfile.write(car_seqs)


if __name__ == '__main__':
    video_folder = 'output_train'

    files = [file for file in os.listdir(video_folder) if file.endswith('txt')]
    anomaly_candidates = []
    anomaly_str = ''
    for file in files:
        file = '58.mp4.txt'
        video = VideoFileClip(video_folder +'/'+file[:-4])
        print('process' + file)
        total_frames_num = process(os.path.join(video_folder, file))
        r = detect_anomaly(car_seqs, total_frames_num)

        for key in r.keys():
            anomaly_candidates.append([file.split('.')[0], key, r[key]])
            anomaly_str += str(file.split('.')[0]) + ',' + str(r[key] * video.duration) + '\n'
        break

    with open('anomlies_output.txt', 'w') as fi:
        fi.write(anomaly_str)
