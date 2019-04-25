import os, sys
import numpy as np
from moviepy.editor import VideoFileClip


anomaly_candidates = []
SPEED_THRESHOLD = 0.008
MOTION_THRESHOLD = 2.2
CLOSE_POSTION = 180
SPEED_RATIO = 0.5
POSITION_RATIO = 0.7
RECORD_THRESHOLD = 80


def get_distance(a, b):
    return np.linalg.norm(a-b);


def process(filename):

    car_seqs ={}

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
    return total_frames_num, car_seqs


def merge_anomalies(anomalies):
    key_list = list(anomalies.keys())
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
            if dist < CLOSE_POSTION :
                remove_list.append(key2)

    for key in remove_list:
        anomalies.pop(key, None)

    return anomalies


def detect_anomaly(car_seqs, total_frames_num):
    # print(len(car_seqs.keys()))
    anomalies = {}

    for key in car_seqs.keys():
        car = car_seqs[key]
        recorder_num = car.shape[0]

        # if key == 221039:
        #     for i in car:
        #         print(i)

        x_moving = car[1:-1, 1] - car[0:-2, 1]
        x_stopping = x_moving[np.abs(x_moving) < MOTION_THRESHOLD]
        motion_ratio = x_stopping.shape[0]/float(recorder_num)
        stalling_car = car[car[:, 3] == 0]
        speed_ratio = stalling_car.shape[0]/float(recorder_num)
        if (car[-1, 0] - car[0, 0]) / float(total_frames_num) > 0.9:
            continue

        x_center = np.mean(car[:, 1])
        pre_points = car[1:4, :] if recorder_num > 4 else car
        if motion_ratio > POSITION_RATIO and speed_ratio > SPEED_RATIO and car.shape[0] > RECORD_THRESHOLD \
                and np.abs(x_center - 400) < 300:
            print(key)
            anomalies[key] = car
        # a[np.where((a[:, 0] == 0) * (a[:, 1] == 1))]
        # if key == 4374:
        #     print(motion_ratio)
        #     print(speed_ratio)
        #     print(car.shape[0])

    anomalies = merge_anomalies(anomalies)

    final_results = {}
    print('anomalies: ', len(anomalies.keys()))
    for key in anomalies.keys():
        anomaly = anomalies[key]
        zero_case = anomaly[anomaly[:, 3] == 0][0, 0]
        final_results[key] = zero_case / total_frames_num
        print(key)
        print(anomalies[key])
        print('anomaly time (percent): ')
        print(zero_case / total_frames_num)

    return final_results


if __name__ == '__main__':
    video_folder = 'output_train'
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    files = [int(file.split('.')[0]) for file in os.listdir(video_folder) if file.endswith('txt')]
    files.sort()
    anomaly_candidates = []
    anomaly_candidates_str = ''
    anomaly_str = ''
    for file in files:
        file = str(file) + '.mp4.txt'
        video = VideoFileClip(video_folder +'/'+file[:-4])
        print('process ' + file)
        total_frames_num, car_seqs = process(os.path.join(video_folder, file))

        r = detect_anomaly(car_seqs, total_frames_num)

        for key in r.keys():
            anomaly_candidates.append([file.split('.')[0], key, r[key]])
            anomaly_candidates_str += str(file.split('.')[0]) + ', ' + str(key) + ', ' + str(r[key] * video.duration) + '\n'
            anomaly_str += str(file.split('.')[0]) + ' ' + str(r[key] * video.duration*10) + '\n'
        # break

    with open('output_anomalies/anomalies_' + video_folder + '.txt', 'w') as fi:
        fi.write(anomaly_str)

    with open('output_anomalies/anomalies_' + video_folder + '_with_trackid.txt', 'w') as fi:
        fi.write(anomaly_candidates_str)
