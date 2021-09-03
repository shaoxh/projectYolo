# coding=utf-8
# This is a sample Python script.

import os
import threading

import cv2
import numpy as np
import pandas as pd


def mask_image(image, offset):
    # 创建矩形区域，填充白色255
    rectangle = np.zeros(image.shape[0:2], dtype="uint8")
    cv2.rectangle(rectangle, (25 + offset, 25 + offset), (175 + offset, 175 + offset), 255, -1)
    # cv2.imshow("Rectangle", rectangle)

    # 非运算，非0为1, 非1为0
    bitwiseNot = cv2.bitwise_not(rectangle)
    bitwiseAnd_img = cv2.bitwise_and(image, image, mask=bitwiseNot)
    return bitwiseAnd_img


def find_by_time_interval(file_name: str, time_1: float, time_2: float):
    time = float(file_name[0:-4])
    return time_1 <= time <= time_2


def find_images_by_time(base_img_dir: str, time_interval: list):
    file_name = list()
    list_file = os.listdir(base_img_dir)
    for file in list_file:
        for time in time_interval:
            time_1 = time[0]
            time_2 = time[1]
            if find_by_time_interval(file, time_1, time_2):
                file_name.append(file)
                break
            else:
                continue
    return file_name


def initial_yolo_model(config_path: str, weight_path: str):
    net = cv2.dnn_DetectionModel(config_path, weight_path)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    return net


def find_mask(image: np.ndarray, net: cv2.dnn_DetectionModel):
    classes, confidences, boxes = net.detect(image, confThreshold=0.1, nmsThreshold=0.4)
    return classes, confidences, boxes


def filter_mask_class(yolo_result: tuple, desired_class: list, confidence_threshold):
    classes = yolo_result[0]
    confidences = yolo_result[1]
    boxes = yolo_result[2]
    filtered_classes = list()
    filtered_confidences = list()
    filtered_boxes = list()
    for class_id, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        if class_id in desired_class and confidence > confidence_threshold:
            filtered_classes.append(class_id)
            filtered_confidences.append(confidence)
            filtered_boxes.append(box)
    return np.array(filtered_classes), np.array(filtered_confidences), np.array(filtered_boxes)


def put_mask(img: np.ndarray, boxes: np.ndarray):
    for box in boxes:
        cv2.rectangle(img, box, [0, 0, 0], -1)
        # bitwiseNot = cv2.bitwise_not(rectangle)
        # bit_and = cv2.bitwise_and(img, img, mask=bitwiseNot)
        # img = bit_and
    return img


def read_yoloed_polygons():
    # img_timestamp
    # classes
    # confidences
    # boxes
    box_path = ""
    box_file = pd.read_csv(box_path)
    df = pd.DataFrame(box_file)
    for i in range(len(df)):
        pass
    pass


# 给图片添加 yolo 并且输出
def multi_thread_by_time_interval(star_t, stop_t):
    time_interval.append([star_t, stop_t])
    list_file = find_images_by_time(base_img_dir=origin_path, time_interval=time_interval)
    total_num = len(list_file)
    i = 0
    for file in list_file:
        i += 1
        print("{} / {}".format(i, total_num))
        # already exist file do not write anymore
        if file in masked_files:
            continue
        img = cv2.imread(os.path.join(origin_path, file))
        result = find_mask(img, net)
        if result != ((), (), ()):
            result = filter_mask_class(result, list(desired_class), confidence_threshold)
            mask_img = put_mask(img, result[2])
        else:
            mask_img = img

        # fast = cv2.FastFeatureDetector_create(100)
        # kp = fast.detect(mask_img, None)
        # img2 = cv2.drawKeypoints(mask_img, kp, None, color=(255, 0, 0))
        # cv2.imshow("kp_not_rectangle", img2)

        cv2.imwrite(os.path.join(mask_path, file), mask_img)
        # cv2.waitKey(0)


# 快速地查看 yolo 结果
def take_look_yolo_img(path, desired_class, confidence_threshold):
    img = cv2.imread(path)
    result = find_mask(img, net)
    result = filter_mask_class(result, list(desired_class), confidence_threshold)
    mask_image = put_mask(img, result[2])
    cv2.imshow("img", mask_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    origin_path = "/media/godu/Element/Data/1204/mynteye/mav0/cam0/data"
    mask_path = "/media/godu/Element/Data/1204/mynteye/half/cap-6/mav0/cam0/data"
    config_path = "/home/godu/darknet/cfg/yolov4.cfg"
    weight_path = "/home/godu/darknet/yolov4.weights"
    # desired_class = np.array([0, 1, 2, 3])
    desired_class = np.array([0])
    confidence_threshold = 0.3
    begin_time = 1607060970.0
    end_time = 1607060978.0

    print("origin path: " + origin_path)
    print("mask_path: " + mask_path)
    print("start time: " + str(begin_time))
    print("stop_time: " + str(end_time))

    masked_files = os.listdir(mask_path)
    net = initial_yolo_model(config_path, weight_path)
    time_interval = list()
    threads = []

    # take_look_yolo_img("/home/godu/Desktop/labelme/1204-capture-label/cut/cap-1-cut.png", desired_class, confidence_threshold)

    # mid_time = (begin_time + end_time) / 2.0
    multi_thread_by_time_interval(begin_time, end_time)
    print("OK")

    # list_file = os.listdir(left_path)
    # start = 450
    # end = start + 100
    # for i in range(list_file.__len__()):
    #     if i < start:
    #         continue
    #     elif start <= i <= end:
    #         full_path = os.path.join(left_path, list_file[i])
    #         img = cv2.imread(full_path)
    #         offset = i - start + 1
    #         mask_img = img
    #         cv2.imshow('mask image', mask_img)
    #         full_path = os.path.join(mask_path, list_file[i])
    #         cv2.imwrite(full_path, mask_img)
    #         cv2.waitKey(0)
    #     else:
    #         break
