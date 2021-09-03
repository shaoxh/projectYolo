# coding=utf-8

import time
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np


def get_img_list(file_path):
    files = list()
    # cycle and if judgement all by one line
    # files.append(file for file in listdir(file_path) if isfile(join(file_path, file)))
    for file in listdir(file_path):
        if isfile(join(file_path, file)):
            files.append(file)

    return files[::1]


def filter_classId(classId, confidence):
    # frisbee, skateboard, pottedplant, traffic light, handbag, kite, stop sign skip
    if classId in [29, 36, 58, 9, 26, 32, 73, 11]:
        return True
    # if classId in [0, 1, 7, 2]:
    #     return False
    return False


def put_mask(image, boxes, classes):
    # 1.读取图片
    # 2.获取标签
    # 标签格式　bbox = [xl, yl, xr, yr]
    bbox1 = [72, 41, 208, 330]
    bbox2 = [100, 80, 248, 334]

    # 3.画出mask
    zeros1 = np.zeros((image.shape), dtype=np.uint8)
    zeros2 = np.zeros((image.shape), dtype=np.uint8)

    zeros_mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                                color=(0, 0, 255), thickness=-1)  # thickness=-1 表示矩形框内颜色填充
    zeros_mask2 = cv2.rectangle(zeros2, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]),
                                color=(0, 255, 0), thickness=-1)

    zeros_mask = np.array((zeros_mask1 + zeros_mask2))

    # mask_person_list = list()
    # mask_non_person_list = list()
    # zeros_mask1 = zeros1
    # zeros_mask2 = zeros2
    # factor = boxes
    # for i in range(len(boxes)):
    #     bbox = boxes[i]
    #     if classes[i] == 0:
    #         # person
    #         a = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
    #                                               color=(0, 0, 255), thickness=-1)
    #         mask_person_list.append(a)
    #     else:
    #         mask_non_person_list.append(cv2.rectangle(zeros1, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]),
    #                                                   color=(0, 0, 255), thickness=-1))  # thickness=-1 表示矩形框内颜色填充)
    #
    # for mask_person in mask_person_list:
    #     zeros_mask1 += mask_person
    # for mask_none_person in mask_non_person_list:
    #     zeros_mask2 += mask_none_person
    # zeros_mask = np.array((zeros_mask1 + zeros_mask2))

    try:
        # alpha 为第一张图片的透明度
        alpha = 1
        # beta 为第二张图片的透明度
        beta = 0.5
        gamma = 0
        # cv2.addWeighted 将原始图片与 mask 融合
        mask_img = cv2.addWeighted(image, alpha, zeros_mask, beta, gamma)
        return mask_img
    except:
        print('异常')


def output_box(img_name, classes, confidences, boxes, output_path, file_name):
    img_timestamp = img_name[0:-4]
    file = join(output_path, file_name)
    str_classes = str(np.transpose(classes)[0]).replace('\n', '')
    str_confidences = str(np.transpose(confidences)[0]).replace('\n', '')
    str_boxes = str(boxes.flatten()).replace('\n', ' ')
    if not isfile(join(output_path, file_name)):
        with open(file, 'w') as csv_f:
            csv_f.write("img_timestamp" + "," + "classes" + "," + "confidences" + "," + "boxes")

    with open(file, 'a+') as box_file:
        line = str(img_timestamp + "," + str_classes + "," + str_confidences + "," + str_boxes)
        box_file.write(line + '\n')
        box_file.close()
    pass


def output_statistics(img_name, statistics, output_path, file_name):
    file = join(output_path, file_name)
    img_timestamp = img_name[0:-4]
    line = ""
    for key in statistics:
        line += "," + str(key) + "," + str(statistics[key])
    line = img_timestamp + line
    with open(file, "a+") as statistics_file:
        statistics_file.write(line + '\n')
        statistics_file.close()


if __name__ == '__main__':

    # 更改至相应的文件目录
    configPath = "/home/godu/darknet/cfg/yolov4.cfg"
    weightPath = "/home/godu/darknet/yolov4.weights"
    metaPath = "/home/godu/darknet/cfg/coco.data"
    imgPath = "/home/godu/Documents/Data/1204_captured/cap1"
    namePath = "/home/godu/darknet/data/coco.names"
    outputPath = "./"

    img_2_yolo = get_img_list(imgPath)
    img_2_yolo.sort()
    # print(len(img_2_yolo))
    # print(img_2_yolo[0])
    # print(img_2_yolo[len(img_2_yolo)-1])

    # use python-opencv implement YOLO V4 model params
    net = cv2.dnn_DetectionModel(configPath, weightPath)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    with open(namePath, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    print(names)
    statistics = dict()
    cost_time = 0;
    for img in img_2_yolo:
        frame = cv2.imread(join(imgPath, img))
        startTime = time.time()
        classes, confidences, boxes = net.detect(frame, confThreshold=0.9, nmsThreshold=0.9)
        endTime = time.time()
        print("Time: {}s".format(endTime - startTime))
        cost_time = cost_time + (endTime - startTime)
        cv2.imshow("in", frame)

        if len(classes) != 0:
            # statistics = dict()
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                # filter some classes that don't count
                if filter_classId(classId, confidence):
                    continue
                # classes statistics and create statistic label
                if names[classId] in statistics:
                    statistics[names[classId]] = statistics.get(names[classId]) + 1
                else:
                    statistics[names[classId]] = 1

                # label = '%.2f' % confidence
                # label = '%s: %s' % (names[classId], label)
                label = names[classId]
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                top = max(top, labelSize[1])
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # put colored mask into rectangle
            #   frame = put_mask(frame, boxes, classes)
            #
            # output boxes & statistics
            output_box(img, classes, confidences, boxes, outputPath, "box_file.csv")
            output_statistics(img, statistics, outputPath, "statistics_file.csv")

            sta_label_size, sta_baseline = cv2.getTextSize(str(statistics), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            st_top = 10
            st_left = 32
            cv2.rectangle(frame, (st_top, st_left - sta_label_size[1]),
                          (st_top + sta_label_size[0], st_left + sta_baseline),
                          (255, 255, 255),
                          cv2.FILLED)
            cv2.putText(frame, str(statistics), (st_top, st_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            time_left = 15
            time_seq = "Timestamp:" + str(img.split(".")[0]) + "s"
            time_label_size, time_baseline = cv2.getTextSize(time_seq, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (st_top, time_left - time_label_size[1]),
                          (st_top + time_label_size[0], time_left + time_baseline),
                          (255, 255, 255),
                          cv2.FILLED)
            cv2.putText(frame, time_seq, (st_top, time_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            cv2.imshow('out', frame)
            cv2.waitKey(10)

    print("Total {} images cost time: {}s".format(len(img_2_yolo), cost_time))
    print(statistics)
