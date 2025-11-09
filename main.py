#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness,
                                             keypoint_classifier_labels[hand_sign_id])

        debug_image = draw_info(debug_image, fps)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    points = np.array([[int(l.x * image_width), int(l.y * image_height)]
                       for l in landmarks.landmark])
    x, y, w, h = cv.boundingRect(points)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[int(l.x * image_width), int(l.y * image_height)]
            for l in landmarks.landmark]


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    temp = [[x - base_x, y - base_y] for x, y in temp]
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(list(map(abs, temp)))
    return [n / max_value for n in temp]


def draw_landmarks(image, landmark_point):
    for pts in [(2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
        for i in range(len(pts)-1):
            cv.line(image, tuple(landmark_point[pts[i]]), tuple(landmark_point[pts[i+1]]),
                    (0,0,0), 6)
            cv.line(image, tuple(landmark_point[pts[i]]), tuple(landmark_point[pts[i+1]]),
                    (255,255,255), 2)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]),
                     (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0,0,0), -1)
    info_text = handedness.classification[0].label + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    return image


def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255,255,255), 2, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()