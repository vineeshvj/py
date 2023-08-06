import cv2
import mtcnn
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np


def detect_face(image):
    detector = MTCNN()
    img_rgb = np.array(image, cv2.COLOR_BGR2RGB)
    bounding_boxes = detector.detect_faces(img_rgb)
    return bounding_boxes



def draw_bounding_boxes(image, bboxes):
    for box in bboxes:
        x1, y1, w, h = box['box']
        cv2.rectangle(image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)


def mark_key_point(image, keypoint):
    cv2.circle(image, (keypoint), 1, (0,255,0), 2)




