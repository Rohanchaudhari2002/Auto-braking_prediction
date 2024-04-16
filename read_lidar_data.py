import cv2
import numpy as np

# load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getUnconnectedOutLayerNames()

cap = cv2.VideoCapture("data/video/bus.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
