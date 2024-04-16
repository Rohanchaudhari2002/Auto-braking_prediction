import math
import cv2 #To access the video
import dlib #To detect the object
import time 
import threading # To manage the process
import csv #To operate the csv file
import numpy as np #To perform array operation
import pandas as pd #For CSV operations

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

video_path = 'CutPOV.mp4'
WIDTH = 1280
HEIGHT = 720

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

def calculate_heading_angle(location1, location2):
    dx = location2[0] - location1[0]
    dy = location2[1] - location1[1]
    heading_angle = math.atan2(dy, dx) * 180 / math.pi
    return heading_angle

def position_x(speed, pos_x, heading_angle):
    return float(pos_x) + float(speed * math.cos(heading_angle) * float(0.083))

def position_y(speed, pos_y, heading_angle):
    return float(pos_y) + float(speed * math.cos(heading_angle) * float(0.083))

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = 1

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= min(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = 0
        p1x, p1y = p2x, p2y
    return inside
def trackMultipleObjects(video):
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000
    ROI_check = [[(350, 580), (1000, 580), (700, 400), (550, 400)]]

    with open('final_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Car ID', 'Speed (km/hr)', 'Heading Angle', 'Position X', 'Position Y', 'Pred X', 'Pred Y', 'Output'])
        
        while True:
            start_time = time.time()
            rc, image = video.read()
            if not rc:  # Check if frame is read successfully
                break

            image = cv2.resize(image, (WIDTH, HEIGHT))
            resultImage = image.copy()

            frameCounter += 1

            carIDtoDelete = []

            for carID in carTracker.keys():
                trackingQuality = carTracker[carID].update(image)
                if trackingQuality < 7:
                    carIDtoDelete.append(carID)

            for carID in carIDtoDelete:
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)

            if not (frameCounter % 10):
                # Detecting cars using YOLO
                blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5 and class_id == 2:
                            # Object is a car
                            center_x = int(detection[0] * WIDTH)
                            center_y = int(detection[1] * HEIGHT)
                            w = int(detection[2] * WIDTH)
                            h = int(detection[3] * HEIGHT)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        x_bar = x + 0.5 * w
                        y_bar = y + 0.5 * h

                        # Check if car is already being tracked
                        matchCarID = None
                        for carID in carTracker.keys():
                            trackedPosition = carTracker[carID].get_position()

                            t_x = int(trackedPosition.left())
                            t_y = int(trackedPosition.top())
                            t_w = int(trackedPosition.width())
                            t_h = int(trackedPosition.height())

                            t_x_bar = t_x + 0.5 * t_w
                            t_y_bar = t_y + 0.5 * t_h

                            if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                                    x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                matchCarID = carID

                        if matchCarID is None:
                            # Create new tracker
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                            carTracker[currentCarID] = tracker
                            carLocation1[currentCarID] = [x, y, w, h]

                            currentCarID += 1

            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

                carLocation2[carID] = [t_x, t_y, t_w, t_h]

            end_time = time.time()

            if not (end_time == start_time):
                fps = 1.0 / (end_time - start_time)

            for i in carLocation1.keys():
                if frameCounter % 1 == 0:
                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]

                    carLocation1[i] = [x2, y2, w2, h2]
                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        
                        if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                            speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                            
                        if speed[i] != None and y1 >= 180:
                            heading_angle = calculate_heading_angle([x1 + w1 // 2, y1 + h1 // 2],
                                                                    [x2 + w2 // 2, y2 + h2 // 2])
                            cv2.putText(resultImage, str(int(speed[i])) + " km/hr",
                                        (int(x1 + w1 / 2), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (255, 255, 255), 2)
                            x_pred_val = position_x(speed[i], t_x_bar, heading_angle)
                            y_pred_val = position_y(speed[i], t_y_bar, heading_angle)
                            
                            for polygon in ROI_check:
                               
                                if point_in_polygon((x_pred_val, y_pred_val), polygon):
                                    print("Point is outside ROI")
                                    output = 0
                                    cv2.circle(resultImage, (int(x_pred_val), int(y_pred_val)), 5, (255, 0, 0), -1)  # blue dot
                                    break
                                else:
                                    print("Point is inside ROI")
                                    output = 1
                                    cv2.circle(resultImage, (int(x_pred_val), int(y_pred_val)), 5, (0, 0, 255), -1)  # Red dot
                                    break
                            writer.writerow([frameCounter, i, speed[i], heading_angle, t_x_bar, t_y_bar, x_pred_val, y_pred_val, output])

            cv2.imshow('result', resultImage)

            if cv2.waitKey(33) == 27:
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    video = cv2.VideoCapture(video_path)
    trackMultipleObjects(video)
    video.release()  # Release the video capture object
