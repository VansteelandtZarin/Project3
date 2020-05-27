import datetime

import cv2
import numpy as np
import glob
import random
import time

# Load Yolo
# light model
# net = cv2.dnn.readNet("tiny_weights.weights", "tiny_config.cfg")
#  medium model
# net = cv2.dnn.readNet("yolov3_medium.weights", "medium_config.cfg")
#  heavy model
net = cv2.dnn.readNet("yolov3_test_training_last.weights", "yolov3_config.cfg")


# Name custom object
classes = ["Jam", "Knife", "Bread", "Choco"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture('testvids/jam test vid.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('testoutput1.avi', fourcc, 20.0, (640, 480))

codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # MP4
out = cv2.VideoWriter('testoutput1.avi', codec, 60, (640, 480))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        height, width, channels = frame.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    print("%s: %s " % (class_id, confidence) + '%')
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)

        # out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
