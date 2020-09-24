import numpy as np

import matplotlib.pyplot as plt
import cv2

import msvcrt,time


classes = r'C:\Users\dell\OneDrive\Desktop\Projects\Projects+\YOLO_v3\coco.names'
config = r'C:\Users\dell\OneDrive\Desktop\Projects\Projects+\YOLO_v3\yolov3.cfg'
weights = r'C:\Users\dell\OneDrive\Desktop\Projects\Projects+\YOLO_v3\yolov3.weights'

LABELS = open(classes).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

count = 0


model = cv2.dnn.readNet(config, weights)     # 'readNet()' automatically directs the program to 'readNetFromDarknet(config, weights)'

###############################
def predict_yolo():
    THRESHOLD = 0.6
    boxes = []
    confidences = []
    classIDs = []

    for detection in outputs[0]:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > THRESHOLD:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, THRESHOLD,THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
    # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

    # show the output image
    cv2.imwrite("example_"+str(count)+".png", img)

###############################################################


cap=cv2.VideoCapture(0)                # open webcam command
print("\nPress ENTER to Quit")

while cap.isOpened():
    ret, img = cap.read()              # returns 2 values
    (H, W) = img.shape[:2]

    ln = model.getLayerNames()
    output_ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    img_blob = cv2.dnn.blobFromImage(img, 1/255.0)       # Re-Scaling is IMPORTANT
    model.setInput(img_blob)
    outputs = model.forward(output_ln)
    predict_yolo()
    count+=1

    if msvcrt.kbhit():                     
        break                              
    else:                                  
        time.sleep(3)
cap.release()                          # close webcam (not required)