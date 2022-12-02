import numpy as np
import cv2
import time

# Input video name
filename = 'image/cows.png'

# Load Yolo
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Get all classes stored in coco names file
classes = []
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Find which index is for cow
cow_dex = classes.index('cow')

# Set up layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set capture video
frame = cv2.imread(filename)

# Set bounding box font
font = cv2.FONT_HERSHEY_PLAIN

# Set starting time
starting_time = time.time()

####

# Get current frame
# _, frame = cap.read()

# Get size of frame
height, width, channels = frame.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(frame, 0.009392, (416, 416), (0, 0, 0), True, crop=False)

# Out: 3 outs, 8112, 2028, and 507 layers
net.setInput(blob)
outs = net.forward(output_layers)

# Showing info on the screen, initialise info array
class_ids = []
confidences = []
boxes = []
centers = []
centers_x = []
centers_y = []
widths = []
heights = []

# Loop output neural networks
for out in outs:

    # Get detected objects
    for detection in out:

        # Get scores for current detection
        scores = detection[5:]

        # Get class ids for all objects
        class_id = np.argmax(scores)

        # Get confidence of detected objects
        confidence = scores[class_id]

        # Check if confidence is high enough
        # if confidence > confidence_accept and class_id == cow_dex:

        # Get center and size (detection given in percentage of video)
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)           # Width of box
        h = int(detection[3] * height)          # Height of box

        # Rectangle top left coordinates
        x1 = int(center_x - w/1.8)
        y1 = int(center_y - h/1.8)

        # Append current coordinates to info array
        centers.append([center_x, center_y])
        centers_x.append(center_x)
        centers_y.append(center_y)
        boxes.append([x1, y1, w, h])
        widths.append(w)
        heights.append(h)
        confidences.append(float(confidence))
        class_ids.append(class_id)

# Get all index numbers of detected
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

# Draw up bounding boxes
for i in range(len(boxes)):
    if i in indexes:
        cx, cy = centers[i]
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 182, 193), 5)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        # cv2.putText(frame, label + " ", (x, y + 10), font, 5, (255, 182, 193), 10)
        cv2.putText(frame, label + " ", (x, y), font, 5, (0, 0, 255), 10)

# Show frame
# cv2.imshow("Image", frame)

cv2.imwrite('DETECTED_IMAGE.png', frame)

# Get key
# key = cv2.waitKey(1)

# # Close all windows
# cap.release()
# cv2.destroyAllWindows()