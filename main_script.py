import numpy as np
import cv2
import time
import csv
import os
import math
from function_script import newResVideo

#### USER INPUT PARAMETERS

# Input video name
filename = 'cow_out_new_trim'

# Input video file type (DO NOT PUT NAME AND TYPE TOGETHER) (examples: avi, mp4, mov)
video_filename = filename + '.avi'

# Input desired output csv file name
csv_filename = filename + '.csv'

# Set acceptable confidence rate
confidence_accept = 0.45

# Set true if resizing original video
resize = False

####

# Resize video if needed
if resize is True:
    new_res = (640, 360)
    out_filename = filename + '_copy.avi'
    video_placeholder = newResVideo(video_filename, out_filename, new_res)
    video_filename = out_filename

####

# Load Yolo
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Get all classes stored in coco names file
classes = []
with open("data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Find which index is for cow
cow_dex = classes.index('cow')
print('Index for cows is ' + str(cow_dex))

# Set up layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set capture video
cap = cv2.VideoCapture(video_filename)

# Set bounding box font
font = cv2.FONT_HERSHEY_PLAIN

# Set starting time
starting_time = time.time()

# Initialise frame count
frame_id = -1

# Get original video fps and frame count
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

####

# Set out output csv file
path_pre = 'output_csv/'

# Set output csv filename
path_file = csv_filename

# Concatenate path
path = str(path_pre + path_file)

# Check if want to overwrite current file
if os._exists(path) is True:
    continue_input = input('File exists. Continue?\n')
    if continue_input == "":
        os.remove(path)
    else:
        print('Script stopped, change filename\n')
        exit()

####

# Start loop to detect objects

while True:

    # Get current frame
    _, frame = cap.read()

    # Update frame number
    frame_id += 1

    # Skip frame if not a multiple of 3
    if frame_id == 0 or math.remainder(frame_id, 30) != 0:
        continue

    # Print frame number
    print(frame_id)

    # Crop frame
    frame = frame[100:290, 40:600]

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
            if confidence > confidence_accept and class_id == cow_dex:

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

    ####

    # Input all relevant data to output file for import into matlab
    with open(path, 'a') as f:

        # Create csv writer
        writer = csv.writer(f)

        # Write in order: index, ID, center, box
        writer.writerows([[frame_id], [indexes], [confidences], [centers_x], [centers_y], [widths], [heights]])
        # writer.writerow(indexes)
        # writer.writerow(class_ids)
        # writer.writerow(confidences)
        # writer.writerow(centers)
        # writer.writerow(boxes)

    ####

    # Draw up bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            cx, cy = centers[i]
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 182, 193), 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 182, 193), 2)

    # Calculate the duration in original video passed
    duration = frame_id/video_fps

    # Write detection data to checking video
    cv2.putText(frame, "Time: " + str(round(duration, 3)) + "s", (10, 20), font, 2, (0, 0, 0), 3)

    # Show frame
    cv2.imshow("Image", frame)

    # Get key
    key = cv2.waitKey(1)

    # Pause if space key is pressed
    if key == ord(' '):
        cv2.waitKey(-1)

    # Exit loop if "esc" is pressed
    if key == 27:
        break

# Close all windows
cap.release()
cv2.destroyAllWindows()