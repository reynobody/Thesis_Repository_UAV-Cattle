import cv2
import numpy as np
import time
import csv
import os

####

# Function to resize video

def newResVideo(in_filename, out_filename, new_res):

    # Define a video capture object
    cap = cv2.VideoCapture(in_filename)

    # Capture video frame by frame
    success, image = cap.read()

    # Initialise frame count
    count = 0

    # Set fourcc option
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Get fps of original video and stick to that
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialise output video
    out = cv2.VideoWriter(out_filename, fourcc, fps, new_res)

    # Make the first frame the image shown when no images are detected
    img_empty = image

    # Creating a loop for running the video and saving all the frames
    while True:

        # Check if new frame is captured
        # if success is True:

        # Capture video frame by frame
        success, image = cap.read()

        # Write  updated frame to output video
        if image is None:

            b = cv2.resize(img_empty, new_res, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            print(counter)
            print('No Image')
            # exit()

        else:
            b = cv2.resize(image, new_res, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        # Write image to video
        out.write(b)

        # else:
        #     print("Video stopped outputting")
        #     break

        # Show updated video frame
        cv2.imshow('Image', b)

        # Wait
        key = cv2.waitKey(1)

        # Closing the video by Escape button
        if key == 27:
            break

    # Release all windows
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()

####

