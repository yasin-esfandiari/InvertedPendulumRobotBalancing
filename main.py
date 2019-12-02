import cv2
import numpy as np
from find_centers import find_centers
from utils import find_slope
import pandas as pd
import pickle

# Init params
dataset = []

# Path to video file
vidObj = cv2.VideoCapture('video.mp4')
fps = vidObj.get(cv2.CAP_PROP_FPS)

# Used as counter variable
frame_number = 0

# checks whether frames were extracted
success = 1

while success:
    # vidObj object calls read
    # function extract frames
    #
    success, img = vidObj.read()
    if(success):
        centers, img = find_centers(img)
        img_height, img_width, _ = img.shape

        # Oops! I cnat detect, go next
        if(centers == False):
            continue

        frame_number += 1

        # Saves the frames with frame-count
        cv2.imwrite("./frames/%d.jpg" % frame_number, img)

        # Find slope :
        slope = find_slope(centers[0], centers[1])

        # Save row :
        temp = [centers[1][0], slope]
        temp.append(1)
        temp.append(1)
        dataset.append(temp)
        if(frame_number != 1):
            dataset[frame_number - 2][2] = centers[1][0]
            dataset[frame_number - 2][3] = slope

# Save data :
df = pd.DataFrame(dataset, columns=["X2[t]","S[t]","X2[t+1]","S[t+1]"])
df.to_csv('dataset.csv', index=False)

# Save settings :
with open("settings.txt", "wb") as fp:   #Pickling
    pickle.dump({ "fps": fps , "width": img_width, "height": img_height }, fp)