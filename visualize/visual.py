import pickle
import cv2
import numpy as np
import pandas as pd



# Import Settings :
with open('../settings.txt', 'rb') as f:
    x = pickle.load(f)

# Create video :
video = cv2.VideoWriter('visualize.avi', -1, 5, (x['width'],x['height']))
df = pd.read_csv('./dataset.csv')
rows = df[["X2[t]","S[t]"]].values


for row in rows:
    width = int(row[0])
    slope = row[1]
    # White Image :
    base_img = np.ones((x['height'], x['width'])) * 255

    # Read Stick :
    stick = cv2.imread('./stick.jpg', 0)


    # Rescale Stick :
    stick_height, stick_width = stick.shape
    scale = (x['height'] / stick_height) * 0.7

    stick = cv2.resize(stick, None, fx=scale, fy=scale)
    stick_height_scaled, stick_width_scaled = stick.shape

    # Overlay Image :
    base_img[50:50 + stick_height_scaled, width:width + stick_width_scaled] = stick
    base_img = base_img.astype('uint8')

    # Rotate Image Trick :
    base_img_height, base_img_width = base_img.shape
    rotation_matrix = cv2.getRotationMatrix2D((width + stick_width_scaled/2, 50 + stick_height_scaled/2), np.rad2deg(np.arctan(slope)), 1)
    base_img = cv2.warpAffine(base_img, rotation_matrix, (base_img_width, base_img_height), borderValue=(255,255,255))


    video.write(base_img)


cv2.destroyAllWindows()
video.release()


