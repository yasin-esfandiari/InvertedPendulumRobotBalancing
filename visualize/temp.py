import pickle
import cv2
import numpy as np
import pandas as pd



# Import Settings :
with open('../settings.txt', 'rb') as f:
    x = pickle.load(f)

clean_border_vertical = 10
x['width'] += 2*clean_border_vertical


# Create video :
video = cv2.VideoWriter('visualize.avi', 0, 5, (x['width'],x['height']))
df = pd.read_csv('./dataset.csv')
rows = df[["X2[t]","S[t]"]].values


for row in rows:
    width = int(row[0])
    slope = row[1]
    # White Image :
    base_img = np.ones((x['height'], x['width'])) * 255

    # Read Stick :
    stick = cv2.imread('./stick.jpg', 0)


    # Make Border Stick :
    stick_height, stick_width = stick.shape
    clean_border_horizontal = int(stick_height/2)
    stick = cv2.copyMakeBorder(stick, top=clean_border_vertical, bottom=clean_border_vertical, left=clean_border_horizontal, right=clean_border_horizontal,
                                borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # Rotate Stick :
    rotation_matrix = cv2.getRotationMatrix2D(((stick_width + 2*clean_border_horizontal) / 2, (stick_height + 2*clean_border_vertical) / 2), np.rad2deg(np.arctan(slope)), 1)
    stick = cv2.warpAffine(stick, rotation_matrix, (stick_width + 2*clean_border_horizontal, stick_height + 2*clean_border_vertical), borderValue=(255,255,255))

    # Rescale Stick :
    scale = (x['height'] / stick_height) * 0.7
    stick = cv2.resize(stick, None, fx=scale, fy=scale)
    stick_height_scaled, stick_width_scaled = stick.shape
    clean_border_horizontal_scaled = int(clean_border_horizontal * scale)
    clean_border_verical_scaled = int(clean_border_vertical * scale)


    # Overlay Image :
    print(base_img.shape)
    print(base_img[0:(0 + stick_height_scaled), width:(width + stick_width_scaled)].shape)
    print(stick.shape)
    base_img[0:0 + stick_height_scaled, width:width + stick_width_scaled] = stick
    base_img = base_img.astype('uint8')
    print(base_img.shape)

    video.write(base_img)


cv2.destroyAllWindows()
video.release()


