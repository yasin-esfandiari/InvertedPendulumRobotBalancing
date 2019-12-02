import cv2
import numpy as np


def find_centers(img):
    RED = (0, 0, 255)
    height = img.shape[0]
    width = img.shape[1]

    max_width = 3 / 5 * height

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (w < max_width):
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(img, (center_x, center_y), 5, RED, -1)
            centers.append((center_x, center_y))
            cv2.drawContours(img, [contour], 0, RED, 2)

    # cv2.imwrite('CentersDetected.jpg', img)

    if len(centers) != 2:
        return False, img

    # Everything is ok, sort centers!
    centers.sort(key=lambda x: x[1])

    return (centers, img)
