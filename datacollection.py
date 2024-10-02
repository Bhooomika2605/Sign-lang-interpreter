import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of two hands
offset = 20
counter = 0
imgSize = 300 

folder = "/Users/bhoomika/OneDrive/Desktop/main project/Data/Think"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']

            # Ensure crop coordinates stay within image bounds
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)
            
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:  # Check if imgCrop is valid
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropShape = imgCrop.shape

            aspectratio = h/w
            if aspectratio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
