import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imageSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        x, y, w, h = hand1["bbox"]

        imgWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imageSize - wCal) / 2)
            imgWhite[:, widthGap:widthGap + wCal] = imgResize
        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imageSize - hCal) / 2)
            imgWhite[heightGap:heightGap + hCal, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        if len(hands) == 2:
            hand2 = hands[1]
            x2, y2, w2, h2 = hand2["bbox"]
            imgWhite2 = np.ones((imageSize, imageSize, 3), np.uint8) * 255
            imgCrop2 = img[y2-offset:y2 + h2 + offset, x2-offset:x2 + w2 + offset]

            imgCrop2Shape = imgCrop2.shape
            aspectRatio2 = h2 / w2

            if aspectRatio2 > 1:
                k2 = imageSize / h2
                wCal2 = math.ceil(k2 * w2)
                imgResize2 = cv2.resize(imgCrop2, (wCal2, imageSize))
                imgResize2Shape = imgResize2.shape
                widthGap2 = math.ceil((imageSize - wCal2) / 2)
                imgWhite2[:, widthGap2:widthGap2 + wCal2] = imgResize2
            else:
                k2 = imageSize / w2
                hCal2 = math.ceil(k2 * h2)
                imgResize2 = cv2.resize(imgCrop2, (imageSize, hCal2))
                imgResize2Shape = imgResize2.shape
                heightGap2 = math.ceil((imageSize - hCal2) / 2)
                imgWhite2[heightGap2:heightGap2 + hCal2, :] = imgResize2

            cv2.imshow("ImageCrop2", imgCrop2)
            cv2.imshow("ImageWhite2", imgWhite2)

    cv2.imshow("Image", img)
    
    # Wait for 1 ms and check if 'q' key is pressed to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


