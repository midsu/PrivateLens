#!/.venv/bin/python
import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
# window frame
cap.set(3,648) # width
cap.set(4,480) #height


# --------------------Face Detection--------------------------
detector = FaceDetector(minDetectionCon=.50) #0 < minDC < 1

while True:
    success, img = cap.read()
    img, bboxs  = detector.findFaces(img,draw=False)
    # bboxs is a list of dicts 
    if bboxs:
        for i, bbox in enumerate(bboxs):
            # dimensions of bounding box
            x,y,w,h = bbox['bbox'] 
            if x < 0: x = 0
            if y < 0: y = 0

            imgCrop = img[y:y+h, x:x+w]
            imgBlur = cv2.blur(imgCrop, (35,35)) # must give odd numbers
            img[y:y+h, x:x+w] = imgBlur

            # snapshots of bbox
            # cv2.imshow(f'Imafe Cropped {i}', imgCrop)


    cv2.imshow("Image", img)
    cv2.waitKey(1)