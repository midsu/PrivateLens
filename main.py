# #!/.venv/bin/python
# import cv2
# from cvzone.FaceDetectionModule import FaceDetector

# cap = cv2.VideoCapture(0)
# # window frame
# cap.set(3,1280) # monitor x pos, width   #648
# cap.set(4,720)  # monitor y pos, height  #480


# # --------------------Face Detection--------------------------
# detector = FaceDetector(minDetectionCon=.50) #0 < minDC < 1

# while True:
#     success, img = cap.read()
#     img, bboxs  = detector.findFaces(img,draw=True)
#     # bboxs is a list of dicts 
#     if bboxs:
#         for i, bbox in enumerate(bboxs):
#             # dimensions of bounding box
#             x,y,w,h = bbox['bbox'] 
#             if x < 0: x = 0
#             if y < 0: y = 0

#             imgCrop = img[y:y+h, x:x+w]
#             imgBlur = cv2.blur(imgCrop, (35,35)) # must give odd numbers
#             img[y:y+h, x:x+w] = imgBlur

#             # snapshots of bbox
#             # cv2.imshow(f'Image Cropped {i}', imgCrop)


#     cv2.imshow("Private Lens", img)
#     cv2.waitKey(1)

import face_recognition
import os, sys
import cv2
import numpy as np
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0-face_distance)/(range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value,2)) + '%'
    
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings  = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
    
    def encode_faces(self):
        for img in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{img}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(img)

        print(self.known_face_names)

    def run_recognition(self):
        # VideoCapture(0) is defeault to first camera might need to adjust if more than one camera source
        video_capture = cv2.VideoCapture(0)

        # this checks for if the progam has permission to use camera
        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        #loop grabs every other frame and processes it
        while True:
            ret, frame = video_capture.read()
            print("FRAME:", frame)
            if self.process_current_frame:
                #resizing the frame by 1/4th to save resources 
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                #some reason its rbg so we switched it to be rgb
                rgb_small_frame = small_frame[:,:,::-1]

                #finding all faces in current frames
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                    
                    self.face_names.append(f'{name} ({confidence})')
            
            self.process_current_frame = not self.process_current_frame
            
            #Display Annotations
            for(top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left,top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left,bottom-35), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord('q'):
                break

            video_capture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()