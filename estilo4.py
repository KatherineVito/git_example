import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
face_image = cv2.imread("pelo7.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        top_face = (landmarks.part(27).x, landmarks.part(27).y)
        center_face = (landmarks.part(27).x, landmarks.part(27).y)
        left_face = (landmarks.part(0).x, landmarks.part(0).y)
        right_face = (landmarks.part(16).x, landmarks.part(16).y)

        face_width = int(hypot(left_face[0] - right_face[0],
                           left_face[1] - right_face[1] * 2))
        face_height = int(face_width * 1.30)

        top_face = (int(center_face[0] - face_width / 2),
                        int(center_face[1] - face_height / 2))
        botton_right = (int(center_face[0] + face_width / 2),
                        int(center_face[1] + face_height / 2))


        #cv2.circle(frame, top_face, 3, (255, 0, 0), 4)
        face_h = cv2.resize(face_image, (face_width, face_height))
        face_h_gray = cv2.cvtColor(face_h, cv2.COLOR_BGR2GRAY)
        _, face_mask = cv2.threshold(face_h_gray, 25, 255, cv2.THRESH_BINARY_INV)

        face_area = frame[top_face[1]: top_face[1] + face_height,
                    top_face[0]: top_face[0] + face_width]
        face_area_no_face = cv2.bitwise_and(face_area, face_area)
        final_face = cv2.add(face_area_no_face, face_h)

        frame[top_face[1]: top_face[1] + face_height,
              top_face[0]: top_face[0] + face_width] = final_face


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break