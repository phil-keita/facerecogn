# Author: Philippe Keita
# Facial recognition based on LBPH

import cv2
import numpy as np
import os


def detect_face(input_img):
    # face detection modal
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # converting image to gray scale
    gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # detection
    faces = face_cascade.detectMultiScale(input_img, scaleFactor=1.2, minNeighbors= 5)
    if(len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return gray_image[y:y+h, x:x+w], faces[0]

def draw_rectangle(input_img, rec, label, wrong = 0):
    # Box coordinates
    (x, y, w, h) = rec
    # Color of box
    color = (0,255,0)
    if wrong:
        color = (0, 0, 255)
        label = 'Not ' + label
    # Font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Padding for text (only for visibility)
    padding = 5
    
    output_image = cv2.rectangle(input_img, (x,y), (x+w, y+h), color, 2)
    return cv2.putText(output_image, label, (x, y-padding), font, 1, color, 2, cv2.LINE_AA)

def get_training_data(training_data_path):
    num_img = os.listdir(training_data_path)
    detected_faces = []
    labels = []
    # for i in range(len(num_img)):
    #     image_path = training_data_path + str(i) + '.jpg'
    #     print(image_path)
    #     gray_image = cv2.imread(image_path)
    #     face, rect = detect_face(gray_image)
    #     resized = cv2.resize(face, (120,120), interpolation=cv2.INTER_AREA)
    #     detected_faces.append(resized)
    #     labels.append(1)
        
    for i in range(10):
        image_path = training_data_path + 'lebron/' +  str(i) + '.jpg'
        print(image_path)
        gray_image = cv2.imread(image_path)
        face, rect = detect_face(gray_image)
        resized = cv2.resize(face, (120,120), interpolation=cv2.INTER_AREA)
        detected_faces.append(resized)
        labels.append(1)   
    for i in range(10):
        image_path = training_data_path + 'stephen/' +  str(i) + '.jpg'
        print(image_path)
        gray_image = cv2.imread(image_path)
        face, rect = detect_face(gray_image)
        resized = cv2.resize(face, (120,120), interpolation=cv2.INTER_AREA)
        detected_faces.append(resized)
        labels.append(2)  
    return detected_faces, labels

def predict_image(input_img, expected_label):
    lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
    path_to_training = 'assets/training/'
    detected_faces, labels = get_training_data(path_to_training)

    lbph_recognizer.train(detected_faces, np.array(labels))

    face, rec = detect_face(input_img)
    resized = cv2.resize(face, (120,120), interpolation=cv2.INTER_AREA)
    cv2.imshow("sample",resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    label, confidence = lbph_recognizer.predict(resized)
    
    print('Confidence: ' + str(confidence) + ', label: ' + str(label))
    # cv2.imshow("Prediction", draw_rectangle(input_img, rec, "Lebron", confidence <= 99.0))

# testing image
sample_img = cv2.imread('assets/testing/5.jpg')

predict_image(sample_img, 'lebron')

# cv2.imshow("data",sample_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

