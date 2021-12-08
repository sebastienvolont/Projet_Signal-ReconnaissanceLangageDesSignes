import math
import uuid
import cv2
import mediapipe as mp
import time
import os
from pprint import pprint
import numpy as np
from string import ascii_lowercase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = 256
IMG_RESIZE = 32
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def hand_square(hand):
    min_coord_x = math.inf
    max_coord_x = - math.inf
    min_coord_y = math.inf
    max_coord_y = - math.inf
    for landmark in hand.landmark:
        min_coord_x = min(min_coord_x, landmark.x)
        max_coord_x = max(max_coord_x, landmark.x)
        min_coord_y = min(min_coord_y, landmark.y)
        max_coord_y = max(max_coord_y, landmark.y)
    return [[min_coord_x, min_coord_y], [max_coord_x, max_coord_y]]


def center(min_pt, max_pt, height, width):
    center_x = (max_pt[0] + min_pt[0]) / 2 * width
    center_y = (max_pt[1] + min_pt[1]) / 2 * height
    center_x = max(min(center_x, width - (IMG_SIZE / 2)), IMG_SIZE / 2)
    center_y = max(min(center_y, height - (IMG_SIZE / 2)), IMG_SIZE / 2)
    return int(center_x), int(center_y)


def pre_process(frame):
    frame = cv2.resize(frame, (IMG_RESIZE, IMG_RESIZE))
    frame = tf.image.rgb_to_grayscale(frame).numpy()
    return frame


def create_model():
    model = Sequential([
        layers.Rescaling(1 / 255),
        layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(26)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.load_weights("data/cp.ckpt")
    return model


def main():
    cap = cv2.VideoCapture(0)
    model = create_model()
    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, c = image.shape
                    pt1, pt2 = hand_square(hand_landmarks)
                    center_x, center_y = center(pt1, pt2, h, w)
                    square_pt1 = (center_x - (IMG_SIZE // 2), center_y - (IMG_SIZE // 2))
                    square_pt2 = (center_x + (IMG_SIZE // 2), center_y + (IMG_SIZE // 2))
                    clone = image.copy()[square_pt1[1]:square_pt2[1], square_pt1[0]:square_pt2[0]]

                    # print(pre_process(clone).shape)
                    result = model.predict([pre_process(clone)])
                    result = ascii_lowercase[np.argmax(result)]
                    cv2.putText(image, result, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    cv2.rectangle(image, square_pt1, square_pt2, (0, 255, 0), 1)
            # print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


if __name__ == '__main__':
    main()
