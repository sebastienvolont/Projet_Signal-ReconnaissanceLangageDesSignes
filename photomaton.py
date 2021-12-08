import cv2
import os
import time
import uuid

import numpy as np

DEVICE = 0
IMAGE_SIZE = 256
IMAGES_PATH = os.path.join(os.getcwd(), 'img')
LABELS = [chr(i) for i in range(97, 123)]
NUMBER_IMG = 1


def take_picture(img: np.ndarray, path: str, label: str):
    pt1, pt2 = get_coord(*img.shape)
    pt1_x, pt1_y = pt1
    pt2_x, pt2_y = pt2
    path = os.path.join(path, label)
    os.makedirs(path, exist_ok=True)
    img_path = os.path.join(path, label + "." + f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(img_path, img[pt1_y:pt2_y, pt1_x:pt2_x])
    pass


def get_coord(height: int, width: int, color: int):
    width_pt1 = int((width - IMAGE_SIZE) / 2)
    height_pt1 = int((height - IMAGE_SIZE) / 2)
    width_pt2 = width_pt1 + IMAGE_SIZE
    height_pt2 = height_pt1 + IMAGE_SIZE
    pt1 = (width_pt1, height_pt1)
    pt2 = (width_pt2, height_pt2)
    return pt1, pt2


def draw_rectangle(img: np.ndarray):
    pt1, pt2 = get_coord(*img.shape)
    rectangle = cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
    return rectangle




def main():
    cap = cv2.VideoCapture(DEVICE)
    index = 0
    number_img = 0
    rafal = False
    while True:
        suc, img = cap.read()
        img_clone = img.copy()
        img = draw_rectangle(img)
        cv2.imshow("webcam", img)
        key = cv2.waitKey(1)
        if key == 27:  # esc stop code
            break
        elif key == 32 and not rafal:  # space for take picture
            rafal = True
            print(f"Starting to take image with '{LABELS[index]}'.")
        elif key == 113 and not rafal:
            last_index = index
            index -= 1
            if 0 > index:
                index = 0
            print(f"Before: {LABELS[last_index]}, now {LABELS[index]}")
        elif key == 100 and not rafal:
            previous_index = index
            index += 1
            if index == len(LABELS):
                index = (len(LABELS) - 1)
            print(f"Before: {LABELS[previous_index]}, now {LABELS[index]}")
        if rafal:
            take_picture(img_clone, IMAGES_PATH, LABELS[index])
            number_img += 1
            if number_img == NUMBER_IMG:
                print(f"Image with labels '{LABELS[index]}' is in the box.")
                rafal = False
                number_img = 0
    cap.release()


if __name__ == '__main__':
    print()
    main()
