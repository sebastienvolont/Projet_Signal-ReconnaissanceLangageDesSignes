import glob
import os
from pathlib import Path

from PIL import Image
import random
import cv2


def add_noise(img):
    row, col, color = img.shape

    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)

        x_coord = random.randint(0, col - 1)

        img[y_coord][x_coord] = 255

    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)

        x_coord = random.randint(0, col - 1)

        img[y_coord][x_coord] = 0

    return img




files = list(glob.glob('img/*/*.jpg', recursive=True))
for i, filename in enumerate(files):
    if i % 100 == 0:
        print(f'Files {i}/{len(files)}')

    _, folder, fn = filename.split(os.path.sep)
    dest = Path('dataset') / folder / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = add_noise(cv2.imread(filename))
    img = Image.fromarray(img).convert('L')
    img.resize((32, 32)).save(dest)

print('Finished processing files')
