import cv2
import numpy as np


def load_img(path, shape=(448, 448, 3)):
    if type(shape) is not tuple:
        shape = tuple(shape)
    if len(shape) < 3:
        shape = shape + (3,)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[0:2]
    image = cv2.resize(image, shape[:2])
    image = np.reshape(image, shape)
    image = image / 255.
    return image, image_h, image_w
