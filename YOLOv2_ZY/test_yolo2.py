'''
Created on Jun 20, 2020

@author: monky
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input,  \
                                   MaxPooling2D, BatchNormalization,     \
                                    Flatten, Dense, Lambda, LeakyReLU,  concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from yolov2_pro import draw_boxes, yolo_eval, YoloLoss, anchors

gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpu_list)
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)


# ### Start detect
# img_filepath= "./data/airplane/images/airplane_004.jpg"
imagefile = 'airplane (7).jpg'
img_filepath= "test\\" + imagefile
img_outpath = "./output/output_" + imagefile
print(img_filepath)

images = cv2.imread(img_filepath) 
images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

input_image = cv2.resize(images, (448, 448)) 
input_image = input_image / 255. 
input_image = np.expand_dims(input_image, 0) 

# Load
yolo_loss = YoloLoss(anchors, classes=1)
model = tf.keras.models.load_model("model/yolov2_airplane.h5", custom_objects={'yolo_loss': yolo_loss})


print('===')

yolo_outputs = model.predict(input_image)

print(yolo_outputs)
image = cv2.resize(images, (448, 448))
scores, boxes, classes = yolo_eval(yolo_outputs, score_threshold=0.3)

# Draw bounding boxes on the image file
class_names = ['airplane']
image = draw_boxes(image, scores, boxes, classes, class_names)

# Save
cv2.imwrite(img_outpath, image)

cv2.imshow('result', image)
cv2.waitKey(2000)

