import os

import cv2
import numpy as np
import tensorflow as tf

from VOC2007.tiny_yolov1 import YOLO_head, iou
from YOLOv1.trainAirplane import ReshapeYOLO
from YOLOv2.trainAirplane import yolo_loss
from utils import load_img, X_Y_W_H_To_Min_Max

classes_name = ['aeroplane']
image_size = 448


def testV2():
    save_dir = '../checkpoints'
    weights_path = os.path.join(save_dir, 'V2-weights.hdf5')
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={
            "ReshapeYOLO": ReshapeYOLO,
            "yolo_loss": yolo_loss
        }
    )
    path = '../Data/Images'
    for e, i in enumerate(os.listdir(path)):
        if i.startswith("airplane"):
            predict_single_file(model=model, image_path=os.path.join(path, i))


def predict_single_file(model, image_path):
    image, _, _ = load_img(path=image_path, shape=model.input.shape[1:])
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predict_class = prediction[..., :1]  # 1 * 7 * 7 * 20
    predict_trust = prediction[..., 1:3]  # 1 * 7 * 7 * 2
    predict_box = prediction[..., 3:]  # 1 * 7 * 7 * 8

    predict_class = np.reshape(predict_class, [7, 7, 1, 1])
    predict_trust = np.reshape(predict_trust, [7, 7, 2, 1])
    predict_box = np.reshape(predict_box, [7, 7, 2, 4])

    predict_scores = predict_class * predict_trust  # 7 * 7 * 2 * 1

    box_classes = np.argmax(predict_scores, axis=-1)  # 7 * 7 * 2
    box_class_scores = np.max(predict_scores, axis=-1)  # 7 * 7 * 2
    best_box_class_scores = np.max(box_class_scores, axis=-1, keepdims=True)  # 7 * 7 * 1

    box_mask = box_class_scores >= best_box_class_scores  # ? * 7 * 7 * 2

    filter_mask = box_class_scores >= np.max(box_class_scores) / 2  # 7 * 7 * 2
    filter_mask *= box_mask  # 7 * 7 * 2

    filter_mask = np.expand_dims(filter_mask, axis=-1)  # 7 * 7 * 2 * 1

    predict_scores *= filter_mask  # 7 * 7 * 2 * 20
    predict_box *= filter_mask  # 7 * 7 * 2 * 4

    box_classes = np.expand_dims(box_classes, axis=-1)
    box_classes *= filter_mask  # 7 * 7 * 2 * 1

    box_xy, box_wh = YOLO_head(predict_box, img_size=image_size)  # 7 * 7 * 2 * 2
    box_xy_min, box_xy_max = X_Y_W_H_To_Min_Max(box_xy, box_wh)  # 7 * 7 * 2 * 2

    predict_trust *= filter_mask  # 7 * 7 * 2 * 1
    nms_mask = np.zeros_like(filter_mask)  # 7 * 7 * 2 * 1
    predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框
    max_i = max_j = max_k = 0
    while predict_trust_max > 0:
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if predict_trust[i, j, k, 0] == predict_trust_max:
                        nms_mask[i, j, k, 0] = 1
                        filter_mask[i, j, k, 0] = 0
                        max_i = i
                        max_j = j
                        max_k = k
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if filter_mask[i, j, k, 0] == 1:
                        iou_score = iou(box_xy_min[max_i, max_j, max_k, :],
                                        box_xy_max[max_i, max_j, max_k, :],
                                        box_xy_min[i, j, k, :],
                                        box_xy_max[i, j, k, :])
                        if iou_score > 0.2:
                            filter_mask[i, j, k, 0] = 0
        predict_trust *= filter_mask  # 7 * 7 * 2 * 1
        predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框

    box_xy_min *= nms_mask
    box_xy_max *= nms_mask

    image = cv2.imread(image_path)
    origin_shape = image.shape[0:2]
    image = cv2.resize(image, (image_size, image_size))
    detect_shape = filter_mask.shape

    for i in range(detect_shape[0]):
        for j in range(detect_shape[1]):
            for k in range(detect_shape[2]):
                if nms_mask[i, j, k, 0]:
                    cv2.rectangle(image, (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                                  (int(box_xy_max[i, j, k, 0]), int(box_xy_max[i, j, k, 1])),
                                  (0, 0, 255))
                    cv2.putText(image, classes_name[box_classes[i, j, k, 0]],
                                (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                                1, 1, (0, 0, 255))

    # image = cv2.resize(image, (origin_shape[1], origin_shape[0]))
    cv2.imshow('image', image)
    cv2.imwrite('..\\Results\\YOLOv2-' + image_path.split("\\")[-1], image)
    cv2.waitKey(0)


if __name__ == '__main__':
    testV2()
