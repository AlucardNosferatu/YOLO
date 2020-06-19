import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

from YOLOv2 import YOLO_ANCHORS, YoloLoss, yolo_bboxes, class_names, Activate_GPU


def yolo_eval(yolo_outputs, image_shape=(416, 416), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    anchors = YOLO_ANCHORS
    # Retrieve outputs of the YOLO model.
    box_xy, box_wh, box_confidence, box_class_probs = yolo_bboxes(yolo_outputs, anchors, 1)

    # Convert boxes to be ready for filtering functions.
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Perform Score-filtering with a threshold of score_threshold.
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)

    # perform Non-max suppression with a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    # Compute box scores
    box_scores = box_confidence * box_class_probs

    # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = tf.argmax(box_scores, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)

    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold

    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    print(">>>", scores.shape, boxes.shape, classes.shape, scores)

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_boxes, iou_threshold)

    # Use tf.gather() to select only selected_indices from scores, boxes and classes
    scores = tf.gather(scores, selected_indices)
    boxes = tf.gather(boxes, selected_indices)
    classes = tf.gather(classes, selected_indices)

    return scores, boxes, classes


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names):
    h, w, _ = image.shape
    for box in out_boxes:
        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        # colors: RGB
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    return image


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.concat([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2],  # y_max
    ], -1)


def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = tf.stack([width, height, width, height])
    image_dims = tf.cast(tf.reshape(image_dims, [1, 4]), tf.float32)
    boxes = boxes * image_dims
    return boxes


def load_model():
    anchors = YOLO_ANCHORS
    model = tf.keras.models.load_model(
        "model_data/yolov2-trained.h5",
        custom_objects={
            "yolo_loss": YoloLoss(anchors, classes=1)
        }
    )
    return model


def predict_single_image(model, img_filepath="./data/kangaroo/images/00054.jpg"):
    print(img_filepath)
    # 使用OpenCV讀入圖像
    images = cv2.imread(img_filepath)  # 載入圖像
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    # 進行圖像輸入的前處理
    input_image = cv2.resize(images, (416, 416))  # 修改輸入圖像大小來符合模型的要求
    input_image = input_image / 255.  # 進行圖像歸一處理
    input_image = np.expand_dims(input_image, 0)  # 增加 batch dimension
    # 進行圖像偵測
    yolo_outputs = model.predict(input_image)
    image = cv2.resize(images, (416, 416))
    scores, boxes, classes = yolo_eval(yolo_outputs, score_threshold=0.3)
    # Draw bounding boxes on the image file
    image = draw_boxes(image, scores, boxes, classes, class_names)
    # Save
    # cv2.imwrite("./model_data/output.jpg", image)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.show()


def predict_batch():
    path = "./data/kangaroo/images"
    model = load_model()
    for e, i in enumerate(os.listdir(path)):
        predict_single_image(model, os.path.join(path, i))


Activate_GPU()
predict_batch()
