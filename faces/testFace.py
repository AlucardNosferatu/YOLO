import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tiny_yolov1 import YOLO_head, iou
from airplanes.trainAirplane import model_tiny_YOLOv1
from utils import X_Y_W_H_To_Min_Max, load_img

classes_name = ['face']


class TinyYOLOv1(object):
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.classes_name = ['face']
        outputs, inputs = model_tiny_YOLOv1()
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.load_weights(self.weights_path, by_name=True)
        self.inputs = inputs

    def predict_path(self, input_path):
        image, _, _ = load_img(path=input_path, shape=self.inputs.shape[1:])
        image = np.expand_dims(image, axis=0)
        y = self.model.predict(image, batch_size=1)
        return y

    def predict_snap(self, input_snap):
        shape = self.inputs.shape[1:]
        if type(shape) is not tuple:
            shape = tuple(shape)
        if len(shape) < 3:
            shape = shape + (3,)
        image = input_snap
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, shape[:2])
        image = np.reshape(image, shape)
        image = image / 255.
        image = np.expand_dims(image, axis=0)
        y = self.model.predict(image, batch_size=1)
        return y


def _main(image_instance, tiny_YOLO_v1, snap=False):
    if snap:
        prediction = tiny_YOLO_v1.predict_snap(image_instance)
    else:
        prediction = tiny_YOLO_v1.predict_path(image_instance)
    captured = None
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

    box_xy, box_wh = YOLO_head(predict_box, img_size=224)  # 7 * 7 * 2 * 2
    box_xy_min, box_xy_max = X_Y_W_H_To_Min_Max(box_xy, box_wh)  # 7 * 7 * 2 * 2

    predict_trust *= filter_mask  # 7 * 7 * 2 * 1
    nms_mask = np.zeros_like(filter_mask)  # 7 * 7 * 2 * 1
    predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框
    score = predict_trust.copy()
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

    if snap:
        image = image_instance
    else:
        image = cv2.imread(image_instance)
    origin_shape = image.shape[0:2]
    image = cv2.resize(image, (224, 224))
    image_copy = image.copy()
    detect_shape = filter_mask.shape
    for i in range(detect_shape[0]):
        for j in range(detect_shape[1]):
            for k in range(detect_shape[2]):
                if nms_mask[i, j, k, 0]:
                    x1 = int(box_xy_min[i, j, k, 0])
                    y1 = int(box_xy_min[i, j, k, 1])
                    x2 = int(box_xy_max[i, j, k, 0])
                    y2 = int(box_xy_max[i, j, k, 1])
                    if snap and score[i, j, k, 0] > 0.95:
                        modified_x1 = max(0, x1 - 20)
                        modified_y1 = max(0, y1 - 20)
                        modified_x2 = min(image.shape[1] - 1, x2 + 20)
                        modified_y2 = min(image.shape[0] - 1, y2 + 20)
                        captured = image_copy[modified_y1:modified_y2, modified_x1:modified_x2]
                        # cv2.imshow('224*224', image)
                        # cv2.imshow('cap_224*224', captured)
                        captured = cv2.resize(
                            captured,
                            (
                                int(captured.shape[1] * origin_shape[1] / 224),
                                int(captured.shape[0] * origin_shape[0] / 224)
                            )
                        )
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255)
                    )
                    # cv2.putText(image, classes_name[box_classes[i, j, k, 0]],
                    #             (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                    #             1, 1, (0, 0, 255))
                    cv2.putText(image, str(score[i, j, k, 0]),
                                (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                                1, 1, (0, 0, 255))

    image = cv2.resize(image, (origin_shape[1], origin_shape[0]))
    cv2.imshow('image', image)
    if snap:
        if captured is not None and captured.shape[0] > 0 and captured.shape[1] > 0:
            return captured
        else:
            return None
    else:
        cv2.imwrite('YOLOv1-' + image_instance.split("\\")[-1], image)
        cv2.waitKey(0)
        return None


def test_images():
    years = ['2002', '2003']
    months = ['07', '08']
    dates = [str(i) for i in range(19, 29)]
    tyv1 = TinyYOLOv1('faces-tiny-yolov1.hdf5')
    for year in years:
        for month in months:
            for date in dates:
                path = 'C:/BaiduNetdiskDownload/originalPics/' + year + '/' + month + '/' + date + '/big'
                for e, i in enumerate(os.listdir(path)):
                    _main(
                        image_instance=os.path.join(path, i),
                        tiny_YOLO_v1=tyv1
                    )


def test_cam():
    tyv1 = TinyYOLOv1('faces-tiny-yolov1.hdf5')
    rec = tf.keras.models.load_model(
        'C:\\Users\\16413\\Desktop\\FFCS\\SVN\\CV_Toolbox\\SmartServerRoom\\Models\\Classifier.h5'
    )
    sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    # url = "http://admin:admin@10.78.50.12:8081"
    # sample = cv2.VideoCapture(url)
    while sample.isOpened():
        ret, frame = sample.read()
        slide_window = []
        if frame is not None:
            cap = _main(image_instance=frame, tiny_YOLO_v1=tyv1, snap=True)
            if cap is not None:
                cap = cv2.resize(cap, (224, 224)) / 255
                result = rec.predict(np.expand_dims(cap, axis=0))
                result = np.argmax(result[0])
                slide_window.append(result)
                if len(slide_window) > 50:
                    slide_window.pop(0)
                sum_sw = sum(slide_window)
                mean = sum_sw / len(slide_window)
                int_mean = int(mean)
                cv2.imshow(str(int_mean), cap)
        k = cv2.waitKey(50)
        if k & 0xff == ord('q'):
            break
        elif k & 0xff == ord('c'):
            if cap is not None:
                time = datetime.datetime.now()
                cv2.imwrite('YOLOv1-' + str(time.minute) + '-' + str(time.microsecond) + '.png', cap)
        # endregion
    sample.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_cam()
