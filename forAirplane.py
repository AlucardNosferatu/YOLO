import os

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from data_sequence import SequenceForAirplanes
from models.model_tiny_yolov1 import original_backbone_network
from yolo.yolo import x_y_w_h_2_min_max, iou, yolo_head

data_path = "Data"
annotation_path = os.path.join(data_path, "Airplanes_Annotations")
img_path = os.path.join(data_path, "Images")
batch_size = 32
epochs = 100


def model_tiny_YOLOv1():
    inputs = Input((448, 448, 3))
    x = original_backbone_network(inputs)
    x = Dense(539, activation='linear', name='connected_0')(x)
    outputs = ReshapeYOLO((7, 7, 11))(x)
    return outputs, inputs


class ReshapeYOLO(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeYOLO, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def call(self, inputs, **kwargs):
        S = [self.target_shape[0], self.target_shape[1]]
        C = 1
        B = 2
        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B
        # class prediction
        class_probability = K.reshape(
            inputs[:, :idx1], (K.shape(inputs)[0],) + tuple([S[0], S[1], C]))
        class_probability = K.softmax(class_probability)
        # confidence
        confidence = K.reshape(
            inputs[:, idx1:idx2], (K.shape(inputs)[0],) + tuple([S[0], S[1], B]))
        confidence = K.sigmoid(confidence)
        # boxes
        boxes = K.reshape(
            inputs[:, idx2:], (K.shape(inputs)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)
        # return np.array([class_probability, confidence, boxes])
        outputs = K.concatenate([class_probability, confidence, boxes])
        return outputs

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(ReshapeYOLO, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :1]  # ? * 7 * 7 * 1
    # 分类
    label_box = y_true[..., 1:5]  # ? * 7 * 7 * 4
    # BB1的坐标
    response_mask = y_true[..., 5]  # ? * 7 * 7
    # BB1的置信度
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :1]  # ? * 7 * 7 * 1
    # 分类
    predict_trust = y_pred[..., 1:3]  # ? * 7 * 7 * 2
    # BB1和BB2的置信度
    predict_box = y_pred[..., 3:]  # ? * 7 * 7 * 8
    # BB1和BB2的坐标

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = x_y_w_h_2_min_max(label_xy,
                                                   label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = x_y_w_h_2_min_max(predict_xy,
                                                       predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss


def _main():
    outputs, inputs = model_tiny_YOLOv1()
    input_shape = tuple(inputs.shape[1:])
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=yolo_loss, optimizer='adam')
    tf.keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    save_dir = 'checkpoints'
    weights_path = os.path.join(save_dir, 'airplanes-weights.hdf5')
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(weights_path):
        model = tf.keras.models.load_model(
            weights_path,
            custom_objects={
                "ReshapeYOLO": ReshapeYOLO,
                "yolo_loss": yolo_loss
            }
        )
    else:
        # model.load_weights('tiny-yolov1.hdf5', by_name=True)
        print('no train history')
    opt = Adam(lr=0.001)
    model.compile(loss=yolo_loss, optimizer=opt)

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    train_generator = SequenceForAirplanes(
        'train', annotation_path, img_path, input_shape, batch_size)
    validation_generator = SequenceForAirplanes(
        'val', annotation_path, img_path, input_shape, int(batch_size / 2))

    with tf.device('/gpu:0'):
        model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            # use_multiprocessing=True,
            workers=4,
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
    model.save_weights('airplanes-tiny-yolov1.hdf5')


if __name__ == '__main__':
    _main()
