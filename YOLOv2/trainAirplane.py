import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint

from VOC2007.yolo.yolo import yolo_head, iou
from YOLOv1.trainAirplane import ReshapeYOLO, annotation_path, img_path
from data_sequence import SequenceForAirplanes
from utils import X_Y_W_H_To_Min_Max

image_size = 448


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

    label_xy, label_wh = yolo_head(_label_box, img_size=image_size)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = X_Y_W_H_To_Min_Max(label_xy,
                                                    label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box, img_size=image_size)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = X_Y_W_H_To_Min_Max(predict_xy,
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

    label_xy, label_wh = yolo_head(_label_box, img_size=image_size)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box, img_size=image_size)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / image_size)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / image_size)
    box_loss = K.sum(box_loss)
    # K.print_tensor("Conf Loss")
    # K.print_tensor(confidence_loss)
    # K.print_tensor("Class Loss")
    # K.print_tensor(class_loss)
    # K.print_tensor("Box Loss")
    # K.print_tensor(box_loss)
    loss = confidence_loss + class_loss + box_loss

    return loss


def yolo_loss_v2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # V2专用的loss计算函数
    # y_true除尺寸维度以外不变（边长从7提升到28）
    # y_pred格式如下：
    # 第1到第36位（0:36）：锚框中心坐标偏移量、锚框高宽缩放系数
    # 第37到45位（36:45）：9个锚框的置信度
    # 第46到54位（45:54）：9个锚框的分类（单分类为1*9=9，二分类为2*9=18）
    label_class = y_true[..., :1]  # ? * 28 * 28 * 1
    # 分类
    label_box = y_true[..., 1:5]  # ? * 28 * 28 * 4
    # BB1的坐标
    response_mask = K.expand_dims(y_true[..., 5], axis=-1)  # ? * 28 * 28 * 1
    # BB1的置信度

    predict_class = y_pred[..., 45:]  # ? * 28 * 28 * 9
    # 分类
    predict_trust = y_pred[..., 36:45]  # ? * 28 * 28 * 9
    # BB1和BB2的置信度
    predict_box = y_pred[..., :36]  # ? * 28 * 28 * 36
    # BB1和BB2的坐标

    _label_box = K.reshape(label_box, [-1, 28, 28, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 28, 28, 9, 4])

    label_xy, label_wh = yolo_head(_label_box, img_size=image_size)
    # ? * 28 * 28 * 1 * 2, ? * 28 * 28 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)
    # ? * 28 * 28 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)
    # ? * 28 * 28 * 1 * 1 * 2
    label_xy_min, label_xy_max = X_Y_W_H_To_Min_Max(
        label_xy,
        label_wh
    )  # ? * 28 * 28 * 1 * 1 * 2, ? * 28 * 28 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(
        _predict_box,
        img_size=image_size,
        forTrue=False
    )  # ? * 28 * 28 * 9 * 2, ? * 28 * 28 * 9 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 28 * 28 * 9 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 28 * 28 * 9 * 1 * 2
    predict_xy_min, predict_xy_max = X_Y_W_H_To_Min_Max(
        predict_xy,
        predict_wh
    )
    # ? * 28 * 28 * 9 * 1 * 2, ? * 28 * 28 * 9 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 28 * 28 * 9 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    # K.print_tensor("")
    # K.print_tensor("no_object_loss")
    # K.print_tensor(K.sum(no_object_loss))
    # K.print_tensor("object_loss")
    # K.print_tensor(K.sum(object_loss))

    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 28, 28, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 28, 28, 9, 4])

    label_xy, label_wh = yolo_head(_label_box, img_size=image_size)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(
        _predict_box,
        img_size=image_size,
        forTrue=True
    )  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / image_size)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / image_size)
    box_loss = K.sum(box_loss)
    # K.print_tensor("")
    # K.print_tensor("Conf Loss")
    # K.print_tensor(confidence_loss)
    # K.print_tensor("Class Loss")
    # K.print_tensor(class_loss)
    # K.print_tensor("Box Loss")
    # K.print_tensor(box_loss)
    loss = confidence_loss + class_loss + box_loss

    return loss


def ChangeInputSize():
    save_dir = '../checkpoints'
    model_path = os.path.join(save_dir, 'YOLOv1.hdf5')
    previous_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ReshapeYOLO": ReshapeYOLO,
            "yolo_loss": yolo_loss
        }
    )
    y = Input((448, 448, 3))
    x = y
    for layer in previous_model.layers[1:20]:
        x = layer(x)
    x = Dense(539, activation='linear', name='connected_0')(x)
    x = ReshapeYOLO((7, 7, 11))(x)
    v2 = Model(inputs=y, outputs=x)
    v2.compile(loss=yolo_loss, optimizer='adam')
    tf.keras.utils.plot_model(
        v2,
        to_file='../model.png',
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    v2.save('..\\TrainedModels\\YOLOv2-448X448.hdf5')
    print("Done")


def UseAnchor():
    k = 9
    model_path = "..\\TrainedModels\\YOLOv2-448X448.hdf5"
    previous_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ReshapeYOLO": ReshapeYOLO,
            "yolo_loss": yolo_loss
        }
    )
    x = None
    for i in range(len(previous_model.layers)):
        if type(previous_model.layers[i]) == tf.keras.layers.Flatten:
            x = previous_model.layers[i - 2].output
    convolution_3x3 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        name="3x3"
    )(x)
    output_delta = Conv2D(
        filters=4 * k,
        kernel_size=(1, 1),
        activation="sigmoid",
        kernel_initializer="uniform",
        name="deltas1"
    )(convolution_3x3)
    output_trust = Conv2D(
        filters=1 * k,
        kernel_size=(1, 1),
        activation="sigmoid",
        kernel_initializer="uniform",
        name="scores1"
    )(convolution_3x3)
    output_class = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        activation="softmax",
        kernel_initializer="uniform",
        name="scores2"
    )(convolution_3x3)
    result = tf.concat([output_delta, output_trust, output_class], axis=-1)

    v2 = Model(inputs=previous_model.input, outputs=result)
    v2.compile(loss=yolo_loss_v2, optimizer='adam')
    tf.keras.utils.plot_model(
        v2,
        to_file='../model.png',
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    v2.save('..\\TrainedModels\\YOLOv2-448X448-RPN.hdf5')
    print("Done")


def AddBatchNorm(BN_count=100):
    model_path = "..\\TrainedModels\\YOLOv2-448X448.hdf5"

    previous_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ReshapeYOLO": ReshapeYOLO,
            "yolo_loss": yolo_loss
        }
    )
    x = None
    for layer in previous_model.layers:
        if x is not None:
            x = layer(x)
            if type(layer) == tf.keras.layers.Conv2D and BN_count > 0:
                x = BatchNormalization()(x)
                BN_count -= 1
        elif type(layer) == tf.keras.layers.Conv2D and BN_count > 0:
            x = layer.output
            x = BatchNormalization()(x)
            BN_count -= 1
        print(type(layer))
    v2 = Model(inputs=previous_model.input, outputs=x)
    v2.compile(loss=yolo_loss, optimizer='adam')
    tf.keras.utils.plot_model(
        v2,
        to_file='../model.png',
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    v2.save('..\\TrainedModels\\YOLOv2-448X448-BN.hdf5')
    print("Done")


def trainV2():
    weights_path = '..\\TrainedModels\\YOLOv2-448X448-RPN.hdf5'

    # region With BN
    # weights_path = '..\\TrainedModels\\YOLOv2-448X448-BN.hdf5'
    # endregion

    # region Continue
    # save_dir = '../checkpoints'
    # weights_path = os.path.join(save_dir, 'YOLOv2.hdf5')
    # endregion

    # region 448*448 Only
    # weights_path = "..\\TrainedModels\\YOLOv2-448X448.hdf5.hdf5"
    # endregion

    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={
            "ReshapeYOLO": ReshapeYOLO,
            "yolo_loss": yolo_loss,
            "yolo_loss_v2": yolo_loss_v2
        }
    )
    tf.keras.utils.plot_model(
        model,
        to_file='../model.png',
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    model.compile(loss=yolo_loss_v2, optimizer=Adam(lr=0.0001))
    input_shape = tuple(model.input.shape[1:])
    train_generator = SequenceForAirplanes(
        'train', annotation_path, img_path, input_shape, 1)
    validation_generator = SequenceForAirplanes(
        'val', annotation_path, img_path, input_shape, 1)
    save_dir = '../checkpoints'
    weights_path = os.path.join(save_dir, 'YOLOv2.hdf5')
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="../TensorBoard",
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True
    )
    with tf.device('/gpu:0'):
        model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=100,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[checkpoint, tb_callback, early_stopping],
            workers=4,
            verbose=1
        )


if __name__ == '__main__':
    # ChangeInputSize()
    trainV2()
    # UseAnchor()
