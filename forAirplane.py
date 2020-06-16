import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from data_sequence import SequenceForAirplanes
from models.model_tiny_yolov1 import model_tiny_YOLOv1, ReshapeYOLO
from yolo.yolo import yolo_loss


def _main():
    outputs, inputs = model_tiny_YOLOv1()
    input_shape = tuple(inputs.shape[1:])
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=yolo_loss, optimizer='adam')
    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     show_layer_names=False,
    #     rankdir='TB',
    #     expand_nested=False,
    #     dpi=96
    # )
    save_dir = 'checkpoints'
    weights_path = os.path.join(save_dir, 'weights.hdf5')
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
        # model.load_weights(weights_path, by_name=True)
        model = tf.keras.models.load_model(
            weights_path,
            custom_objects={
                "ReshapeYOLO": ReshapeYOLO,
                "yolo_loss": yolo_loss
            }
        )
    else:
        model.load_weights('tiny-yolov1.hdf5', by_name=True)
        print('no train history')
    opt = Adam(lr=0.001)
    model.compile(loss=yolo_loss, optimizer=opt)
    # epoch_file_path = 'checkpoints/epoch.txt'
    # try:
    #     with open(epoch_file_path, 'r') as f:
    #         now_epoch = int(f.read())
    #     epochs -= now_epoch
    # except IOError:
    #     print('no train history')
    # myCallback = callback.MyCallback()

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    # log_dir = 'logs'
    # tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
    #                          histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                          batch_size=batch_size,  # 用多大量的数据计算直方图
    #                          write_graph=True,  # 是否存储网络结构图
    #                          write_grads=True,  # 是否可视化梯度直方图
    #                          write_images=True,  # 是否可视化参数
    #                          embeddings_freq=0,
    #                          embeddings_layer_names=None,
    #                          embeddings_metadata=None)
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)

    # 数据生成器
    train_generator = SequenceForAirplanes(
        'train', data_path, input_shape, batch_size)
    validation_generator = SequenceData(
        'val', data_path, input_shape, batch_size)
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
    model.save_weights('my-tiny-yolov1.hdf5')


if __name__ == '__main__':
    _main()
