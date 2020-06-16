import os
import tensorflow as tf

from YOLOv1.trainAirplane import ReshapeYOLO, yolo_loss

save_dir = '../checkpoints'
weights_path = os.path.join(save_dir, 'airplanes-weights.hdf5')

model = tf.keras.models.load_model(
    weights_path,
    custom_objects={
        "ReshapeYOLO": ReshapeYOLO,
        "yolo_loss": yolo_loss
    }
)


print("Done")