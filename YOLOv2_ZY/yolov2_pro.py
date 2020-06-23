import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, \
    MaxPooling2D, BatchNormalization, \
    Flatten, Dense, Lambda, LeakyReLU, concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# For kangaroo dataset
YOLO_ANCHORS = np.array(
    [(9.42263202, 9.18164528), (2.04368121, 2.28255682), (5.80786578, 5.43905892),
     (2.95958038, 7.13051561), (5.3929678, 9.97381383)], np.float32)

YOLO_ANCHORS /= 13

# loss
object_scale = 5.0
no_object_scale = 1.0
class_scale = 1.0
coordinates_scale = 1.0

num_max_box = 50

# Create a dictionary describing the features.
image_feature_description = {
    'filename': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'classes': tf.io.VarLenFeature(tf.int64),
    'x_mins': tf.io.VarLenFeature(tf.float32),
    'y_mins': tf.io.VarLenFeature(tf.float32),
    'x_maxes': tf.io.VarLenFeature(tf.float32),
    'y_maxes': tf.io.VarLenFeature(tf.float32),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

anchors = YOLO_ANCHORS
BATCH_SIZE = 8


def parse_tf_example(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)

    x_train = tf.image.decode_jpeg(parsed_example['image_raw'], channels=3)
    # x_train = tf.image.resize(x_train, (416, 416)) # yolov2 - input size
    # x_train /= 255.

    width = tf.cast(parsed_example['width'], tf.float32)
    height = tf.cast(parsed_example['height'], tf.float32)

    labels = tf.sparse.to_dense(parsed_example['classes'])
    labels = tf.cast(labels, tf.float32)

    xmin = tf.sparse.to_dense(parsed_example['x_mins'])
    ymin = tf.sparse.to_dense(parsed_example['y_mins'])
    xmax = tf.sparse.to_dense(parsed_example['x_maxes'])
    ymax = tf.sparse.to_dense(parsed_example['y_maxes'])

    y_train = tf.stack([xmin, ymin,
                        xmax, ymax,
                        labels], axis=1)
    print('def parse_tf_example(example_proto):')
    return x_train, y_train


def preprocess_data(x_train, y_train, anchors, size=448):
    """preprocess the data"""
    # Resize the image data.
    x_train = tf.image.resize(x_train, (size, size))
    x_train /= 255.

    # Box preprocessing.
    # Origin boxes: (xmin, ymi, xmax, ymax, classes)
    boxes = tf.reshape(y_train, [-1, 5])
    boxes_xy = 0.5 * (boxes[:, 2:4] + boxes[:, 0:2])
    boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]
    # New boxes: (x_center, y_center, w, h, classes)
    new_boxes = tf.concat((boxes_xy, boxes_wh, boxes[:, 4:]), axis=-1)

    """Add zero pad for training 
    
    paddings = [[row_top, row_bottom], [col_left, col_right]]
    """
    paddings = [[0, num_max_box - tf.shape(new_boxes)[0]], [0, 0]]
    new_y_train = tf.pad(new_boxes, paddings)

    new_y_train = transform_bboxes_for_output(new_y_train, anchors)

    return x_train, new_y_train


# ## Preprocess the dataset
# 
# - images: resize to (416, 416)
# - labels: `(xmin, ymi, xmax, ymax, class)` -> `(num_max_boxes, (x_center, y_center, w, h, class))`
# 
# - max ture boxes: 50
def transform_bboxes_for_output(true_boxes, anchors, image_size=448):
    grid_size = image_size // 32

    """Area of anchor."""
    # anchors shape: (5, 2)
    # Note: normalization anchors to 0~1 (anchors / 13)
    #       -> anchors and boxes_wh are moved to origin point
    #       -> we can conveniently find the minimum 
    #          between anchors and boxes_wh to find the intersection area.
    anchor_area = anchors[..., 0] * anchors[..., 1]  # (5,)

    """ Area of boxes."""
    # true_boxes shape: (num_boxes, (x, y, w, h, classes))
    boxes = tf.reshape(true_boxes, [-1, 5])
    boxes_xy = boxes[..., 0:2]
    boxes_wh = boxes[..., 2:4]  # (N, 2)

    # expand dimension for compare with anchor
    boxes_wh = tf.tile(tf.expand_dims(boxes_wh, -2),
                       (1, tf.shape(anchors)[0], 1))  # (N, 5, 2)
    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]  # (N, 5)

    """Find IOU between box shifted to origin and anchor box."""
    intersection_wh = tf.minimum(boxes_wh, anchors)  # (N, 5, 2)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  # (N, 5)
    iou = intersection_area / (boxes_area + anchor_area - intersection_area)  # (N, 5)

    """Find the best iou."""
    best_anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    best_anchor_idx = tf.expand_dims(best_anchor_idx, -1)  # (N, 1)

    N = tf.shape(true_boxes)[0]
    new_y_train = tf.zeros((grid_size, grid_size, tf.shape(anchors)[0], tf.shape(true_boxes)[1] + 1), tf.float32)

    # Find which grid includes the center of object
    grid_xy = tf.cast(boxes_xy * grid_size, tf.int32)
    best_anchor_idx = tf.cast(best_anchor_idx, tf.int32)

    indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

    for i in range(N):
        if boxes[i][0] != 0.:
            indices = indices.write(i, [grid_xy[i][1], grid_xy[i][0], best_anchor_idx[i][0]])
            updates = updates.write(i, [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], 1, boxes[i][4]])
            # tf.print("indices:\n", indices.stack())
    # tf.print("updates:\n", updates.stack())

    new_y_train = tf.tensor_scatter_nd_update(
        new_y_train, indices.stack(), updates.stack()
    )

    return new_y_train


# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)


def YOLOV2(num_anchors=5, num_classes=80):
    x = inputs = tf.keras.Input(shape=(448, 448, 3))

    # Layer 1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(x)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    conv13 = x  #

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18 - darknet
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    darknet = LeakyReLU(alpha=0.1)(x)

    ##################################################################################

    # Layer 19
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(darknet)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    conv20 = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    conv21 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(conv13)
    conv21 = BatchNormalization(name='norm_21')(conv21)
    conv21 = LeakyReLU(alpha=0.1)(conv21)
    conv21_reshaped = Lambda(space_to_depth_x2, name='space_to_depth')(conv21)

    x = concatenate([conv21_reshaped, conv20])

    # Layer 22
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23 - Output
    x = Conv2D(num_anchors * (num_classes + 5), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    outputs = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                              num_anchors, num_classes + 5)))(x)

    return Model(inputs, outputs)


# ### Load pre-trained model
def load_darknet_weights(model, weights_file):
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen = np.fromfile(wf, dtype=np.int32, count=4)
        conv_idx = 0
        num_conv = 23  # 總共有23層的卷積層

        for i, layer in enumerate(model.layers):
            if not layer.name.startswith('conv'):
                continue
            conv_idx = int(layer.name[5:])

            # BatchNormalization layer
            batch_norm = None
            if conv_idx < num_conv:
                batch_norm = model.get_layer('norm_' + str(conv_idx))  # 取得BatchNormalization層
            print("{}/{} {}".format(
                model.name, layer.name, 'bn' if batch_norm else 'bias'), end=', ')

            filters = layer.filters
            kerner_size = layer.kernel_size[0]
            input_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, input_dim, height, width)
            conv_shape = (filters, input_dim, kerner_size, kerner_size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))

            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

            print("Completed!")
    print("End!")


def yolo_bboxes(pred, anchors, classes):
    """ YOLO - bounding box formula
    
    bx = sigmoid(tx) + cx
    by = sigmoid(ty) + cy
    bw = pw * exp^(tw)
    bh = ph * exp^(th)
    Pr(obj) * IOU(b, object) = sigmoid(to) # confidence
    
    (tx, ty, tw, th, to) are the output of the model.
    """
    # pred: (batch_size, grid, grid, anchors, (tx, ty, tw, th, conf, ...classes))
    grid_size = tf.shape(pred)[1]

    box_xy = tf.sigmoid(pred[..., 0:2])
    box_wh = tf.exp(pred[..., 2:4])
    box_confidence = tf.sigmoid(pred[..., 4:5])
    box_class_probs = tf.sigmoid(pred[..., 5:])

    # box_xy: (grid_size, grid_size, num_anchors, 2)
    # grid: (grdid_siez, grid_size, 1, 2)
    #       -> [0,0],[0,1],...,[0,12],[1,0],[1,1],...,[12,12]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = box_wh * anchors

    return box_xy, box_wh, box_confidence, box_class_probs


def YoloLoss(anchors, classes=80, object_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        """"1. transform all pred outputs."""
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, conf, ...cls))
        pred_xy, pred_wh, pred_confidence, pred_class_probs = yolo_bboxes(
            y_pred, anchors, classes)

        """2. transform all true outputs."""
        # y_true: (batch_size, grid, grid, anchors, (x, y, w, h, conf, cls))
        true_xy, true_wh, true_confidence, true_class = tf.split(
            y_true, (2, 2, 1, classes), axis=-1)
        true_boxes = tf.concat([true_xy, true_wh], axis=-1)

        """3. true_box: remove noobject cell."""
        # true_confidence: cell has object or not
        #                 0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        # true_confidence_mask: [conv_height, conv_width, num_anchors]
        true_conf_mask = tf.squeeze(true_confidence, -1)

        """4. Calculate `coordinate loss`."""
        xy_loss = coordinates_scale * true_conf_mask * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = coordinates_scale * true_conf_mask * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        """5. Calculate `classification loss`."""
        # square(one_hot(true_class) - pred_class_probs)
        classification_loss = class_scale * true_conf_mask * tf.keras.losses.sparse_categorical_crossentropy(
            true_class, pred_class_probs)

        """6. Calculate `confidence loss.`
        
        1. Find the IOU score of each predicted box with each ground truth box.
        2. Find the Best IOU scores.
        3. A confidence detector that this cell has object if IOU > threshold otherwise no object.
        4. Calculate confidence loss.
        """
        ############################################################################################################
        # Reshape true_box: (N, grid, grid, num_anchors, 4) to (N, num_true_boxes, 4)     
        true_boxes_flat = tf.boolean_mask(true_boxes, tf.cast(true_conf_mask, tf.bool))

        # broadcast shape: (N, grid, grid, num_anchors, num_true_boxes, (x, y, w, h))
        true_boxes = tf.expand_dims(true_boxes, -2)  # (N, 13, 13, 5, 1, 4)
        true_boxes_flat = tf.expand_dims(true_boxes_flat, 0)  # (1, num_true_boxes, 4)
        new_shape = tf.broadcast_dynamic_shape(tf.shape(true_boxes),
                                               tf.shape(true_boxes_flat))  # (N, 13, 13, 5, num_true_boxes, 4)

        # reshape: (batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params)
        true_boxes = tf.broadcast_to(true_boxes, new_shape)
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]  # (N, 13, 13, 5, num_true_boxes, 2)

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        # Expand pred (x,y,w,h) to allow comparison with ground truth.
        # (batch, conv_height, conv_width, num_anchors, 1, box_params)
        pred_xy = tf.expand_dims(pred_xy, 4)
        pred_wh = tf.expand_dims(pred_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersection_mins = tf.maximum(pred_mins, true_mins)
        intersection_maxes = tf.minimum(pred_maxes, true_maxes)
        intersection_wh = tf.maximum(intersection_maxes - intersection_mins, 0.)
        intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]  # (-1, 13, 13, 5, num_true_boxes)

        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        """6-1. Calculate IOU scores for each location."""
        union_areas = pred_areas + true_areas - intersection_areas
        iou_scores = intersection_areas / union_areas  # (-1, 13, 13, 5, num_true_boxes)

        """6-2. Best IOU scores."""
        best_ious = tf.reduce_max(iou_scores, axis=4)
        best_ious = tf.expand_dims(best_ious, axis=-1)  # (-1, 13, 13, 5, 1)

        """6-3. Confidence mask (In YAD2K: object_detections)."""
        pred_conf_mask = tf.cast(best_ious > object_thresh, tf.float32)  # (-1, 13, 13, 5, 1)

        """6-4. Confidence loss."""
        no_object_weights = (no_object_scale * (1 - pred_conf_mask) * (1 - true_confidence))
        no_objects_loss = no_object_weights * tf.square(0 - pred_confidence)

        objects_loss = object_scale * true_confidence * tf.square(1 - pred_confidence)

        confidence_loss = objects_loss + no_objects_loss
        confidence_loss = tf.reduce_sum(confidence_loss, -1)
        ############################################################################################################

        # 7. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss_sum = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss_sum = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        confidence_loss_sum = tf.reduce_sum(confidence_loss, axis=(1, 2, 3))
        classification_loss_sum = tf.reduce_sum(classification_loss, axis=(1, 2, 3))

        '''
        tf.print(xy_loss_sum)
        tf.print(wh_loss_sum)
        tf.print(confidence_loss_sum)
        tf.print(classification_loss_sum)
        '''

        return 0.5 * (xy_loss_sum + wh_loss_sum + confidence_loss_sum + classification_loss_sum)

    return yolo_loss


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


def yolo_eval(yolo_outputs, image_shape=(448, 448), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
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


def main():
    # path = "/home/mitom/ASR/workspace/yolo2-tf2"
    # os.chdir(path)
    # load data from local
    raw_train_dataset = tf.data.TFRecordDataset('./data/airplane/train_airplane.tfrecord')
    #     raw_train_dataset = tf.data.TFRecordDataset('./data/train_voc.tfrecord')

    for raw_train_data in raw_train_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_train_data.numpy())

    train_dataset = raw_train_dataset.map(parse_tf_example)

    # ### 2. Show the data
    # - (xmin, ymin, xmax, ymax): already normalization to 0~1.
    for i, (x, y) in enumerate(train_dataset.take(1)):
        img = x.numpy()
        boxes = y.numpy()
        height, width = tf.shape(img)[0].numpy(), tf.shape(img)[1].numpy()

        for box in boxes:
            left = (box[0] * width).astype('int32')
            top = (box[1] * height).astype('int32')
            right = (box[2] * width).astype('int32')
            bottom = (box[3] * height).astype('int32')
            # cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)

        f, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)

    # Use deep copy()
    anchors = YOLO_ANCHORS
    train_dataset_2 = train_dataset.map(lambda x, y: (
        preprocess_data(x, y, anchors)))

    for x, y in train_dataset_2.take(1):
        print(i, x.shape, y.shape)
        print(y[7][8])
        print(y[7][6])

        img = x.numpy()
        img = np.array(img)
        true_boxes = y.numpy()

        # Custom (rgb) grid color and object color
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        grid_color = [255, 255, 255]

        # Plot grid box
        # Modify the image to include the grid
        dx, dy = (32, 32)  # downsamples the input by 32
        img[:, ::dy, :] = grid_color
        img[::dx, :, :] = grid_color

        # Plot anchor box center 
        anchor_boxes_xy = [(8 * 32, 7 * 32), (6 * 32, 7 * 32)]
        for i, box_xy in enumerate(anchor_boxes_xy):
            cv2.circle(img, box_xy, 10, colors[i], -1)

        # Plot anchor box
        anchor_box_wh = YOLO_ANCHORS[2] * 448
        anchor_box_wh_half = anchor_box_wh / 2.
        bbox_mins = anchor_boxes_xy - anchor_box_wh_half
        bbox_maxes = anchor_boxes_xy + anchor_box_wh_half

        for i in range(len(bbox_mins)):
            cv2.rectangle(img, (int(bbox_mins[i][0]), int(bbox_mins[i][1])),
                          (int(bbox_maxes[i][0]), int(bbox_maxes[i][1])), colors[i], 2)

        '''
        # Plot true box
        true_xy = true_boxes[..., 0:2] * 416
        ture_wh = true_boxes[..., 2:4] * 416
        true_wh_half = ture_wh / 2.
        true_mins = tf.cast(true_xy - true_wh_half, tf.int32)
        true_maxes = tf.cast(true_xy + true_wh_half, tf.int32)
        true_bbox = tf.concat((true_mins, true_maxes), axis=-1)
        
        for grid_y in true_bbox:
            for grid_x in grid_y:
                for box in grid_x:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        '''

        f, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        f.subplots_adjust(hspace=.2, wspace=.05)

        ax1.imshow(img)

    # ## Prepare the data for training
    train_ds = train_dataset_2.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    model = YOLOV2(num_classes=1)
    print(model.summary())

    weights_file = 'model/pre_yolov2.weights'
    load_darknet_weights(model, weights_file)

    layer = model.layers[-2]  # conv_23
    weights = layer.get_weights()
    print(weights)

    new_kernel = np.random.normal(size=weights[0].shape) / (14 * 14)
    new_bias = np.random.normal(size=weights[1].shape) / (14 * 14)
    print(new_kernel)
    print(new_bias)

    layer.set_weights([new_kernel, new_bias])  # 重新初始化權重

    # optimizer = tf.keras.optimizers.Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizer = tf.keras.optimizers.Adam(lr=0.1e-4)
    yolo_loss = YoloLoss(anchors, classes=1)

    model.compile(optimizer=optimizer, loss=yolo_loss,
                  run_eagerly=True)

    model.fit(train_ds, epochs=50)  # 300

    # Save
    # tf.saved_model.save(model, path + "/model_data")
    model.save("model/yolov2_airplane.h5")

    # Load
    # model = tf.keras.models.load_model("model_data/yolov2.h5", custom_objects={'yolo_loss': yolo_loss})

    class_names = ['airplane']

    # ### Start detect
    img_filepath = "./data/airplane/images/airplane_001.jpg"
    print(img_filepath)

    # 使用OpenCV讀入圖像
    images = cv2.imread(img_filepath)  # 載入圖像
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    # 進行圖像輸入的前處理
    input_image = cv2.resize(images, (448, 448))  # 修改輸入圖像大小來符合模型的要求
    input_image = input_image / 255.  # 進行圖像歸一處理
    input_image = np.expand_dims(input_image, 0)  # 增加 batch dimension

    # 進行圖像偵測
    yolo_outputs = model.predict(input_image)

    image = cv2.resize(images, (448, 448))
    scores, boxes, classes = yolo_eval(yolo_outputs, score_threshold=0.3)

    # Draw bounding boxes on the image file
    image = draw_boxes(image, scores, boxes, classes, class_names)

    # Save
    cv2.imwrite("./model_data/output.jpg", image)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)


if __name__ == '__main__':
    gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpu_list)
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)
    main()
