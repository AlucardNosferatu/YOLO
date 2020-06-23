'''
Created on Jun 21, 2020

@author: monky
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import xml.etree.ElementTree as ET
from datetime import datetime

import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data")
DATA_SET_PATH = os.path.join(DATA_PATH, "airplane")
ANNOTATIONS_PATH = os.path.join(DATA_SET_PATH, "annots")
IMAGES_PATH = os.path.join(DATA_SET_PATH, "images")
CLASSES = ['airplane']


def process_image(image_file):
    """Decode image at given path."""
    # return <class 'tf.Tensor'>
    image_string = tf.io.read_file(image_file)
    # return <class 'bytes'>
    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        # image_data = tf.image.resize(image_data, [300, 300])
        # image_data /= 255.0 # normalize to [0, 1] range
        return 0, image_string, image_data
    except tf.errors.InvalidArgumentError:
        print('{}: Invalid JPEG data or crop window'.format(image_file))
        return 1, image_string, None


def parse_annot(annot_file, image):
    image_info = {}
    image_info_list = []
    img = cv2.imread(image)

    width = img.shape[0]
    height = img.shape[1]
    file_name = image.split('/')[-1]
    depth = 3

    xmin, ymin, xmax, ymax = [], [], [], []

    box_num = 0

    with open(annot_file, 'r') as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        box_num = int(reader_list[0][0])
        for row in reader_list[1:]:
            row_str = row[0].split()
            coordinate = [int(i) for i in row_str]
            xmin.append(coordinate[0] / width)
            ymin.append(coordinate[1] / height)
            xmax.append(coordinate[2] / width)
            ymax.append(coordinate[3] / height)

    classes = [0 for x in range(box_num)]

    image_info['filename'] = file_name
    image_info['width'] = width
    image_info['height'] = height
    image_info['depth'] = depth
    image_info['class'] = classes
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax

    image_info_list.append(image_info)

    return image_info_list


def convert_voc_to_tf_example(image_string, image_info_list):
    """Convert Pascal VOC ground truth to TFExample protobuf."""
    for info in image_info_list:
        filename = info['filename']
        width = info['width']
        height = info['height']
        depth = info['depth']
        classes = info['class']
        xmin = info['xmin']
        ymin = info['ymin']
        xmax = info['xmax']
        ymax = info['ymax']

    if isinstance(image_string, type(tf.constant(0))):
        encoded_image = [image_string.numpy()]
    else:
        encoded_image = [image_string]

    base_name = [tf.compat.as_bytes(os.path.basename(filename))]

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'x_mins': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'y_mins': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'x_maxes': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'y_maxes': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }))

    return example  # example.SerializeToString()


def main():
    images = sorted(glob.glob(os.path.join(IMAGES_PATH, 'airplane_*.jpg')))
    annots = sorted(glob.glob(os.path.join(ANNOTATIONS_PATH, 'airplane_*.csv')))
    train_file = 'data/airplane/train_airplane.tfrecord'
    counter = 0
    skipped = 0

    with tf.io.TFRecordWriter(train_file) as writer:
        for image, annot in (zip(images, annots)):
            error, image_string, image_data = process_image(image)
            image_info_list = parse_annot(annot, image)
            if not error:
                # convert voc to `tf.Example`
                example = convert_voc_to_tf_example(image_string, image_info_list)

                # write the `tf.example` message to the TFRecord files
                writer.write(example.SerializeToString())
                counter += 1
                print('{} : Processed {:d} of {:d} images.'.format(
                    datetime.now(), counter, len(images)))
            else:
                skipped += 1
                print('{} : Skipped {:d} of {:d} images.'.format(
                    datetime.now(), skipped, len(images)))

    print('{} : Wrote {} images to {}'.format(
        datetime.now(), counter, train_file))


if __name__ == '__main__':
    main()
