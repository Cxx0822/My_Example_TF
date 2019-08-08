# -*-coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image

import yaml

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader)

image_train_dir = my_data['train_dir']
image_val_dir = my_data['val_dir']
train_labels = my_data['train_txt'] 
val_labels = my_data['val_txt'] 

resize_height = my_data['resize_height']
resize_width = my_data['resize_width']
shuffle = my_data['shuffle']
log = my_data['log']


train_record_output = 'dataSet/record/train{}.tfrecords'.format(resize_height)
val_record_output = 'dataSet/record/val{}.tfrecords'.format(resize_height)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_example_nums(tf_records_filenames):
    nums = 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums


def show_image(title, image):
    plt.imshow(image)
    plt.axis('on')    
    plt.title(title)  
    plt.show()


def load_labels_file(filename, labels_num=1, shuffle=False):
    images = []
    labels = []
    with open(filename) as f:
        lines_list = f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line = lines.rstrip().split(' ')
            label = []
            for i in range(labels_num):
                label.append(int(line[i + 1]))
            images.append(line[0])
            labels.append(label)
    return images, labels


def read_image(filename, resize_height, resize_width, normalization=False):
    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape) == 2:
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)

    if normalization:
        rgb_image = rgb_image / 255.0

    return rgb_image


def get_batch_images(images, labels, batch_size, labels_nums, one_hot=False, shuffle=False, num_threads=1):
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images, labels],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue,
                                                            num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch, labels_batch


def read_records(filename, resize_height, resize_width, type=None):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)

    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)

    tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3]) 

    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type == 'normalization':
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  
    elif type == 'centralization':
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5 

    return tf_image, tf_label


def create_records(image_dir, file, output_record_dir, resize_height, resize_width, shuffle, log=5):
    images_list, labels_list = load_labels_file(file, 1, shuffle)

    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path = os.path.join(image_dir, images_list[i])
        if not os.path.exists(image_path):
            print('Err:no image', image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)
        image_raw = image.tostring()
        if i % log == 0 or i == len(images_list) - 1:
            print('------------processing:%d-th------------' % (i))
            print('current image_path=%s' % (image_path), 'shape:{}'.format(image.shape), 'labels:{}'.format(labels))
        label = labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def create_records_train(): 
    create_records(image_train_dir, train_labels, train_record_output, resize_height, resize_width, shuffle, log)
    train_nums = get_example_nums(train_record_output)
    print("save train example nums={}".format(train_nums))


def create_records_val(): 
    create_records(image_val_dir, val_labels, val_record_output, resize_height, resize_width, shuffle, log)
    val_nums = get_example_nums(val_record_output)
    print("save val example nums={}".format(val_nums))


def disp_records(record_file, resize_height, resize_width, show_nums=4):
    tf_image, tf_label = read_records(record_file, resize_height, resize_width, type='normalization')
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image, label = sess.run([tf_image, tf_label])  
            print('shape:{},tpye:{},labels:{}'.format(image.shape, image.dtype, label))
            show_image("image:%d" % (label), image)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    create_records_train()
    create_records_val()
    # disp_records(train_record_output, resize_height, resize_width)