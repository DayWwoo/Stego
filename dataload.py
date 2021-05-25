import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


'''cwd = 'D:/Dataload/J_0.1/'
writer = tf.io.TFRecordWriter("J_0.1.tfrecords")'''
'''cwd = 'D:/Dataload/J_0.2/'
writer = tf.io.TFRecordWriter("J_0.2.tfrecords")'''
'''cwd = 'D:/Dataload/J_0.3/'
writer = tf.io.TFRecordWriter("J_0.3.tfrecords")'''
'''cwd = 'D:/Dataload/J_0.4/'
writer = tf.io.TFRecordWriter("J_0.4.tfrecords")'''
'''cwd = 'D:/Dataload/U_0.1/'
writer = tf.io.TFRecordWriter("U_0.1.tfrecords")'''
'''cwd = 'D:/Dataload/U_0.2/'
writer = tf.io.TFRecordWriter("U_0.2.tfrecords")'''
'''cwd = 'D:/Dataload/U_0.3/'
writer = tf.io.TFRecordWriter("U_0.3.tfrecords")'''
cwd = 'D:/Dataload/U_0.4/'
writer = tf.io.TFRecordWriter("U_0.4.tfrecords")

classes = {'cover', 'stego'}
for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name
        img = Image.open(img_path)
        img = img.resize((512, 512))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img_img = tf.decode_raw(features['img_raw'], tf.uint8)
    img_img = tf.reshape(img_img, [512, 512, 1])
    img_img = tf.cast(img_img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img_img, label
