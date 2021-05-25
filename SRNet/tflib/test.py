import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


'''

print('begin writing')
# cwd = 'D:/atry_data/train/'
cwd = 'D:/atry_data/test/'
classes = {'cover', 'stego'}
# writer = tf.io.TFRecordWriter("coverstego_train.tfrecords")
writer = tf.io.TFRecordWriter("coverstego_test.tfrecords")

for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name
        img = Image.open(img_path)
        img = img.resize((512, 512))
        # print(np.shape(img))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对label和image数据进行封装
        writer.write(example.SerializeToString())

writer.close()
print('end writing')

'''


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [512, 512, 1])
    img = tf.cast(img, tf.float32)*(1. / 255)-0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


# 上面为数据读取，下面为模型训练

batch_size = 20
epoch = 2


def one_hot(labels, label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(label_class)] for j in range(len(labels))])
    return one_hot_label


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def avg_pool_4x4(x):
    return tf.nn.avg_pool2d(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")


x = tf.placeholder(tf.float32, [batch_size, 512, 512, 1])
y_ = tf.placeholder(tf.float32, [batch_size, 2])

W_conv1 = weight_variable([3, 3, 1, 24])
b_conv1 = bias_variable([24])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_4x4(h_conv1)

W_conv2 = weight_variable([3, 3, 24, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_4x4(h_conv2)

W_conv3 = weight_variable([3, 3, 32, 48])
b_conv3 = bias_variable([48])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_4x4(h_conv3)

W_conv4 = weight_variable([3, 3, 48, 60])
b_conv4 = bias_variable([60])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = avg_pool_4x4(h_conv4)

W_conv5 = weight_variable([3, 3, 60, 72])
b_conv5 = bias_variable([72])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = avg_pool_4x4(h_conv5)

W_conv6 = weight_variable([3, 3, 72, 96])
b_conv6 = bias_variable([96])
h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
h_pool6 = avg_pool_4x4(h_conv6)

W_conv7 = weight_variable([3, 3, 96, 108])
b_conv7 = bias_variable([108])
h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
h_pool7 = avg_pool_4x4(h_conv7)

W_conv8 = weight_variable([3, 3, 108, 192])
b_conv8 = bias_variable([192])
h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8) + b_conv8)
h_pool8 = avg_pool_4x4(h_conv8)

reshape = tf.reshape(h_pool8, [batch_size, -1])
dim = reshape.get_shape()[1].value
W_fc1 = weight_variable([dim, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


img_train, label_train = read_and_decode("coverstego_train.tfrecords")
img_test, label_test = read_and_decode("coverstego_test.tfrecords")

img_train, label_train = tf.train.shuffle_batch([img_train, label_train],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)
img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                              batch_size=batch_size, capacity=2000,
                                              min_after_dequeue=1000)
init = tf.initialize_all_variables()
t_vars = tf.trainable_variables()
print(t_vars)
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    batch_idxs = int(2000/batch_size)
    for i in range(epoch):
        for j in range(batch_idxs):
            val, l = sess.run([img_train, label_train])
            l = one_hot(l, 2)
            _, acc = sess.run([train_step, accuracy], feed_dict={x: val[0:150], y_: l[0:150], keep_prob: 0.5})
            print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f]" % (i, j, batch_idxs, acc))
    val, l = sess.run([img_test, label_test])
    l = one_hot(l, 2)
    # print(l)
    y, acc = sess.run([y_conv, accuracy], feed_dict={x: val[0:150], y_: l[0:150], keep_prob: 1})
    # print(y)

    coord.request_stop()
    coord.join(threads)















