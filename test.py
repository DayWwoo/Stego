import time
import tensorflow as tf
import conv
import train
import numpy as np
import os
import dataload
import dctr

batch_size = 5
epoch = 5


def one_hot(labels, label_class):
    one_hot_label = np.array([[int(m == int(labels[n])) for m in range(label_class)] for n in range(len(labels))])
    return one_hot_label


x = tf.placeholder(tf.float32, [batch_size, 512, 512, 1])
y_ = tf.placeholder(tf.float32, [batch_size, 2])

y_conv = dctr.dct(x)
y_conv = conv.inference(y_conv, None)
# y_conv = tf.nn.dropout(y_conv, 0.5)
# y_conv = tf.nn.softmax(y_conv)
'''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y_conv)
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
mean_loss = tf.reduce_mean(tf.cast(cross_entropy, tf.float32))
tf.add_to_collection('losses', mean_loss)
loss = tf.add_n(tf.get_collection('losses'))
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
'''
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.name_scope('acct'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acct', accuracy)
merged_summary_op = tf.summary.merge_all()

img_train, label_train = dataload.read_and_decode("J_0.4.tfrecords")
img_test, label_test = dataload.read_and_decode("U_0.4.tfrecords")
img_train, label_train = tf.train.shuffle_batch([img_train, label_train],
                                                batch_size=batch_size, capacity=2500, min_after_dequeue=1250)
img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                              batch_size=batch_size, capacity=2500, min_after_dequeue=1250)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("D:/python/PycharmProjects/ProjectStego/test/log", tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    batch_idxs = int(2500 / batch_size)
    for i in range(epoch):
        for j in range(batch_idxs):
            val, la = sess.run([img_test, label_test])
            la = one_hot(la, 2)
            ckpt = tf.train.get_checkpoint_state(train.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                acc = sess.run(accuracy, feed_dict={x: val, y_: la})
                print("Epoch:%4d %4d/%4d, after training steps, accuracy=%.8f" % (i, j, batch_idxs, acc))
            else:
                print("No checkpoint file found")
    writer.close()
    coord.request_stop()
    coord.join(threads)
