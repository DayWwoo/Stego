import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import xavier_initializer
import distance


def get_weights(shape, regularizer):
    weights = tf.get_variable("weights", shape, dtype=tf.float32,
                              initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):  # 512*512*16?
    with arg_scope([layers.batch_norm], decay=0.99, center=True, scale=True, updates_collections=None,
                   is_training=True, fused=True, data_format='NHWC'):
        # , arg_scope([tf.nn.conv2d], strides=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope("layer1"):  # 512*512*12
            w1 = get_weights([3, 3, 16, 16], regularizer)
            b1 = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.2))
            conv1 = tf.nn.conv2d(input_tensor, w1, strides=[1, 1, 1, 1], padding='SAME')
            bn1 = layers.batch_norm(tf.nn.bias_add(conv1, b1))
            actv1 = tf.nn.relu(bn1)

        with tf.variable_scope("layer2"):  # 256*256*24
            w2 = get_weights([3, 3, 16, 24], regularizer)
            b2 = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.2))
            conv2 = tf.nn.conv2d(actv1, w2, strides=[1, 2, 2, 1], padding='SAME')
            bn2 = layers.batch_norm(tf.nn.bias_add(conv2, b2))
            res = layers.batch_norm(tf.nn.bias_add(
                tf.nn.conv2d(input_tensor, w2, strides=[1, 2, 2, 1], padding='SAME'), b2))
            actv2 = tf.nn.relu(tf.add(res, bn2))

        with tf.variable_scope("layer3-4"):  # 256*256*24
            w3 = get_weights([3, 3, 24, 24], regularizer)
            b3 = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.2))
            conv3 = tf.nn.conv2d(actv2, w3, padding='SAME')
            actv3 = tf.nn.relu(layers.batch_norm(tf.nn.bias_add(conv3, b3)))

            conv4 = tf.nn.conv2d(actv3, w3, padding='SAME')
            actv4 = tf.nn.relu(tf.add(actv2, layers.batch_norm(tf.nn.bias_add(conv4, b3))))

        with tf.variable_scope("layer5"):  # 256*256*24
            w5 = get_weights([3, 3, 24, 24], regularizer)
            b5 = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.2))
            conv5 = tf.nn.bias_add(tf.nn.conv2d(actv4, w5, padding='SAME'), b5)
            actv5 = tf.nn.relu(layers.batch_norm(conv5))

        with tf.variable_scope("layer6"):  # 128*128*48
            w6 = get_weights([3, 3, 24, 48], regularizer)
            b6 = tf.get_variable("bias", [48], initializer=tf.constant_initializer(0.2))
            conv6 = layers.batch_norm(tf.nn.bias_add(
                tf.nn.conv2d(actv5, w6, strides=[1, 2, 2, 1], padding='SAME'), b6))
            res2 = layers.batch_norm(tf.nn.bias_add(
                tf.nn.conv2d(actv4, w6, strides=[1, 2, 2, 1], padding='SAME'), b6))
            actv6 = tf.nn.relu(tf.add(res2, conv6))

        with tf.variable_scope("layer7-8"):  # 128*128*48
            w7 = get_weights([3, 3, 48, 48], regularizer)
            b7 = tf.get_variable("bias", [48], initializer=tf.constant_initializer(0.2))
            conv7 = tf.nn.bias_add(tf.nn.conv2d(actv6, w7, padding='SAME'), b7)
            actv7 = tf.nn.relu(layers.batch_norm(conv7))

            conv8 = tf.nn.bias_add(tf.nn.conv2d(actv7, w7, padding='SAME'), b7)
            actv8 = tf.nn.relu(tf.add(actv6, layers.batch_norm(conv8)))

        with tf.variable_scope("layer9"):  # 128*128*48
            w9 = get_weights([3, 3, 48, 48], regularizer)
            b9 = tf.get_variable("bias", [48], initializer=tf.constant_initializer(0.2))
            conv9 = tf.nn.bias_add(tf.nn.conv2d(actv8, w9, padding='SAME'), b9)
            actv9 = tf.nn.relu(layers.batch_norm(conv9))

        with tf.variable_scope("layer10"):  # 64*64*96
            w10 = get_weights([3, 3, 48, 96], regularizer)
            b10 = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.2))
            conv10 = tf.nn.bias_add(tf.nn.conv2d(actv9, w10, strides=[1, 2, 2, 1], padding='SAME'), b10)
            res3 = layers.batch_norm(tf.nn.bias_add(
                tf.nn.conv2d(actv8, w10, strides=[1, 2, 2, 1], padding='SAME'), b10))
            actv10 = tf.nn.relu(tf.add(res3, layers.batch_norm(conv10)))

        with tf.variable_scope("layer11-12"):  # 64*64*96
            w11 = get_weights([3, 3, 96, 96], regularizer)
            b11 = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.2))
            conv11 = tf.nn.bias_add(tf.nn.conv2d(actv10, w11, padding='SAME'), b11)
            actv11 = tf.nn.relu(layers.batch_norm(conv11))

            conv12 = tf.nn.bias_add(tf.nn.conv2d(actv11, w11, padding='SAME'), b11)
            actv12 = tf.nn.relu(tf.add(actv10, layers.batch_norm(conv12)))

        with tf.variable_scope("layer13"):  # 64*64*96
            w13 = get_weights([3, 3, 96, 96], regularizer)
            b13 = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.2))
            conv13 = tf.nn.bias_add(tf.nn.conv2d(actv12, w13, padding='SAME'), b13)
            actv13 = tf.nn.relu(layers.batch_norm(conv13))

        with tf.variable_scope("layer14"):  # 32*32*192
            w14 = get_weights([3, 3, 96, 192], regularizer)
            b14 = tf.get_variable("bias", [192], initializer=tf.constant_initializer(0.2))
            conv14 = tf.nn.bias_add(tf.nn.conv2d(actv13, w14, strides=[1, 2, 2, 1], padding='SAME'), b14)
            res4 = layers.batch_norm(tf.nn.bias_add(
                tf.nn.conv2d(actv12, w14, strides=[1, 2, 2, 1], padding='SAME'), b14))
            actv14 = tf.nn.relu(tf.add(res4, layers.batch_norm(conv14)))

        with tf.variable_scope("layer15-16"):  # 32*32*192
            w15 = get_weights([3, 3, 192, 192], regularizer)
            b15 = tf.get_variable("bias", [192], initializer=tf.constant_initializer(0.2))
            conv15 = tf.nn.bias_add(tf.nn.conv2d(actv14, w15, padding='SAME'), b15)
            actv15 = tf.nn.relu(layers.batch_norm(conv15))

            conv16 = layers.batch_norm(tf.nn.bias_add(tf.nn.conv2d(actv15, w15, padding='SAME'), b15))
            actv16 = tf.nn.relu(tf.add(actv14, conv16))

        with tf.variable_scope("layer17"):  # 32*32*192
            w17 = get_weights([3, 3, 192, 192], regularizer)
            b17 = tf.get_variable("bias", [192], initializer=tf.constant_initializer(0.2))
            conv17 = tf.nn.bias_add(tf.nn.conv2d(actv16, w17, padding='SAME'), b17)
            actv17 = tf.nn.relu(layers.batch_norm(conv17))

        with tf.variable_scope("layer18"):  # 16*16*384
            w18 = get_weights([3, 3, 192, 384], regularizer)
            b18 = tf.get_variable("bias", [384], initializer=tf.constant_initializer(0.2))
            conv18 = layers.batch_norm(tf.nn.bias_add(
                tf.nn.conv2d(actv17, w18, strides=[1, 2, 2, 1], padding='SAME'), b18))
            res5 = layers.batch_norm(tf.nn.bias_add(
                tf.nn.conv2d(actv16, w18, strides=[1, 2, 2, 1], padding='SAME'), b18))
            actv18 = tf.nn.relu(tf.add(res5, conv18))

        with tf.variable_scope("layer19-20"):  # 16*16*384
            w19 = get_weights([3, 3, 384, 384], regularizer)
            b19 = tf.get_variable("bias", [384], initializer=tf.constant_initializer(0.2))
            conv19 = tf.nn.bias_add(tf.nn.conv2d(actv18, w19, padding='SAME'), b19)
            actv19 = tf.nn.relu(layers.batch_norm(conv19))

            conv20 = layers.batch_norm(tf.nn.bias_add(tf.nn.conv2d(actv19, w19, padding='SAME'), b19))
            actv20 = tf.nn.relu(tf.add(actv18, conv20))

        with tf.variable_scope("layer-gap"):  # 1*1*384
            gap = tf.nn.avg_pool2d(actv20, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding="SAME")

        with tf.variable_scope("layer-fc1"):
            weights = tf.get_variable("weights", [384, 192], dtype=tf.float32,
                                      initializer=layers.xavier_initializer())
            flat = layers.flatten(gap)
            convfc = tf.matmul(flat, weights)
            actvfc = tf.nn.relu(convfc)
            if regularizer is not None:
                tf.add_to_collection('losses', regularizer(weights))

        with tf.variable_scope("layer-fc2"):
            fc2 = layers.fully_connected(layers.flatten(actvfc), num_outputs=2,
                                         activation_fn=tf.nn.relu, normalizer_fn=None,
                                         weights_initializer=layers.xavier_initializer())
    return fc2





















