import tensorflow as tf
import numpy as np
import os
import conv
import dataload
import matplotlib.pyplot as plt
import dctr
from skimage import io

'''img_train, label_train = dataload.read_and_decode("J_0.4.tfrecords")
with tf.Session() as sess:
    img, label = sess.run(img_train)
    deimg = sess.run(tf.image.decode_jpeg(img))
    for i in 1:
        print('source pic', img)
        plt.imshow(img)
        plt.show()
        print('decode', deimg)
        plt.imshow(deimg)
        plt.show()
        break'''

'''x = [[1, 2], [3, 4], [5, 6]]
plt.imshow(x)
plt.show()'''

img_path = tf.gfile.GFile("D:/Dataload/J_0.1/cover/1.JPEG", 'rb').read()
'''img2 = io.imread("D:/Dataload/J_0.1/cover/1.JPEG")
io.imshow(img2)
io.show()
print(img2.shape)'''
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(img_path)
    # print(img_data.eval())
    img_data = tf.squeeze(img_data)
    img_data = tf.cast(img_data, tf.float32)

    '''hpf_kernel = ([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    hpf_kernel = tf.constant_initializer(hpf_kernel)
    w_hpf = tf.get_variable("weights", [3, 3, 1, 1], dtype=tf.float32,
                            initializer=hpf_kernel)
    img_data = tf.reshape(img_data, shape=[1, 512, 512, 1])
    sess.run(tf.global_variables_initializer())
    img_data = tf.nn.conv2d(img_data, w_hpf, strides=[1, 1, 1, 1], padding='SAME')
    print(img_data.eval())
    img_data = tf.squeeze(img_data)'''

    dct_kernel = np.zeros((4, 4), dtype=np.float32)
    k = 2
    l = 2
    for m in range(0, 4):
        for n in range(0, 4):
            if k == 0:
                wk = 1.0 / np.math.sqrt(2)
            else:
                wk = 1.0
            if l == 0:
                wl = 1.0 / np.math.sqrt(2)
            else:
                wl = 1.0
            dct_kernel[m, n] = \
                (wk * wl / 4) * np.math.cos(np.math.pi * k * (2 * m + 1) / 16) * np.math.cos(
                    np.math.pi * l * (2 * n + 1) / 16)

    dct_kernel = tf.constant_initializer(dct_kernel)
    w_dct = tf.get_variable("weights", [4, 4, 1, 1], initializer=dct_kernel, dtype=tf.float32)

    img_data = tf.reshape(img_data, shape=[1, 512, 512, 1])
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    img_data = tf.nn.conv2d(img_data, w_dct, strides=[1, 1, 1, 1], padding='SAME')
    img_data = tf.abs(img_data)
    # print(img_data.eval())
    img_data = tf.squeeze(img_data)

    plt.imshow(img_data.eval(), cmap=plt.cm.gray)
    plt.show()
