import tensorflow as tf
import numpy as np
reuse = tf.compat.v1.AUTO_REUSE


def dct(x):
    i = 0
    dct_kernel = np.zeros((4, 4, 1, 16), dtype=np.float32)
    for k in range(0, 4):
        for l in range(0, 4):
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
                    dct_kernel[m, n, 0, i] = \
                        (wk*wl/4)*np.math.cos(np.math.pi*k*(2*m+1)/16)*np.math.cos(np.math.pi*l*(2*n+1)/16)
            i = i+1
    dct_kernel = tf.constant_initializer(dct_kernel)
    w_dct = tf.get_variable("weights", [4, 4, 1, 16], initializer=dct_kernel, dtype=tf.float32)
    conv_dct = tf.nn.conv2d(x, w_dct, strides=[1, 1, 1, 1], padding='VALID')
    feature = tf.abs(conv_dct)
    return feature


def hpf(x):
    hpf_kernel = ([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    hpf_kernel = tf.constant_initializer(hpf_kernel)
    w_hpf = tf.get_variable("weights", [3, 3, 1, 16], dtype=tf.float32, 
                            initializer=hpf_kernel)
    feature = tf.nn.conv2d(x, w_hpf, strides=[1, 1, 1, 1], padding='SAME')
    return feature


def srm(x):
    srm_kernel = ([[[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -2, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -2, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 3, -3, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 2, -1, 0], [0, 0, -4, 2, 0], [0, 0, 2, -1, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, 0, 0, 0], [0, -1, 2, 0, 0], [0, 2, -4, 0, 0], [0, -1, 2, 0, 0], [0, 0, 0, 0, 0]]],
               [[[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]],
               [[[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
               [[[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2], [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]]],
               [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]],
               [[[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0], [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]]]])
    w_srm = tf.get_variable("weights", [5, 5, 1, 30], initializer=srm_kernel, dtype=tf.float32)
    feature = tf.nn.conv2d(x, w_srm, strides=[1, 1, 1, 1], padding='SAME')
    return feature






