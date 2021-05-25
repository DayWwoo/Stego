import tensorflow as tf


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_sample = int(source.size()[0]) + int(target.size()[0])
    total = tf.concat([source, target], concat_dim=0)
    total0 = tf.tile(tf.reshape(total, (1, int(total.size(0)), int(total.size(1)))),
                     multiples=(int(total.size(0)), int(total.size(0)), int(total.size(1))))
    total1 = tf.tile(tf.reshape(total, (int(total.size(0)), 1, int(total.size(1)))),
                     multiples=(int(total.size(0)), int(total.size(0)), int(total.size(1))))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance.data) / (n_sample**2-n_sample)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def linear_kernel(source, target):
    delta = source - target
    loss = tf.reduce_mean((delta[:-1] * delta[1:]).sum(1))
    return loss


def forward(source, target, kernel_type):
    if not kernel_type:
        loss = tf.sqrt(forward(source, target, 'linear')**2 + forward(source, target, 'rbf')**2)
        return loss
    elif kernel_type == 'linear':
        return linear_kernel(source, target)
    elif kernel_type == 'rbf':
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target)
        with tf.no_gradient():
            XX = tf.reduce_mean(kernels[:batch_size, :batch_size])
            YY = tf.reduce_mean(kernels[batch_size:, batch_size:])
            XY = tf.reduce_mean(kernels[:batch_size, batch_size:])
            YX = tf.reduce_mean(kernels[batch_size:, :batch_size])
            loss = tf.reduce_mean(XX+YY-XY-YX)
        return loss
    elif kernel_type == 'mix':
        loss = tf.sqrt(forward(source, target, 'linear')**2 + forward(source, target, 'rbf')**2)
        return loss

