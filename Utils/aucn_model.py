from Utils.utils import *
from keras.layers.core import Reshape
import tensorflow as tf


def bn_relu(input_layer, is_train):
    next_layer = tf.contrib.layers.batch_norm(input_layer, scale=True, is_training=is_train,
                                              updates_collections=None)
    next_layer = tf.nn.relu(next_layer)
    return next_layer


def build_model(input_images, num_output, is_train, keep_prob):
    k = 36
    first_channel = 64
    t_channel = 48

    print(input_images.get_shape())

    x1_0 = first_conv(input_images, channel=first_channel)

    # build spectral blocks
    print('the input shape of spectral blocks is: ', x1_0.get_shape())
    x1, x_1_transit_feature = loop_block(x1_0, channels_per_layer=k, kernel_size=(1, 1, 7), layer_num=2,
                                         is_train=is_train, block_name='spectral_block', loop_num=1)

    # transition layer
    print('the output shape of spectral blocks is: ', x1.get_shape())
    x1 = bn_relu(x1, is_train=is_train)
    tran1_var = conv_var(kernel_size=(1, 1, x1.get_shape()[3]), in_channels=x1.get_shape()[4], out_channels=t_channel,
                         init_method='msra', name='first_transition')
    tran1 = tf.nn.conv3d(x1, tran1_var, [1, 1, 1, 1, 1], padding='VALID')

    tran1 = bn_relu(tran1, is_train=is_train)
    print(tran1.get_shape())
    tran2 = Reshape((tran1.get_shape()[1], tran1.get_shape()[2], tran1.get_shape()[4], 1))(tran1)

    print(tran2.get_shape())
    tran2_var = conv_var(kernel_size=(3, 3, t_channel), in_channels=1, out_channels=first_channel,
                         init_method='msra', name='second_transition')
    x2_0 = tf.nn.conv3d(tran2, tran2_var, [1, 1, 1, 1, 1], padding='VALID')

    print('the input of spatial block:', x2_0.get_shape())
    # build spatial blocks
    x2, x_2_transit_feature = loop_block(x2_0, channels_per_layer=k, kernel_size=(3, 1, 1), layer_num=2,
                                         is_train=is_train, block_name='spatial_block_1', loop_num=1)

    print('the output of spatial block:', x2.get_shape())
    x3, x_3_transit_feature = loop_block(x2_0, channels_per_layer=k, kernel_size=(1, 3, 1), layer_num=2,
                                         is_train=is_train, block_name='spatial_block_2', loop_num=1)
    x4 = tf.concat([x2, x3], axis=4)

    # Classifier block

    pool1 = tf.nn.avg_pool3d(x4, ksize=[1, x4.get_shape()[1], x4.get_shape()[2], 1, 1],
                             strides=[1, 1, 1, 1, 1], padding='VALID')
    print(pool1.get_shape())
    flatten = tf.layers.flatten(pool1)
    print(flatten.get_shape())
    # flatten = tf.nn.dropout(flatten, keep_prob=keep_prob)
    wfc = tf.get_variable(name='FC_W', shape=[flatten.get_shape()[1], num_output],
                          initializer=tf.contrib.layers.xavier_initializer())
    bfc = tf.get_variable(name='FC_b', initializer=tf.constant(0.0, shape=[num_output]))

    logits = tf.matmul(flatten, wfc) + bfc
    print(logits.get_shape())
    prob = tf.nn.softmax(logits)

    return logits, prob
