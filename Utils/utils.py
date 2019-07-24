import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, ThresholdedReLU


def bias_var(out_channels, init_method):
    initial_value = tf.constant(0.0, shape=[out_channels])
    biases = tf.Variable(initial_value)

    return biases


def conv_var(kernel_size, in_channels, out_channels, init_method, name):
    shape = [kernel_size[0], kernel_size[1], kernel_size[2], in_channels, out_channels]
    if init_method == 'msra':
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    elif init_method == 'xavier':
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def first_conv(input_layer, channel):
    filters = conv_var(kernel_size=(1, 1, 7), in_channels=1, out_channels=channel, init_method='msra',
                       name='first_convolution')
    x1_0 = tf.nn.conv3d(input_layer, filters, [1, 1, 1, 2, 1], padding='VALID')
    return x1_0


def loop_block(input_layer, channels_per_layer, kernel_size, layer_num, is_train, block_name, loop_num=1):
    channels = channels_per_layer
    node_0_channels = input_layer.get_shape().as_list()[-1]
    # init param
    param_dict = {}
    # kernel_size = (1, 1, 1) if if_b == True else (1, 1, 7)
    for layer_id in range(1, layer_num):
        add_id = 1
        while layer_id + add_id <= layer_num:
            ## ->
            filters = conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra',
                               name=block_name + '-' + str(layer_id) + '_' + str(layer_id + add_id))
            param_dict[str(layer_id) + '_' + str(layer_id + add_id)] = filters
            ## <-
            filters_inv = conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels,
                                   init_method='msra',
                                   name=block_name + '-' + str(layer_id + add_id) + '_' + str(layer_id))
            param_dict[str(layer_id + add_id) + '_' + str(layer_id)] = filters_inv
            add_id += 1

    for layer_id in range(layer_num):
        filters = conv_var(kernel_size=kernel_size, in_channels=node_0_channels, out_channels=channels,
                           init_method='msra', name=block_name + '-' + str(0) + '_' + str(layer_id + 1))
        param_dict[str(0) + '_' + str(layer_id + 1)] = filters

    assert len(param_dict) == layer_num * (layer_num - 1) + layer_num

    # init blob
    blob_dict = {}

    for layer_id in range(1, layer_num + 1):
        bottom_blob = input_layer
        bottom_param = param_dict['0_' + str(layer_id)]
        for layer_id_id in range(1, layer_id):
            bottom_blob = tf.concat((bottom_blob, blob_dict[str(layer_id_id)]), axis=4)
            bottom_param = tf.concat((bottom_param, param_dict[str(layer_id_id) + '_' + str(layer_id)]), axis=3)

        mid_layer = tf.contrib.layers.batch_norm(bottom_blob, scale=True, is_training=is_train,
                                                 updates_collections=None)
        mid_layer = tf.nn.relu(mid_layer)
        # mid_layer = bn_prelu(bottom_blob, axis=4)
        mid_layer = tf.nn.conv3d(mid_layer, bottom_param, [1, 1, 1, 1, 1], padding='SAME')
        # mid_layer = tf.nn.dropout(mid_layer, 0.5)
        blob_dict[str(layer_id)] = mid_layer

    # begin loop
    for loop_id in range(loop_num):
        for layer_id in range(1, layer_num + 1):  ##   [1,2,3,4,5]

            layer_list = [str(l_id) for l_id in range(1, layer_num + 1)]
            layer_list.remove(str(layer_id))

            bottom_blobs = blob_dict[layer_list[0]]
            bottom_param = param_dict[layer_list[0] + '_' + str(layer_id)]
            for bottom_id in range(len(layer_list) - 1):
                bottom_blobs = tf.concat((bottom_blobs, blob_dict[layer_list[bottom_id + 1]]),
                                         axis=4)  ###  concatenate the data blobs
                bottom_param = tf.concat((bottom_param, param_dict[layer_list[bottom_id + 1] + '_' + str(layer_id)]),
                                         axis=3)  ###  concatenate the parameters

            mid_layer = tf.contrib.layers.batch_norm(bottom_blobs, scale=True, is_training=is_train,
                                                     updates_collections=None)
            mid_layer = tf.nn.relu(mid_layer)
            # mid_layer = bn_prelu(bottom_blobs, axis=4)
            mid_layer = tf.nn.conv3d(mid_layer, bottom_param, [1, 1, 1, 1, 1], padding='SAME')  ###  update the data blob
            # mid_layer = tf.nn.dropout(mid_layer, 0.5)
            blob_dict[str(layer_id)] = mid_layer

    transit_feature = blob_dict['1']
    for layer_id in range(2, layer_num + 1):
        transit_feature = tf.concat((transit_feature, blob_dict[str(layer_id)]), axis=4)

    block_feature = tf.concat((input_layer, transit_feature), axis=4)

    return block_feature, transit_feature
