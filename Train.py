# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import time
import datetime
import collections
from sklearn import metrics, preprocessing
from operator import truediv
from Utils import aucn_model, record, extract_samll_cubic
import tensorflow as tf
from keras.utils.np_utils import to_categorical


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        nb_val = int(proportion * len(indexes))
        train[i] = indexes[:-nb_val]
        test[i] = indexes[-nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def count_params():
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        params = 1
        for dim in shape:
            params = params * dim.value
        total_params += params
    print("Total training params: %.1f" % total_params)


def into_batch(data, label, batch_size, shuffle):
    if shuffle:
        rand_indexes = np.random.permutation(data.shape[0])
        data = data[rand_indexes]
        label = label[rand_indexes]

    batch_count = len(data) // batch_size
    batches_data = np.split(data[:batch_count * batch_size], batch_count)
    batches_data.append(data[batch_count * batch_size:])
    batches_labels = np.split(label[:batch_count * batch_size], batch_count)
    batches_labels.append(label[batch_count * batch_size:])
    if len(data) % batch_size == 0:
        batch_count = batch_count
    else:
        batch_count += 1

    return batches_data, batches_labels, batch_count


def get_center_loss(features, labels, alpha, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)
    loss = tf.nn.l2_loss(features - centers_batch)

    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op


print('-----Importing Dataset-----')

global Dataset
dataset = input('Please input the name of Dataset(IN, SS or KSC):')
Dataset = dataset.upper()
if Dataset == 'KSC':
    KSC = sio.loadmat('datasets/KSC.mat')
    gt_KSC = sio.loadmat('datasets/KSC_gt.mat')
    data_hsi = KSC['KSC']
    gt_hsi = gt_KSC['KSC_gt']
    TOTAL_SIZE = 5211
    VALIDATION_SPLIT = 0.962875  # 200: 0.962875 400: 0.9245 600: 0.8862 800:84765

if Dataset == 'IN':
    mat_data = sio.loadmat('datasets/Indian_pines_corrected.mat')
    data_hsi = mat_data['indian_pines_corrected']
    mat_gt = sio.loadmat('datasets/Indian_pines_gt.mat')
    gt_hsi = mat_gt['indian_pines_gt']
    TOTAL_SIZE = 10249
    VALIDATION_SPLIT = 0.9812  # 200:0.9812 400:0.9617 600: 0.9422 800:

if Dataset == 'SS':
    Salinas = sio.loadmat('datasets/Salinas_corrected.mat')
    gt_Salinas = sio.loadmat('datasets/Salinas_gt.mat')
    data_hsi = Salinas['salinas_corrected']
    gt_hsi = gt_Salinas['salinas_gt']
    TOTAL_SIZE = 54129
    VALIDATION_SPLIT = 0.996453  # 200：0.996453 400：  3200:


print(data_hsi.shape)
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
nb_classes = max(gt)
print('The class numbers of the HSI data is:', nb_classes)

print('-----Importing Setting Parameters-----')
ITER = 10
PATCH_LENGTH = 4

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels = data_hsi.shape[2]

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
padded_data = np.lib.pad(data_, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, nb_classes))

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for index_iter in range(ITER):
    print("-----Starting the  %d Iteration-----" % (index_iter + 1))
    best_weights_path = 'models/'+Dataset+'_aucn_'+day_str+'@'+str(index_iter+1)

    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    ALL_TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    VAL_SIZE = int(0.5*TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE
    print('Test size: ', TEST_SIZE)

    y_train = gt[train_indices]-1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices]-1
    y_test = to_categorical(np.asarray(y_test))

    print('-----Selecting Small Pieces from the Original Cube Data-----')
    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, data_,
                                                        PATCH_LENGTH, padded_data, img_channels)
    test_data = extract_samll_cubic.select_small_cubic(ALL_TEST_SIZE, test_indices, data_,
                                                       PATCH_LENGTH, padded_data, img_channels)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], img_channels)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], img_channels)
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]
    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)

    print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)

    print("----- Training the AUCN　Model-------")
    graph = tf.Graph()
    with graph.as_default():
        input_hsi = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2], x_train.shape[3], 1],
                                   name='input_hsi')
        labels_hsi = tf.placeholder(tf.float32, [None, nb_classes], name='labels_hsi')
        is_train = tf.placeholder(tf.bool, shape=[], name='is_train')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

        logits, prob = aucn_model.build_model(input_hsi, nb_classes, is_train=is_train, keep_prob=keep_prob)
        pred = tf.argmax(prob, 1)
        tf.add_to_collection('pred', pred)

        # Loss and accuracy
        loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                                       labels=labels_hsi))
        if_correct = tf.equal(pred, tf.argmax(labels_hsi, 1))
        accuracy = tf.reduce_mean(tf.cast(if_correct, tf.float32))

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        center_loss, center, center_updata_op = get_center_loss(logits, tf.argmax(labels_hsi, 1), 0.5, nb_classes)
        loss_total = loss_cross_entropy + center_loss * 0.001

        # Optimizer
        # rms = tf.train.RMSPropOptimizer(learning_rate)
        adam = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)

        train_op = adam.minimize(loss_cross_entropy + l2_loss * 1e-4 + center_loss * 1e-2)
        saver = tf.train.Saver()

    # begin training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    lr = 0.0001
    batch_size = 16
    nb_epoch = 400

    batches_data_val, batches_labels_val, batch_count_val = into_batch(x_val, y_val, batch_size, shuffle=False)

    with tf.Session(config=config, graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        count_params()

        loss_train = []
        acc_train = []
        loss_val = []
        acc_val = []
        loss_test = 0
        acc_test = 0
        best_acc = 0

        min_loss = float('inf')

        val_acc_patience = 0
        val_loss_patience = 0
        patience = 10

        tic1 = time.clock()
        for epoch in range(1, nb_epoch + 1):
            if val_loss_patience == patience:
                val_loss_patience = 0
                lr = lr * 0.5
                print('Epoch %3d: reducing learning rate to %.9f * 10-5' % (epoch, lr*100000))
            else:
                print('Epoch %3d: the learning rate is still %.9f * 10-5' % (epoch, lr*100000))

            batches_data, batches_labels, batch_count = into_batch(x_train, y_train, batch_size, shuffle=True)

            #  train
            loss_per_batch = []
            acc_per_batch = []
            loss_total_per_batch = []

            for batch_id in range(batch_count):
                data_per_batch = batches_data[batch_id]
                label_per_batch = batches_labels[batch_id]
                result_per_batch = sess.run([train_op, loss_cross_entropy, accuracy, loss_total, center_updata_op],
                                            feed_dict={input_hsi: data_per_batch, labels_hsi: label_per_batch,
                                                       learning_rate: lr, is_train: True, keep_prob: 0.5})

                loss_per_batch.append(result_per_batch[1])
                acc_per_batch.append(result_per_batch[2])
                loss_total_per_batch.append(result_per_batch[3])
                print('%d/%d: - loss: %.5f - acc :%.5f -total loss:%.5f'
                      % (batch_id + 1, batch_count, result_per_batch[1], result_per_batch[2], result_per_batch[3]))

            loss_train.append(np.mean(loss_per_batch))
            acc_train.append(np.mean(acc_per_batch))
            saver.save(sess, best_weights_path + '.ckpt')

            # Validation
            val_loss_per_batch = []
            val_acc_per_batch = []

            for val_batch_id in range(batch_count_val):
                val_data_per_batch = batches_data_val[val_batch_id]
                val_label_per_batch = batches_labels_val[val_batch_id]
                val_result_per_batch = sess.run([loss_cross_entropy, accuracy],
                                                feed_dict={input_hsi: val_data_per_batch,
                                                           labels_hsi: val_label_per_batch,
                                                           is_train: False,
                                                           keep_prob: 1})
                val_loss_per_batch.append(val_result_per_batch[0])
                val_acc_per_batch.append(val_result_per_batch[1])
            loss_val.append(np.mean(val_loss_per_batch))
            acc_val.append(np.mean(val_acc_per_batch))

            print('Epoch %3d/%3d: - loss: %.5f - acc: %.5f - val_loss: %.5f - val_acc: %.5f'
                  % (epoch, nb_epoch, loss_train[-1], acc_train[-1], loss_val[-1], acc_val[-1]))

            if acc_val[-1] > best_acc:
                val_acc_patience = 0
                best_acc = acc_val[-1]
                print('val_acc improve to %.4f' % (best_acc * 100))
            else:
                val_acc_patience += 1
                print('val_acc did not improve from %.4f' % (best_acc * 100))

            if loss_val[-1] < min_loss:
                val_loss_patience = 0
                min_loss = loss_val[-1]
                print('val_loss reduce to %.5f,' % min_loss
                      + 'saving the best model to ' + str(best_weights_path) + '\n')
            else:
                val_loss_patience += 1
                print('val_loss did not improve from %.5f' % min_loss + '\n')

            # rs = sess.run(merged)
            # writer.add_summary(rs, epoch)

        toc1 = time.clock()

        print('------Test the best AUCN model------')
        batches_data_test, batches_labels_test, batch_count_test = into_batch(x_test, y_test, batch_size, shuffle=False)

        pred_aucn = []
        pred_per_batch = []
        test_loss_per_batch = []
        test_acc_per_batch = []

        tic2 = time.clock()
        for test_batch_id in range(batch_count_test):
            test_data_per_batch = batches_data_test[test_batch_id]
            test_label_per_batch = batches_labels_test[test_batch_id]
            pred_per_batch = sess.run([loss_cross_entropy, accuracy, pred], feed_dict={
                input_hsi: test_data_per_batch,
                labels_hsi: test_label_per_batch,
                is_train: False,
                keep_prob: 1})

            if test_batch_id % 50 == 0:
                print('%3d/%3d:Testing the ACUN Model' % (test_batch_id + 1, batch_count_test))
            test_loss_per_batch.append(pred_per_batch[0])
            test_acc_per_batch.append(pred_per_batch[1])

            for i in range(len(pred_per_batch[2])):
                pred_aucn.append(pred_per_batch[2][i])

        toc2 = time.clock()
        loss_test = np.mean(test_loss_per_batch)
        acc_test = np.mean(test_acc_per_batch)
        print('Training Time: ', toc1 - tic1)
        print('Best verified accuracy:', best_acc)
        print('Min verified loss:', min_loss)
        print('Test time:', toc2 - tic2)
        print('Test loss:', loss_test)
        print('Test accuracy', acc_test)

    collections.Counter(pred_aucn)
    gt_test = gt[test_indices]-1
    overall_acc_aucn = metrics.accuracy_score(pred_aucn, gt_test[:-VAL_SIZE])
    confusion_matrix_aucn = metrics.confusallion_matrix(pred_aucn, gt_test[:-VAL_SIZE])
    each_acc_aucn, average_acc_aucn = aa_and_each_accuracy(confusion_matrix_aucn)
    kappa = metrics.cohen_kappa_score(pred_aucn, gt_test[:-VAL_SIZE])

    KAPPA.append(kappa)
    OA.append(overall_acc_aucn)
    AA.append(average_acc_aucn)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_aucn

print("--------AUCN Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'records/'+Dataset+'_aucn_'+day_str+'.txt')
print('The save mode is:'+Dataset+'_aucn_'+day_str)
