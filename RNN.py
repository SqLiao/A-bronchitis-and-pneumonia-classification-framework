from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from random import shuffle
import librosa
import time
from tools import labelExchangeLoad, labelDicLoad

# training parameters
learning_rate = 0.001
batch_size = 32
display_step = 2
test_size = 96

# model parameters
num_input = 20
timesteps = 50
num_hidden = 5
num_classes = 2

# data parameters
height = 20  # mfcc features
width = 50  # (max) length of utterance
classes = 2

# Initialization
with tf.variable_scope('initAll'):
    train_path = ''
    test_path = ''

    logs_path = ''
    model_path = ''

    labelDic = {'': -1}  # using Label mapping
    labelExDic = {'': -1}  # using Label convertion

    # GPU setting
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options)
    gpu_config.gpu_options.allow_growth = True

    labelExchangeLoad(labelExDic)

with tf.variable_scope('Placeholder'):
    x1 = tf.placeholder(tf.float32, [None, timesteps, num_input], name='x1')
    y1 = tf.placeholder(tf.float32, [None, num_classes], name='y1')

with tf.variable_scope('NN'):
    w1 = tf.Variable(tf.random_normal([num_hidden, num_classes]), name='w1')
    b1 = tf.Variable(tf.random_normal([num_classes]), name='b1')


def RNN(x):
    x = tf.reshape(x, [-1, timesteps, num_input])
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[:, -1, :], w1) + b1


# model definition
logits = RNN(x1)
prediction = tf.nn.softmax(logits)

tf.add_to_collection('logits', logits)
tf.add_to_collection('prediction', prediction)

with tf.variable_scope('Loss'):
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y1), name='loss_op')

with tf.variable_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    train_op = optimizer.minimize(loss_op, name='train_op')

with tf.variable_scope('Accuracy'):
    pre = tf.argmax(prediction, 1)  # predictive labels
    res = tf.argmax(y1, 1)  # Actual labels
    correct_pred = tf.equal(pre, res)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

with tf.variable_scope('summary'):
    tf.summary.scalar("loss", loss_op)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()


def mfcc_batch_generator(p, isTest, batch_size=10):
    batch_features = []
    labels = []
    label_ids = []
    files = os.listdir(p)
    while True:
        shuffle(files)
        for file in files:
            if not file.endswith(".npy"): continue
            if isTest: print('testing : ' + file)
            label_id = int(1) if 'pneumonia' in file else int(0)
            label_ids.append(label_id)
            label = dense_to_one_hot(label_id, num_classes)
            labels.append(label)
            mfcc = np.load(p + file)
            batch_features.append(mfcc)
            if len(batch_features) >= batch_size:
                yield np.array(batch_features), np.array(labels), np.array(label_ids)
                batch_features = []
                labels = []
                label_ids = []


def dense_to_one_hot(labels_dense, num_classes=10):
    '''
    :return: A one-dimensional array used to describe the predictions
    '''
    return np.eye(num_classes)[labels_dense]


def test(sess):
    print("Staring Testing...")
    step = 1
    count_acc = 0.0
    test_batch_size = test_size
    while True:
        batch = mfcc_batch_generator(test_path, True, test_batch_size)
        batch_x, batch_y, label = next(batch)
        batch_x = batch_x.reshape((test_batch_size, timesteps, num_input))
        print('prediction : ', end='')
        print(sess.run(pre, feed_dict={x1: batch_x, y1: batch_y}))
        print('result     : ', end='')
        print(sess.run(res, feed_dict={x1: batch_x, y1: batch_y}))
        globe_acc = sess.run(accuracy, feed_dict={x1: batch_x, y1: batch_y})
        count_acc = globe_acc + count_acc
        print("Iter " + str(step * test_batch_size) + \
              ", Testing Accuracy = " + "{:.5f}".format(globe_acc) + \
              ", AVG Accuracy = " + "{:.5f}".format(count_acc / step))
        if (count_acc / step) > 0.83:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if step == 1: break
        step += 1
    print("Testing Finished")


def train():
    print("Staring Training...")
    count_loss = 0.0  #
    count_acc = 0.0
    saver = tf.train.Saver(max_to_keep=200)
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        save_path = os.path.join(logs_path, STARTED_DATESTRING)
        summary_writer = tf.summary.FileWriter(save_path, graph=tf.get_default_graph())
        model_save_temp_path = os.path.join(model_path, STARTED_DATESTRING)
        os.mkdir(model_save_temp_path)
        step = 1
        while True:
            batch = mfcc_batch_generator(train_path, False, batch_size)
            batch_x, batch_y, label = next(batch)
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            sess.run(train_op, feed_dict={x1: batch_x, y1: batch_y})  # model optimization
            if step % display_step == 0:
                start_time = time.time()
                print('prediction : ', end='')
                print(sess.run(pre, feed_dict={x1: batch_x, y1: batch_y}))
                print('result     : ', end='')
                print(sess.run(res, feed_dict={x1: batch_x, y1: batch_y}))
                globe_acc, globe_loss, summary = sess.run([accuracy, loss_op, summary_op],
                                                          feed_dict={x1: batch_x, y1: batch_y})
                cou = step / display_step
                count_loss = globe_loss + count_loss
                count_acc = globe_acc + count_acc
                summary_writer.add_summary(summary, step * batch_size)
                duration = time.time() - start_time
                print("Iter " + str(step * batch_size) + \
                      ", Minibatch Loss = " + "{:.6f}".format(globe_loss) + \
                      ", AVG Loss = " + "{:.6f}".format(count_loss / cou) + \
                      ", Training Accuracy = " + "{:.5f}".format(globe_acc) + \
                      ",({:.3f} sec/step)".format(duration))

            saver.save(sess, model_save_temp_path + '/RNN.ckpt', global_step=step * batch_size)
            print('Save the model and test it.')
            test(sess)
            step += 1

            print('Training and testing is over!')


if __name__ == '__main__':
    train()
