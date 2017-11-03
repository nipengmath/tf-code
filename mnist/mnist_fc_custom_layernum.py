# -*- coding: UTF-8 -*-
import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def inference(input_tensor, avg_class, w1, b1, w2, b2, w3, b3, wo, bo):
    # 使用滑动平均类
    if avg_class is None:
        if FLAGS.hidden1:
            layers1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
            if FLAGS.hidden2 == 0:
                logits = tf.matmul(layers1, wo) + bo
        if FLAGS.hidden2:
            layers2 = tf.nn.relu(tf.matmul(layers1, w2) + b2)
            if FLAGS.hidden3 == 0:
                logits = tf.matmul(layers2, wo) + bo
        if FLAGS.hidden3:
            layers3 = tf.nn.relu(tf.matmul(layers2, w3) + b3)
            logits = tf.matmul(layers3, wo) + bo
        return logits
    else:
        if FLAGS.hidden1:
            layers1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(w1)) + avg_class.average(b1))
            if FLAGS.hidden2 == 0:
                logits = tf.matmul(layers1, avg_class.average(
                    wo)) + avg_class.average(bo)
        if FLAGS.hidden2:
            layers2 = tf.nn.relu(
                tf.matmul(layers1, avg_class.average(w2)) + avg_class.average(b2))
            if FLAGS.hidden3 == 0:
                logits = tf.matmul(layers2, avg_class.average(
                    wo)) + avg_class.average(bo)
        if FLAGS.hidden3:
            layers3 = tf.nn.relu(
                tf.matmul(layers2, avg_class.average(w3)) + avg_class.average(b3))
            logits = tf.matmul(layers3, avg_class.average(wo)
                               ) + avg_class.average(bo)
        return logits


def run_training():
    # 读取数据
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, one_hot=True)

    # 定义输入和标签占位符
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

    # 定义变量
    last_hidden_unit = 784
    w1 = b1 = w2 = b2 = w3 = b3 = None
    if FLAGS.hidden1:
        last_hidden_unit = FLAGS.hidden1
        with tf.name_scope('hidden1'):
            w1 = tf.Variable(tf.truncated_normal(
                [784, FLAGS.hidden1], stddev=0.1), name='W')
            b1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.hidden1]), name='b')
    if FLAGS.hidden2:
        last_hidden_unit = FLAGS.hidden2
        with tf.name_scope('hidden2'):
            w2 = tf.Variable(tf.truncated_normal(
                [FLAGS.hidden1, FLAGS.hidden2], stddev=0.1), name='W')
            b2 = tf.Variable(tf.constant(0.1, shape=[FLAGS.hidden2]), name='b')
    if FLAGS.hidden3:
        last_hidden_unit = FLAGS.hidden3
        with tf.name_scope('hidden3'):
            w3 = tf.Variable(tf.truncated_normal(
                [FLAGS.hidden2, FLAGS.hidden3], stddev=0.1), name='W')
            b3 = tf.Variable(tf.constant(0.1, shape=[FLAGS.hidden3]), name='b')
    with tf.name_scope('output'):
        wo = tf.Variable(tf.truncated_normal(
            [last_hidden_unit, 10], stddev=0.1), name='W')
        bo = tf.Variable(tf.constant(0.1, shape=[10]), name='b')

    # 定义logit
    with tf.name_scope('logit'):
        y = tf.nn.softmax(inference(x, None, w1, b1, w2, b2, w3, b3, wo, bo))

    # 定义训练轮数及相关的滑动平均类
    if FLAGS.average:
        with tf.name_scope('logit_avg'):
            global_step = tf.Variable(0, trainable=False)
            variable_averages = tf.train.ExponentialMovingAverage(
                0.99, global_step)
            variables_averages_op = variable_averages.apply(
                tf.trainable_variables())
            average_y = inference(x, variable_averages,
                                  w1, b1, w2, b2, w3, b3, wo, bo)

    # 定义损失函数
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1), name='xentropy')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='xentropy_mean')
        regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        regularaztion = regularizer(wo)
        if FLAGS.hidden1:
            regularaztion += regularizer(w1)
        if FLAGS.hidden2:
            regularaztion += regularizer(w2)
        if FLAGS.hidden3:
            regularaztion += regularizer(w3)
        loss = tf.add(cross_entropy_mean, regularaztion, name='loss')

    # 定义训练方法
    if FLAGS.average:
        with tf.name_scope('train'):
            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate, global_step,
                data_sets.train.num_examples / FLAGS.batch_size,
                0.99, staircase=True)
            train_step = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(loss, global_step=global_step)
            with tf.control_dependencies([train_step, variables_averages_op]):
                train_op = tf.no_op(name='train')
    else:
        with tf.name_scope('train'):
            train_op = tf.train.GradientDescentOptimizer(
                FLAGS.learning_rate).minimize(loss)

    # 定义精度
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if FLAGS.hidden1:
        tf.summary.histogram('w1', w1)
        tf.summary.histogram('b1', b1)
    if FLAGS.hidden2:
        tf.summary.histogram('w2', w2)
        tf.summary.histogram('b2', b2)
    if FLAGS.hidden3:
        tf.summary.histogram('w3', w3)
        tf.summary.histogram('b3', b3)
    tf.summary.histogram('wo', wo)
    tf.summary.histogram('bo', bo)
    tf.summary.histogram('y', y)
    if FLAGS.average:
        tf.summary.histogram('average_y', average_y)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(
            FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        sess.run(tf.global_variables_initializer())

        for step in range(FLAGS.max_steps + 1):
            # train
            batch_xs, batch_ys = data_sets.train.next_batch(FLAGS.batch_size)

            _, loss_value = sess.run([train_op, loss], feed_dict={
                                     x: batch_xs, y_: batch_ys})

            if step % 100 == 0:
                # print('Step %d: loss = %.2f' % (step, loss_value))
                summary = sess.run(merged, feed_dict={
                                   x: batch_xs, y_: batch_ys})
                train_writer.add_summary(summary, step)

            # test
            if step % 1000 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={
                                        x: data_sets.test.images, y_: data_sets.test.labels})
                test_writer.add_summary(summary, step)
                print('After %d training step, test accuracy is %g' %
                      (step, acc))


def main(_):
    if FLAGS.hidden1:
        FLAGS.log_dir += '_h1_' + str(FLAGS.hidden1)
    if FLAGS.hidden2:
        FLAGS.log_dir += '_h2_' + str(FLAGS.hidden2)
    if FLAGS.hidden3:
        FLAGS.log_dir += '_h3_' + str(FLAGS.hidden3)
    if FLAGS.average:
        FLAGS.log_dir += '_avg'
    FLAGS.log_dir = FLAGS.log_dir + '_lr_' + str(FLAGS.learning_rate)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    print(FLAGS)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden1', type=int, default=100)
    parser.add_argument('--hidden2', type=int, default=0)
    parser.add_argument('--hidden3', type=int, default=0)
    parser.add_argument('--input_data_dir', type=str, default='./MNIST_data')
    parser.add_argument('--log_dir', type=str, default='./tmp/fc')
    parser.add_argument('--average', default=False, action='store_true')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
