# -*- coding: UTF-8 -*-
import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def inference(input_tensor, avg_class, W, b):
    # 使用滑动平均类
    if avg_class is None:
        return tf.matmul(input_tensor, W) + b
    else:
        return tf.matmul(input_tensor, avg_class.average(W))
        + avg_class.average(b)


def run_training():
    # 读取数据
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, one_hot=True)

    # 定义输入和标签占位符
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

    # 定义变量
    with tf.name_scope('layer'):
        # W = tf.Variable(tf.truncated_normal([784, 10]), name='W')
        # b = tf.Variable(tf.constant(0.1, shape=[10]), name='b')
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        b = tf.Variable(tf.zeros([10]), name='b')

    # 定义logit
    with tf.name_scope('Wx_plus_b'):
        y = tf.nn.softmax(inference(x, None, W, b))

    # 定义训练轮数及相关的滑动平均类
    if FLAGS.average:
        with tf.name_scope('Wx_plus_b_Average'):
            global_step = tf.Variable(0, trainable=False)
            variable_averages = tf.train.ExponentialMovingAverage(
                0.99, global_step)
            variables_averages_op = variable_averages.apply(
                tf.trainable_variables())
            average_y = inference(x, variable_averages, W, b)

    # 定义损失函数
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1), name='xentropy')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='xentropy_mean')
        regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        regularaztion = regularizer(W)
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

    tf.summary.histogram('W', W)
    tf.summary.histogram('b', b)
    tf.summary.histogram('y', y)
    if FLAGS.average:
        tf.summary.histogram('average_y', average_y)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    summary = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        for step in range(FLAGS.max_steps):
            batch_xs, batch_ys = data_sets.train.next_batch(FLAGS.batch_size)
            feed_dict = {x: batch_xs, y_: batch_ys}

            _, loss_value = sess.run([train_op, loss], feed_dict)

            if step % 100 == 0:
                # print('Step %d: loss = %.2f' % (step, loss_value))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
            if step % 1000 == 0:
                acc = sess.run(accuracy, feed_dict={
                               x: data_sets.test.images,
                               y_: data_sets.test.labels})
                print('After %d training step, test accuracy is %g' %
                      (step, acc))


def main(_):
    if FLAGS.average:
        FLAGS.log_dir += '_avg'
    FLAGS.log_dir = FLAGS.log_dir + '_lr_' + str(FLAGS.learning_rate)
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--input_data_dir', type=str, default='./MNIST_data')
    parser.add_argument('--log_dir', type=str, default='./tmp/linear')
    parser.add_argument('--average', default=False, action='store_true')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
