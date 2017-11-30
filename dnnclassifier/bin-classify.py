import itertools
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_DATA = "e:/python/室内室外预测数据集.xls"

data = pd.read_excel(INPUT_DATA)
# print(data.head())
# print(data.columns)

# 分离x、y，且去掉最后一行
df = data[["reportcellkey", 'strongestnbpci', 'aoa', 'ta_calc', 'rsrp', 'rsrq', 'ta',
           'tadltvalue', 'mrtime', 'imsi', 'ndskey', 'uerecordid', 'starttime', 'endtime',
           'positionmark_real']]
df = df.drop([df.shape[0] - 1])

# 列处理
# 1.去除第一列report id和第二列strongestnbpci（PCI为物理层小区编号，与预测室内室外无关）
df = df.drop("reportcellkey", axis=1)
df = df.drop("strongestnbpci", axis=1)
# 2.去除路测时间mrtime
df = df.drop("mrtime", axis=1)
# 3.去除所有的同样值imsi和ndskey
df = df.drop("imsi", axis=1)
df = df.drop("ndskey", axis=1)
# 4.去除userid
df = df.drop("uerecordid", axis=1)
# 5.归一成何时通话和童话时长，由于数据全部为2月份，所有只保留几点通话。
df['duration'] = (df['endtime'] - df['starttime']).dt.total_seconds() / 60
df['starttime'] = df['starttime'].dt.hour + df['starttime'].dt.minute / 60
df = df.drop("endtime", axis=1)
# 6.删除包含null数据的行
df = df.dropna()

Y_orgin = df[['positionmark_real']]
X_orgin = df.drop("positionmark_real", axis=1)
# print(X_orgin)
# print(Y_orgin)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X_orgin.values, Y_orgin.values)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# 归一化数据
scaler = StandardScaler().fit(X_train)
# print(scaler.mean_, scaler.scale_)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(X_train)


def estimator_dnn():
    # tf.logging.set_verbosity(tf.logging.INFO)

    # 定义COLUMNS
    COLUMNS = X_orgin.columns
    LABELS_COLUMNS = Y_orgin.columns
    # print(COLUMNS, LABELS_COLUMNS)

    def get_input_fn(x, y, num_epochs=None, shuffle=True):
        x_dict = {COLUMNS[i]: x[:, i] for i in range(len(COLUMNS))}
        return tf.estimator.inputs.numpy_input_fn(
            x=x_dict, y=y,
            num_epochs=num_epochs,
            shuffle=shuffle)

    # Feature cols
    feature_cols = [tf.feature_column.numeric_column(k) for k in COLUMNS]

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.estimator.DNNClassifier(feature_columns=feature_cols,
                                           hidden_units=[100, 100],
                                           model_dir="e:/python/model")

    # Train
    regressor.train(input_fn=get_input_fn(X_train, Y_train), steps=5000)

    # Evaluate loss over one epoch of test_set.
    ev = regressor.evaluate(input_fn=get_input_fn(
        X_test, Y_test, num_epochs=1, shuffle=False))
    print(ev)


def fcn():
    INPUT_NODE = X_train.shape[1]
    OUTPUT_NODE = Y_train.shape[1]
    BATCH_SIZE = 100
    LEARNING_RATE_BASE = 0.8
    LEARNING_RATE_DECAY = 0.99
    REGULARIZATION_RATE = 0.0001
    TRAINING_STEPS = 10000
    MOVING_AVERAGE_DECAY = 0.99
    MODEL_SAVE_PATH = "e:/python/fc_model/"
    MODEL_NAME = "model.ckpt"

    def next_batch(batch_size, train_step):
        global shuffle_index, shuffle_X, shuffle_Y
        if X_train.shape[0] != Y_train.shape[0]:
            return X_train, Y_train
        total_train_step = (X_train.shape[0] + batch_size - 1) // batch_size
        shuffle_index = np.arange(0, X_train.shape[0])
        shuffle_X, shuffle_Y = X_train, Y_train
        if train_step % total_train_step == 0:
            np.random.shuffle(shuffle_index)
            shuffle_X = X_train[shuffle_index]
            shuffle_Y = Y_train[shuffle_index]
        train_step = train_step % total_train_step
        start = train_step * batch_size
        end = min((train_step + 1) * batch_size, X_train.shape[0])
        return shuffle_X[start:end], shuffle_Y[start:end]

    def inference(x, hidden_unit, regularizer):
        def get_weight_variable(shape, regularizer):
            weights = tf.get_variable(
                "weights", shape,
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer is not None:
                tf.add_to_collection('losses', regularizer(weights))
            return weights

        layers_num = len(hidden_unit)
        layers_unit = [x.shape[1]] + hidden_unit

        layer = x
        for i in range(layers_num):
            with tf.variable_scope('layer' + str(i + 1)):
                weights = get_weight_variable(
                    [layers_unit[i], layers_unit[i + 1]], regularizer)
                biases = tf.get_variable(
                    "biases", [layers_unit[i + 1]], initializer=tf.constant_initializer(0.0))
                if i == layers_num - 1:
                    layer = tf.matmul(layer, weights) + biases
                else:
                    layer = tf.nn.relu(tf.matmul(layer, weights) + biases)
        return layer

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, [100, OUTPUT_NODE], regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    tf.summary.scalar('cross_entropy', loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, X_train.shape[0] / BATCH_SIZE,
        LEARNING_RATE_DECAY, staircase=True)
    train_step = tf.train.AdamOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    summary = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(MODEL_SAVE_PATH, MODEL_NAME), sess.graph)
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = next_batch(BATCH_SIZE, i)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 100 == 0:
                summary_str = sess.run(summary, feed_dict={x: xs, y_: ys})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (
                    step, loss_value))
                accuracy_score = sess.run(
                    accuracy, feed_dict={x: X_test, y_: Y_test})
                print("After %d training step(s), validation accuracy = %g" % (
                    step, accuracy_score))


fcn()


# AttributeError: module 'pandas' has no attribute 'computation'
# pip install --upgrade dask
