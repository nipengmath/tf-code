import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名。
MODEL_SAVE_PATH = "log_async"
DATA_PATH = "MNIST_data"
# 和异步模式类似的设置flags。
FLAGS = tf.app.flags.FLAGS

# 指定当前程序是参数服务器还是计算服务器。
# 参数服务器负责TensorFlow中变量的维护和管理
# 计算服务器负责每一轮迭代时运行反向传播过程
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
# 指定集群中的参数服务器地址。
tf.app.flags.DEFINE_string(
    'ps_hosts', ' tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. '
    'e.g. "tf-ps0:2222,tf-ps1:1111" ')
# 指定集群中的计算服务器地址。
tf.app.flags.DEFINE_string(
    'worker_hosts', ' tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. '
    'e.g. "tf-worker0:2222,tf-worker1:1111" ')
# 指定当前程序的任务ID。
# TensorFlow会自动根据参数服务器/计算服务器列表中的端口号来启动服务。
# 注意参数服务器和计算服务器的编号都是从0开始的。
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')


# 和异步模式类似的定义TensorFlow的计算图。唯一的区别在于使用tf.train.SyncReplicasOptimizer函数处理同步更新。
def build_model(x, y_, n_workers, is_chief):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 计算损失函数并定义反向传播过程。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        60000 / BATCH_SIZE, LEARNING_RATE_DECAY)

    # 通过tf.train.SyncReplicasOptimizer函数实现同步更新。
    opt = tf.train.SyncReplicasOptimizer(
        # 定义基础的优化方法
        tf.train.GradientDescentOptimizer(learning_rate),
        # 定义每一轮需要多少个服务器的得出的梯度
        replicas_to_aggregate=n_workers,
        # 指定总共有多少计算服务器
        total_num_replicas=n_workers)
    train_op = opt.minimize(loss, global_step=global_step)

    # 定义每一轮迭代需要运行的操作。
    if is_chief:
        # 计算变量的滑动平均值。
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()

    return global_step, loss, train_op, opt


def main(argv=None):
    # 和异步模式类似的创建TensorFlow集群。
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    print('PS hosts are: %s' % ps_hosts)
    print('Worker hosts are: %s' % worker_hosts)
    n_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # 通过tf.train.ClusterSpec以及当前任务创建tf.train.Server。
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

    # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。server.join()会
    # 一致停在这条语句上。
    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()

    # 定义计算服务器需要运行的操作。
    # 在所有的计算服务器中又一个是主计算服务器
    # 它除了负责计算反向传播的结果，还负责日志和保存模块
    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    # 通过tf.train.replica_device_setter函数来指定执行每一个运算的设备。
    # tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，而
    # 计算分配到当前的计算服务器上，
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id, cluster=cluster)):

        # 定义输入并得到每一轮迭代需要运行的操作。
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        global_step, loss, train_op, opt = build_model(
            x, y_, n_workers, is_chief)
        # 和异步模式类似的声明一些辅助函数。
        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()

        init_op = tf.global_variables_initializer()

        # 在同步模式下，主计算服务器需要协调不同计算服务器计算得到的参数梯度并最终更新参数。
        # 这需要主计算服务器完成一些额外的初始化工作。
        if is_chief:
            # 获取协调不同计算服务器的队列。在更新参数之前，主计算服务器需要先启动这些队列。
            chief_queue_runner = opt.get_chief_queue_runner()
            # 初始化同步更新队列的操作。
            init_tokens_op = opt.get_init_tokens_op(0)

        # 和异步模式类似的声明tf.train.Supervisor。
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=MODEL_SAVE_PATH,
            init_op=init_op,
            summary_op=summary_op,
            saver=saver,
            global_step=global_step,
            save_model_secs=60,
            save_summaries_secs=60)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)

        sess = sv.prepare_or_wait_for_session(
            server.target, config=sess_config)

        # 在主计算服务器上启动协调同步更新的队列并执行初始化操作。
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        # 和异步模式类似的运行迭代的训练过程。
        step = 0
        start_time = time.time()

        # 执行迭代过程。在迭代过程中，Supervisor会帮助输出日志并保存模型，不需要直接调用
        while not sv.should_stop():
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, global_step_value = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if global_step_value >= TRAINING_STEPS:
                break

            # 每隔一段时间输出训练信息。
            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / (global_step_value * n_workers)
                format_str = ('After % d training steps ( % d global steps), '
                              'loss on training batch is % g.  '
                              '( % .3f sec/batch)')
                print(format_str %
                      (step, global_step_value, loss_value, sec_per_batch))
            step += 1
    sv.stop()


if __name__ == "__main__":
    tf.app.run()
