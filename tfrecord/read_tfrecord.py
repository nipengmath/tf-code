import tensorflow as tf

# 创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(["output.tfrecords"])

# 从文件中读取一个样例。也可以使用read_up_to一次性读取多个样例
_, serialized_example = reader.read(filename_queue)

# 解析读入的一个样例。如果需要解析多个样例，可以用parse_example函数
features = tf.parse_single_example(
    serialized_example,
    features={
        # TensorFLow提供两种不同的属性解析方法。一种是tf.FixedLenFeature
        # 这种解析方法的结果为一个Tensor
        # 另一种方法为tf.VarLenFeature，这种方法得到的解析结果为SparseTensor，用于处理稀疏数据
        # 这里解析数据的格式需要与上面程序中的格式一致
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# tf.decode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行读取文件中一个样例，当所有样例读取完毕后，程序后重头开始读取
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
