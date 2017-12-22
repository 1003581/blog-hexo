---
title: TensorFlow 多线程输入数据处理框架
date: 2017-10-30 17:00:00
tags: tensorflow
categories: tensorflow
---

TensorFlow 多线程输入数据处理框架
<!-- more -->

## 队列

- `tf.FIFOQueue`：先进先出队列
- `tf.RandomShuffleQueue`：每次出队都是随机选择一个

```python
import tensorflow as tf

# 创建一个先进先出队列，指定队列中最多存放5个元素，并指定相应的类型
q = tf.FIFOQueue(5, "int32")
# 使用enqueue_many来初始化队列中的元素，需要sess.run
init = q.enqueue_many(([1, 2, 3, 4, 5],))
# 使用dequeue来将队列中第一个元素出队
x = q.dequeue()
# 定义运算并将运算结果入队
q_inc = q.enqueue([x * x])

with tf.Session() as sess:
    init.run()
    for _ in range(10):
        v, _ = sess.run([x, q_inc])
        print(v)
```

## 多线程

- `tf.Coordinator`
    - 协同多个线程一起停止
    - 创建Coordinator对象后，将该对象传入每个线程中
    - 启动的线程会定期轮询`should_stop`函数，若为True，则表示当前线程需要退出
    - 当运行中的某一个线程调用`request_stop`函数时，所有线程的`should_stop`返回值将为True
    - 主线程中使用`join`进行线程等待
- `tf.QueueRunner`

```python
import tensorflow as tf
import numpy as np
import threading
import time


# 线程中运行的程序，每隔1秒检查是否需要退出，并打印自己的id
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d" % worker_id)
            coord.request_stop()
        else:
            print("Working on id: %d" % worker_id)
        time.sleep(1)


coord = tf.train.Coordinator()
# 声明5个线程变量
threads = [threading.Thread(target=MyLoop, args=(coord, i, ))
           for i in range(5)]
for t in threads:
    t.start()
coord.join(threads)
```

```python
import tensorflow as tf


# 声明一个先进先出的队列
queue = tf.FIFOQueue(100, "float")
# 定义入队操作，在线程中会循环不断的入队
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
# tf.train.QueueRunner第一个参数为被操作的队列
# 第二个参数表示创建5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 将定义过的QueueRunner加入TF计算图上的指定集合中
# add_queue_runner默认加入tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 利用Coordinator进行线程同步
    coord = tf.train.Coordinator()
    # 使用QueueRunner时，需要明确调用以下函数来启动所有线程
    # 该函数会启动处于tf.GraphKeys.QUEUE_RUNNERS集合中的所有QueueRunner
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 获取队列中的值
    for _ in range(3):
        print(sess.run(out_tensor))

    coord.request_stop()
    coord.join(threads)

```

## 输入文件队列

模拟生成多个TFRecord格式文件

```python
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据情况下将数据写入不同的文件
num_shards = 2  # 总共写入多少文件
instances_per_shard = 2  # 每个文件中有多少数据

for i in range(num_shards):
    # 将数据分成多个文件时，以类似0000n-of-0000m的后缀区分，其中m表示总共多少文件
    # n表示当前是第几个文件
    filename = ('data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)

    # 将数据封装为Example结构并写入TFRecord
    for j in range(instances_per_shard):
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
```

处理多个TFRecord格式数据

```python
import tensorflow as tf

# 使用match_filenames_once来获取文件列表
files = tf.train.match_filenames_once("data.tfrecords-*")

# 通过string_input_producer函数创建输入队列，输入队列中的文件列表为
# tf.train.match_filenames_once函数获取的文件列表。
# 这里shuffle参数设为False来避免随机打乱读文件的数序，但在一般解决实际
# 问题时，该参数会设置为True
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 读取并解析一个样例
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)
```

## 组合训练数据(batching)

```python
import tensorflow as tf

files = tf.train.match_filenames_once("data.tfrecords-*")

filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })

# 假设Example中i为特征矩阵，j为label
example, label = features['i'], features['j']

# 一个batch中样例的个数
batch_size = 3
# 组合样例的队列中最多可以存储的样例个数，若队列太大，则浪费内存
# 若队列过小，则会导致出队时产生阻塞。
# 一般会取一个跟batch size有关的值
capacity = 1000 + 3 * batch_size

# 使用tf.train.batch来组合数据
# [example, label]指定了要组合的元素
# batch_size指batch大小
# capacity值队列大小，默认为32。当队列满时，暂停入队
# tf.train.shuffle_batch
example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)
```

- `tf.train.batch`
- `tf.train.shuffle_batch`
- `tf.train.shffle_batch_join`

通过指定以上函数中的num_threads参数，来指定用几个线程来操作入队动作，其中`tf.train.shuffle_batch`为多个线程读取相同的文件，`tf.train.shffle_batch_join`为不同的线程读取不同的文件。

## 输入数据处理框架

首先将MNIST数据集转换为TFRecord格式，代码见[blog](http://liqiang311.com/tensorflow-tfrecord/)

处理代码

```python
import tensorflow as tf

# 读取TFRecord格式文件
files = tf.train.match_filenames_once("output.tfrecords")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 读取文件。
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# 解析像素矩阵
decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)
images = tf.reshape(retyped_images, [784])

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch(
    [images, labels],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)


def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# 模型相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 5000

weights1 = tf.Variable(tf.truncated_normal(
    [INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal(
    [LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            print("After %d training step(s), loss is %g " %
                  (i, sess.run(loss)))

        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
```

处理流程

1. `tf.train.string_input_producer`
    1. 输入文件列表 {"A", "B", "C"}
    1. 随机打乱
    1. 输入文件队列 {"B", "B", "C", "A", " ", " "}
1. `tf.train.shuffle_batch`
    1. 生成多个Reader {"Reader1", "Reader2"}
    1. 分别进行数据预处理
    1. 样例组合队列 {"(a,0)", "(c,1)", "(b,0)", "(c,1)", "...", ...}
1. 生成Batch1、Batch2
