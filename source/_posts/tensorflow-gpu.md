---
title: TensorFlow GPU加速
date: 2017-10-31 16:00:00
tags: tensorflow
categories: tensorflow
---

TensorFlow GPU加速
<!-- more -->

# 单卡

尽管机器上多个CPU，但是对于TF来说，所有的CPU都是`/cpu:0`

多个GPU时，设备名称为`/gpu:n`，n从0开始

查看运行每一个操作的设备

CPU上

```python
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b

# 通过log_device_placement参数来记录运行每一个运算的设备。
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
```

输出

```
Device mapping: no known devices.
add: (Add): /job:localhost/replica:0/task:0/cpu:0
b: (Const): /job:localhost/replica:0/task:0/cpu:0
a: (Const): /job:localhost/replica:0/task:0/cpu:0
[ 2.  4.  6.]
```

GPU版本

```python
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b

with tf.device("/cpu:0"):
    d = tf.constant([1.0, 2.0, 3.0], shape=[3], name='d')
    e = tf.constant([1.0, 2.0, 3.0], shape=[3], name='e')

with tf.device("/gpu:1"):
    f = d + e

# 通过log_device_placement参数来记录运行每一个运算的设备。
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
    print(sess.run(f))
```

输出

```
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla P100-SXM2-16GB, pci bus id: 0000:0a:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla P100-SXM2-16GB, pci bus id: 0000:0b:00.0
add_1: (Add): /job:localhost/replica:0/task:0/gpu:1
add: (Add): /job:localhost/replica:0/task:0/gpu:0
e: (Const): /job:localhost/replica:0/task:0/cpu:0
d: (Const): /job:localhost/replica:0/task:0/cpu:0
b: (Const): /job:localhost/replica:0/task:0/gpu:0
a: (Const): /job:localhost/replica:0/task:0/gpu:0
[ 2.  4.  6.]
[ 2.  4.  6.]
```

一些操作无法被放到GPU上，但是可以通过配置来智能的将不能放到GPU上的放到CPU上

```python
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
```

# 并行模式

同步和异步，并行环节为计算和参数更新。

深度学习模型训练流程图

![]()

异步模式深度学习模型训练流程图

![]()

可以简单的认为异步模式就是单机模式复制了多份，每一份使用不同的训练数据进行训练。不同设备之间是完全独立的。

存在的问题：多个设备对参数进行更新，可能会更新过头。如下图，黑球为初始状态，2个设备读取到此时的参数，均使黑球向左移动，而由于不同步问题，设备1先使得黑球移动到灰球，然后设备2再让灰球移动了白球位置。使得无法达到最优点。

同步模式深度学习模型训练流程图

![]()

所有设备会同时读取参数的取值，然后等待所有的设备的传播算法计算完成后，统一更新参数（取平均值）。

![]()

缺点：受所有设备中最慢的设备制约。

# 一机多卡

因为一机上的GPU卡性能相似，所以采用同步方式。

关于数据存储，因为需要为不同的GPU提供不同的训练数据，所以通过placeholder的方式就需要手动准备多份数据。为了方便训练数据的获取过程，可以采用输入队列的方式从TFRecord中读取数据，于是在这里提供的数据文件路径为将MNIST训练数据转化为TFRecords格式后的路径，如何将MNIST转化为TFRecord见[TensorFlow TFRecord格式](http://liqiang311.com/2017/10/tensorflow-tfrecord/)

关于损失函数，对于给定的训练数据、正则化损失计算规则和命名空间，计算在这个命名空间下的总损失。之所以需要给定命名空间是因为不同的GPU上计算得出的正则化损失都会加入名为loss的集合，如果不通过命名空间就会将不同的GPU上的正则化损失都加进来。

```python
from datetime import datetime
import os
import time
import tensorflow as tf
import mnist_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 定义训练神经网络时需要用到的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 2

# 定义日志和模型输出的路径。
MODEL_SAVE_PATH = "logs_and_models/"
MODEL_NAME = "model.ckpt"
DATA_PATH = "output.tfrecords"


# 定义输入队列得到训练数据
def get_input():
    filename_queue = tf.train.string_input_producer([DATA_PATH])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 定义数据解析格式。
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # 解析图片和标签信息。
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [784])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # 定义输入队列并返回。
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    return tf.train.shuffle_batch(
        [retyped_image, label],
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)


# 定义损失函数。
def get_loss(x, y_, regularizer, scope, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y = mnist_inference.inference(x, regularizer)
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # 计算当前命名空间（设备）上的损失
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss
    return loss


# 计算每一个变量梯度的平均值。
def average_gradients(tower_grads):
    average_grads = []

    # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
    for grad_and_vars in zip(*tower_grads):
        # 计算所有GPU上的梯度平均值。
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # 将变量和它的平均梯度对应起来。
        average_grads.append(grad_and_var)
    # 返回所有变量的平均梯度，这个将被用于变量的更新。
    return average_grads

# 主训练过程。


def main(argv=None):
    # 将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上。
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # 定义基本的训练过程
        x, y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                   global_step,
                                                   60000 / BATCH_SIZE,
                                                   LEARNING_RATE_DECAY)

        opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        reuse_variables = False
        # 将神经网络的优化过程跑在不同的GPU上。
        for i in range(N_GPU):
            # 将优化过程指定在一个GPU上。
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    cur_loss = get_loss(x, y_, regularizer,
                                        scope, reuse_variables)
                    # 第一次声明变量之后，将控制变量重用的参数设置为True
                    # 这样可以让不同的GPU更新同一组参数。
                    reuse_variables = True

                    # 使用当前GPU计算所有变量的梯度
                    grads = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)

        # 计算变量的平均梯度。
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(
                    'gradients_on_average/%s' % var.op.name, grad)

        # 使用平均梯度更新参数。
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # 计算变量的滑动平均值。
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)
        # 每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True)) as sess:
            # 初始化所有变量并启动队列。
            init.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

            for step in range(TRAINING_STEPS):
                # 执行神经网络训练操作，并记录训练操作的运行时间。
                start_time = time.time()
                _, loss_value = sess.run([train_op, cur_loss])
                duration = time.time() - start_time

                # 每隔一段时间数据当前的训练进度，并统计训练速度。
                if step != 0 and step % 10 == 0:
                    # 计算使用过的训练数据个数。
                    num_examples_per_step = BATCH_SIZE * N_GPU
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / N_GPU

                    # 输出训练信息。
                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                    # 通过TensorBoard可视化训练过程。
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)

                # 每隔一段时间保存当前的模型。
                if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
```

# 分布式Tensorflow

## 原理

最简单的本地版单机集群

```python
import tensorflow as tf

# 创建一个本地集群。
c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
print(sess.run(c))
```

输出如下：

```
I] Initialize GrpcChannelCache for job local -> {0 -> localhost:57480}
I] tarted server with target: grpc://localhost:57480
I] Start master session f276282b48feaadf with config:
b'Hello, distributed TensorFlow!'
```

上述代码中，首先通过`tf.train.Server.create_local_server`函数在本地建立了一个只有一台机器的TF集群。然后在该集群上生成了一个会话，然后通过会话将运算运行在TF集群上。

TF集群通过一系列的任务(tasks)来执行TF计算图中的运算。

TF集群中的任务被聚合成工作(jobs)，每个工作可以包含一个或者多个任务。比如在训练深度学习模型时，一台运行反向传播的机器是一个任务，而所有运行反向传播机器的集合是一种工作。

本地运行两个任务的TF集群代码如下：

- 生成一个有两个任务的集群，一个任务跑在本地2222端口，一个跑在2223端口
- 通过集群配置生成Server，并通过`job_name`和`task_index`指定当前所启动的任务
- 通过server.target生成会话来使用TensorFlow集群中的资源。

代码1

```python
import tensorflow as tf
c = tf.constant("Hello from server1!")

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)

sess = tf.Session(server.target, config=tf.ConfigProto(
    log_device_placement=True))
print(sess.run(c))
server.join()
```

代码2

```python
import tensorflow as tf
c = tf.constant("Hello from server2!")

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)

sess = tf.Session(server.target, config=tf.ConfigProto(
    log_device_placement=True))
print(sess.run(c))
server.join()
```

输出1

```
I] Initialize GrpcChannelCache for job local -> {0 -> localhost:2222, 1 -> localhost:2223}
I] Started server with target: grpc://localhost:2222
I] CreateSession still waiting for response from worker: /job:local/replica:0/task:1
I] Start master session a8ebceffea7be4f9 with config: log_device_placement: true
Const: (Const): /job:local/replica:0/task:0/cpu:0
I] Const: (Const)/job:local/replica:0/task:0/cpu:0
b'Hello from server1!'
```

输出2

```
I] Initialize GrpcChannelCache for job local -> {0 -> localhost:2222, 1 -> localhost:2223}
I] Started server with target: grpc://localhost:2223
I] Start master session af6a798759682e38 with config: log_device_placement: true
Const: (Const): /job:local/replica:0/task:0/cpu:0
I] Const: (Const)/job:local/replica:0/task:0/cpu:0
b'Hello from server2!'
```

> 代码运行后无法自动结束，也无法ctrl+c结束，必须手动kill

ps-worker架构

一般在做深度学习训练时，会定义两个工作，一个工作专门负责存储、获取以及更新变量的取值，这个工作中的所有任务称为参数服务器(parameter server ,ps)，另外一个工作负责运行反向传播算法来获取参数梯度，这个工作中的所有任务被称之为计算服务器(worker)。

常见集群配置方法

```python
tf.train,ClusterSpec({
    "worker": [
        "tf-worker0:2222",
        "tf-worker1:2222",
        "tf-worker2:2222"
    ],
    "ps": [
        "tf-ps0:2222",
        "tf-ps1:2222"
    ]})
```

> 其中的`tf-worker(i)`和`tf-ps(i)`均为服务器地址

使用分布式TF训练深度学习时，一般有两种方式：计算图内分布式(in-graph relication)和计算图间分布式(between-graph relication)。

图内分布式表示所有的任务参数使用的是一份，只不过运算被分发到了其他GPU上。

图间分布式表示在每个服务器上创建一个独立的计算图，一些相同参数需要统一放到参数服务器上去。

## 模型训练


分布式代码

```python
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名。
MODEL_SAVE_PATH = "log_async"
DATA_PATH = "MNIST_data"

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


# 定义TensorFlow的计算图，并返回每一轮迭代时需要运行的操作。
def build_model(x, y_, is_chief):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 通过和5.5节给出的mnist_inference.py代码计算神经网络前向传播的结果。
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 计算损失函数并定义反向传播过程。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        60000 / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    # 定义每一轮迭代需要运行的操作。
    if is_chief:
        # 计算变量的滑动平均值。
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()
    return global_step, loss, train_op


def main(argv=None):
    # 解析flags并通过tf.train.ClusterSpec配置TensorFlow集群。
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
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
    device_setter = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id, cluster=cluster)
    with tf.device(device_setter):

        # 定义输入并得到每一轮迭代需要运行的操作。
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        global_step, loss, train_op = build_model(x, y_, is_chief)

        # 定义用于保存模型的saver。
        saver = tf.train.Saver()
        # 定义日志输出操作。
        summary_op = tf.summary.merge_all()
        # 定义变量初始化操作。
        init_op = tf.global_variables_initializer()
        # 通过tf.train.Supervisor管理训练深度学习模型时的通用功能。
        # tf.train.Supervisor能统一管理队列操作、模型保存、日志输出以及会话的生成
        sv = tf.train.Supervisor(
            is_chief=is_chief,          # 定义当前计算服务器是否为祝计算服务器
                                        # 只有主服务器会保存模型以及输出日志
            logdir=MODEL_SAVE_PATH,     # 指定保存模型和输出日志的地址
            init_op=init_op,            # 指定初始化操作
            summary_op=summary_op,      # 指定日志生成操作
            saver=saver,                # 指定用于保存模型的saver
            global_step=global_step,    # 指定当前迭代的轮次，这个会用于保存模型文件
            save_model_secs=60,         # 指定保存模型的时间间隔
            save_summaries_secs=60)     # 指定日志输出的时间间隔

        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        # 通过tf.train.Supervisor生成会话。
        sess = sv.prepare_or_wait_for_session(
            server.target, config=sess_config)

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
                sec_per_batch = duration / global_step_value
                format_str = 'After %d training steps (%d global steps), '
                'loss on training batch is %g.  (%.3f sec/batch)'
                print(format_str %
                      (step, global_step_value, loss_value, sec_per_batch))
            step += 1
    sv.stop()


if __name__ == "__main__":
    tf.app.run()
```