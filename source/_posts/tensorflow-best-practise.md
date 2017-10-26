---
title: TensorFlow最佳实践代码
date: 2017-09-26 09:00:00
tags: 
- tensorflow
categories: tensorflow
---

神经网络中一些重构后的代码。

<!-- more -->

# 标准

TOP5

# MNIST-FullConnected

分为三个文件

- `fc_inference.py`：定义了前向传播的过程以及神经网络中的参数。
- `fc_train.py`：定义了神经网络的训练过程。
- `fc_eval.py`：定义了测试过程。

## fc_inference.py

```python
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
```

## fc_train.py

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fc_inference
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="fc_model/"
MODEL_NAME="model.ckpt"


def train(mnist):

    x = tf.placeholder(tf.float32, [None, fc_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, fc_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = fc_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
```

## fc_eval.py

```python
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fc_inference
import fc_train

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, fc_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, fc_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = fc_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(fc_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(fc_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()
```

最高98.62%

# MNIST-LeNet5

## lenet5_inference.py

```python
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
```

卷积层的weights变量为四维矩阵，前面两维代表了过滤器的尺寸，第三个维度表示当前层的深度，第四个维度表示过滤器的深度。

卷积层的biases变量为一维向量，维度为过滤器的深度，即下一层节点矩阵的深度。

卷积层的Wx+b运算如下：先计算节点矩阵中某一块区域和一个过滤器的点积，然后加上这个过滤器的偏置。所以卷积层的参数个数为（`SAME`全零填充）：`(过滤器长×过滤器宽×当前层深度×过滤器深度)+过滤器深度(biases)`，下一层的节点数为`当前层长×当前层宽×过滤器深度`。

`tf.nn.conv2d`函数实现了卷积算法，第一个参数为当前层的节点矩阵，这个节点矩阵为四维矩阵，第一个维度代表图像id，后三个维度代表节点矩阵（长宽深）；第二个参数是卷积层的权重变量；第三个参数是不同维度上的步长，虽然第三个参数提供了长度为4的数组，但是第一维(id)和最后一维(图像深度)数字要求一定是1，这是因为卷积层的步长只对矩阵的长和宽有效。最后一个参数为填充(padding)方法，有2中选择：`SAME`和`VALID`，前者为全0填充，输出图像和输入图像长宽相同，后者为不填充，图像会有缩小。

`tf.nn.bias_add`函数使得给每一个计算后的点积都加上偏置项。官网教程中无此函数，直接使用的加法。

池化层中的`tf.nn.max_pool`参数与conv2d相似，ksize提供了过滤器的大小，strides提供了池化的步长，同样第一位和最后一维必须为1，第二和第三为长和宽。

## lenet5_train.py

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_inference
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="lenet5_model/"
MODEL_NAME="model.ckpt"

def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            lenet5_inference.IMAGE_SIZE,
            lenet5_inference.IMAGE_SIZE,
            lenet5_inference.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, lenet5_inference.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = lenet5_inference.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                lenet5_inference.IMAGE_SIZE,
                lenet5_inference.IMAGE_SIZE,
                lenet5_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
```

## lenet5_eval.py

```python
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_inference
import lenet5_train

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.validation.num_examples,
            lenet5_inference.IMAGE_SIZE,
            lenet5_inference.IMAGE_SIZE,
            lenet5_inference.NUM_CHANNELS],
            name='x-input')
        y_ = tf.placeholder(tf.float32, [None, lenet5_inference.OUTPUT_NODE], name='y-input')

        reshaped_xs = np.reshape(mnist.validation.images, (
                mnist.validation.num_examples,
                lenet5_inference.IMAGE_SIZE,
                lenet5_inference.IMAGE_SIZE,
                lenet5_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}

        y = lenet5_inference.inference(x, None, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(lenet5_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(lenet5_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()
```

验证集98.84%

# Inception-v3

[Github代码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py)

将用不同边长的过滤器通过全0填充的卷积层的卷局函数叠在一起。

Tensorflow-Slim(TODO)

# 迁移学习

数据下载地址[flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

5个子文件夹，每个文件夹代表1个花种，平均一种花有734张图像。RGB色彩，大小不相同。

google训练好的Inception-v3模型[inception_dec_2015.zip](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip)

处理压缩包

```
tar xzf flower_photos.tgz
unzip inception_dec_2015.zip -d inception_dec_2015
```

代码

```python
# -*- coding: utf-8 -*-
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。在谷歌提供的模型中，
# 这个张量名称就是'pool_3/_reshape:0'。在训练模型时，使用tensor.name来获取张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载好的google模型文件目录和模型文件名
MODEL_DIR = 'inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到
# 的特征向量保存到文件中，免去重复的计算。
CACHE_DIR = 'bottleneck'

# 图片数据文件夹
INPUT_DATA = 'flower_photos'

# 验证的数据百分比percentage
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 把样本中所有的图片列表并按训练、验证、测试数据分开
# testing_percentage和validation_percentage参数指定了测试数据集和验证数据集的大小。
def create_image_lists(testing_percentage, validation_percentage):
	# 得到的所有图像都存在result中
    result = {}
    # 获取图片目录下的所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
    	# 得到的第一个值为当前目录，跳过
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有有效文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名，或者类别的名称
        label_name = dir_name.lower()
        
        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    return result

# 通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists参数给出了所有图片信息
# image_dir参数给出了根目录。
# label_name参数给出了类别名称
# index参数给定了需要获取的图片的编号
# category参数指定了需要获取的图像是在训练数据集、测试数据集还是验证数据集
def get_image_path(image_lists, image_dir, label_name, index, category):
	# 获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    # 根据所述数据集的名称获取集合中全部图像信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图像的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终地址为数据根目录的地址加上类别的文件夹加上图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 获取Inception-v3模型处理之后的特征向量的文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

# 使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
	# 将当前图像作为输入，计算瓶颈张量的值。即该图像最新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
	# 获取一张图像对应的特征向量文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 若不存在，则计算模型
    if not os.path.exists(bottleneck_path):

        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)

        image_data = gfile.FastGFile(image_path, 'rb').read()

        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
    	# 直接从文件中获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

# 随机获取一个batch的图片作为训练数据。
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
    	# 随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

# 获取全部的测试数据，并计算正确率。
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
        	# 通过模型计算图像对应的特征向量，并加入最终数据列表中
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 定义主函数。
def main():
	# 读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    # 读取已经训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    
    # 定义一层全链接层
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
        
    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    
    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 训练过程。
        for i in range(STEPS):
 
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                    (i, BATCH, validation_accuracy * 100))
            
        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    main()
```

输出

```
Step 0: Validation accuracy on random sampled 100 examples = 59.0%
Step 100: Validation accuracy on random sampled 100 examples = 86.0%
Step 200: Validation accuracy on random sampled 100 examples = 85.0%
Step 300: Validation accuracy on random sampled 100 examples = 80.0%
Step 400: Validation accuracy on random sampled 100 examples = 87.0%
Step 500: Validation accuracy on random sampled 100 examples = 81.0%
Step 600: Validation accuracy on random sampled 100 examples = 85.0%
Step 700: Validation accuracy on random sampled 100 examples = 88.0%
Step 800: Validation accuracy on random sampled 100 examples = 88.0%
Step 900: Validation accuracy on random sampled 100 examples = 89.0%
Step 1000: Validation accuracy on random sampled 100 examples = 87.0%
Step 1100: Validation accuracy on random sampled 100 examples = 83.0%
Step 1200: Validation accuracy on random sampled 100 examples = 92.0%
Step 1300: Validation accuracy on random sampled 100 examples = 90.0%
Step 1400: Validation accuracy on random sampled 100 examples = 91.0%
Step 1500: Validation accuracy on random sampled 100 examples = 83.0%
Step 1600: Validation accuracy on random sampled 100 examples = 92.0%
Step 1700: Validation accuracy on random sampled 100 examples = 86.0%
Step 1800: Validation accuracy on random sampled 100 examples = 91.0%
Step 1900: Validation accuracy on random sampled 100 examples = 86.0%
Step 2000: Validation accuracy on random sampled 100 examples = 90.0%
Step 2100: Validation accuracy on random sampled 100 examples = 91.0%
Step 2200: Validation accuracy on random sampled 100 examples = 84.0%
Step 2300: Validation accuracy on random sampled 100 examples = 96.0%
Step 2400: Validation accuracy on random sampled 100 examples = 89.0%
Step 2500: Validation accuracy on random sampled 100 examples = 93.0%
Step 2600: Validation accuracy on random sampled 100 examples = 95.0%
Step 2700: Validation accuracy on random sampled 100 examples = 86.0%
Step 2800: Validation accuracy on random sampled 100 examples = 94.0%
Step 2900: Validation accuracy on random sampled 100 examples = 88.0%
Step 3000: Validation accuracy on random sampled 100 examples = 90.0%
Step 3100: Validation accuracy on random sampled 100 examples = 87.0%
Step 3200: Validation accuracy on random sampled 100 examples = 92.0%
Step 3300: Validation accuracy on random sampled 100 examples = 93.0%
Step 3400: Validation accuracy on random sampled 100 examples = 87.0%
Step 3500: Validation accuracy on random sampled 100 examples = 87.0%
Step 3600: Validation accuracy on random sampled 100 examples = 92.0%
Step 3700: Validation accuracy on random sampled 100 examples = 94.0%
Step 3800: Validation accuracy on random sampled 100 examples = 87.0%
Step 3900: Validation accuracy on random sampled 100 examples = 90.0%
Step 3999: Validation accuracy on random sampled 100 examples = 91.0%
Final test accuracy = 89.7%
```

