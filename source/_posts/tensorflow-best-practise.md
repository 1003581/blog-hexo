---
title: TensorFlow最佳实践代码
date: 2017-09-26 09:00:00
tags: 
- tensorflow
categories: tensorflow
---

神经网络中一些重构后的代码。

<!-- more -->

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