---
title: 神经网络常用优化方法(TensorFlow)
date: 2017-09-21 14:29:57
tags: 
- ML
- tensorflow
categories: tensorflow
---

介绍一些神经网络中常用的优化方法。包括动态学习率、正则化防止过拟合、滑动平均模型。

<!-- more -->
# 优化方法

## 学习率的设置

TensorFlow提供了一种学习率设置方法——指数衰减法。全部方法见[Decaying_the_learning_rate](https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate)

`tf.train.exponential_decay`函数先使用较大的学习率来快速得到较优解，然后随着训练步数的增多，学习率逐步降低，最后进行微调。

该函数的官网介绍：[exponential_decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)、[GitHub介绍](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/learning_rate_decay.py)。

函数定义如下：

```python
exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)
```

该函数通过步数动态的计算学习率，计算公式如下：

```python
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
```

其中：

- `decayed_learning_rate`: 每一轮优化时采用的学习率
- `learning_rate`: 实现设定的初始学习率
- `decay_rate`: 衰减系数
- `decay_steps`: 衰减速度

`staircase`如果为`True`，则`global_step / decay_steps`会转化为整数，这使得学习率变化函数从平滑曲线变为了阶梯函数（staircase function）。

示例代码如下：

```python
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,     # 
    global_step,
    mnist.train.num_examples / BATCH_SIZE,
    LEARNING_RATE_DECAY
    )
```

完整代码见后文。

## 通过正则化来避免过拟合问题

有关过拟合，即通过训练得到的模型不具有通用性，在测试集上表现不佳。

为了避免过拟合，常用方法有dropout、正则化（regularization）。

正则化的思想就是在损失函数中加入刻画模型复杂度的指标。如何刻画复杂度，一般做法是对权重W进行正则化。

正则化包含L1正则化和L2正则化。多用L2正则化。原文对比见[github](https://github.com/caicloud/tensorflow-tutorial/blob/master/Deep_Learning_with_TensorFlow/1.0.0/Chapter04/3.%20%E6%AD%A3%E5%88%99%E5%8C%96.ipynb)

> L2比L1好用的原因：
> 1. L1使得参数变得更稀疏，即一些参数会变为0。L2会使得参数保持一个很小的数字，比如0.001。
> 2. L1正则化公式不可导，L2正则化公式可导。

L1和L2的测试代码如下：

```python
import tensorflow as tf
weights = tf.constant( [ [1.0, -2.0], [-3.0, 4.0] ] )
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))
```

输出如下：

```
5.0
7.5
```

计算方法如下：

```
L1 = (|1| + |-2| + |-3| + |4|) * 0.5 = 5
L2 = (1^2 + (-2)^2 + (-3)^2 + 4^2) / 2 * 0.5 = 7.5
```

> 0.5为正则化项的权重lambda。TensorFlow将L2的正则化损失值除以2使求导得到的结果更加简洁。

### 通过集合Collection解决层数过多时代码过长问题

思路：将所有的权重向量加入到一个集合中，最后累加这个集合中的变量。

示例，构建5层神经网络代码如下：

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = []
label = []
np.random.seed(0)

# 以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
for i in range(150):
    x1 = np.random.uniform(-1,1)
    x2 = np.random.uniform(0,2)
    if x1**2 + x2**2 <= 1:
        data.append([np.random.normal(x1, 0.1),np.random.normal(x2,0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)
        
data = np.hstack(data).reshape(-1,2)
label = np.hstack(label).reshape(-1, 1)

#plt.scatter(data[:,0], data[:,1], c=label, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
#plt.show()

BATCH_SIZE = 8        # batch size
LAYER_DIM = [2, 10, 10, 10, 1]  # 各层的神经元个数，首尾为输入和输出层
LAYER_NUM = len(LAYER_DIM)
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 40000

# 定义一个W，并将这个W的正则化值加入集合中
def get_weights(shape, rate):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32)
    # 将这个W加入到集合中, 第一个参数为集合名字，第二个参数为集合内容
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(rate)(weights))
    return weights

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 循环生成网络结构
layer = x
for i in range(LAYER_NUM-1):
    input_dim = LAYER_DIM[i]
    output_dim = LAYER_DIM[i+1]
    weights = get_weights([input_dim, output_dim], REGULARIZATION_RATE)
    biases = tf.Variable(tf.constant(0.1, shape=[output_dim]), dtype=tf.float32)
    layer = tf.nn.relu(tf.matmul(layer, weights) + biases)

# 损失函数的定义。
mse_loss = tf.reduce_mean(tf.square(y_ - layer))
tf.add_to_collection("losses", mse_loss)
loss = tf.add_n(tf.get_collection("losses"))

train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("./tmp", sess.graph)

    sess.run(tf.global_variables_initializer())
   
    for i in range(TRAINING_STEPS):
        sess.run(train_op, feed_dict={x: data, y_: label})
        if i % 2000 == 0:
            print("After %d steps, loss: %f" % (i, sess.run(loss, feed_dict={x: data, y_: label})))
```

## 滑动平均模型

在采用随即梯度下降算法训练时，使用滑动平均模型会在一定程度上提供最终模型在测试数据上的性能。

类`tf.train.ExponentialMovingAverage`([DOC](https://www.tensorflow.org/versions/master/api_docs/python/tf/train/ExponentialMovingAverage)、[GitHub](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/moving_averages.py))

初始化函数

```python
__init__(
    decay,
    num_updates=None,
    zero_debias=False,
    name='ExponentialMovingAverage'
)
```

其中`decay`为衰减率。用来控制模型更新的速度。区间为(0,1)。一般取`0.999`,`0.9999`等。越大模型越趋于稳定。

`num_updates`用来动态设置decay的大小。一般情况下可以用训练步数作为此参数。如果设置了该变量，则衰减率一开始会比较小，后期会越来越大。每次的衰减率计算公式为：`min(decay, (1 + num_updates) / (10 + num_updates))`

```
apply(var_list=None)
```

`ExponentialMovingAverage`通过调用`apply`会对参数列表中的每一个`Variable`维护一个影子变量(shadow variables)。影子变量的初始值是相应变量的初始值，当每次模型更新时，影子变量的值会更新为：`shadow_variable = decay * shadow_variable + (1 - decay) * variable`

文档示例代码

```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)

...train the model by running training_op...
```

其他示例见[Github](https://github.com/caicloud/tensorflow-tutorial/blob/master/Deep_Learning_with_TensorFlow/1.0.0/Chapter04/4.%20%E6%BB%91%E5%8A%A8%E5%B9%B3%E5%9D%87%E6%A8%A1%E5%9E%8B.ipynb)

# 实践——MINST

使用上文所属的三种方法来优化MNIST全连接神经网络，代码如下：

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数。
INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点

# 配置神经网络的参数
LAYER1_NODE = 500    # 隐藏层数       
                              
BATCH_SIZE = 100     # 每次batch打包的样本个数        

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        
MOVING_AVERAGE_DECAY = 0.99


```