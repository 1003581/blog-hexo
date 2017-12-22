---
title: TensorFlow中的变量管理
date: 2017-09-25 10:00:00
tags: 
- tensorflow
categories: tensorflow
---

介绍TensorFlow中的变量管理。

<!-- more -->

# 相关链接

[莫烦Python视频讲解](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-12-scope/)

[官网教程](https://www.tensorflow.org/programmers_guide/variables#sharing_variables)

[name_scope和variable_scope区别](http://sentiment-mining.blogspot.com/2016/12/tensorflow-name-scope-variable-scope.html)

# 变量管理

本文主要介绍`tf.get_variable()`和`tf.variable_scope()`来进行变量管理。

## 两种变量创建方式

`tf.Variable()`的函数原型：

```python
__init__(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None
)
```

`tf.get_variable()`的函数原型：

```python
get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None
)
```

常见创建方式：

```python
import tensorflow as tf

v1 = tf.get_variable("v1", shape=[1], initializer=tf.constant_initializer(1.0))
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name="v2")
```

v1,v2结构如下：

```
<tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'v2:0' shape=(1,) dtype=float32_ref>
```

**以上两种定义方式是等价的**

`tf.get_variable()`函数中初始化器列表，见[官网](https://www.tensorflow.org/versions/master/api_docs/python/tf/initializers)

`tf.get_variable()`中`name`为必填字段

## 结合variable_scope

使用命名空间创建变量，且重复获得变量的代码如下：

```python
import tensorflow as tf

# 创建变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 利用reuse参数来获得该变量
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])

print(v == v1) # 输出为True

with tf.variable_scope("",reuse=True):
    v2 = tf.get_variable("foo/v", [1])

print(v == v2) # 输出为True
```

## 结合variable_scope嵌套使用

代码如下：

```python
import tensorflow as tf

with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)
    
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)
        
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)
            
    print(tf.get_variable_scope().reuse)
```

输出如下：

```
False
True
True
False
```
