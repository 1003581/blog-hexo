---
title: TensorFlow TFRecord格式
date: 2017-10-26 17:00:00
tags: tensorflow
categories: tensorflow
---

TFRecord
<!-- more -->

# 格式介绍

低层采用Protocol Buffer格式存储，Protocol见[blog](http://liqiang311.com/tensorflow-saver/)

# 将MNIST数据转化为TFRecord

[mnist_to_tfrecord.py](https://github.com/liqiang311/tf-code/blob/master/tfrecord/mnist_to_tfrecord.py)

转换前MNIST_data文件夹大小12M，转换后record文件46M

# 读取Record文件

[read_tfrecord.py](https://github.com/liqiang311/tf-code/blob/master/tfrecord/read_tfrecord.py)