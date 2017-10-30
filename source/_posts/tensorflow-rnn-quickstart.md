---
title: TensorFlow RNN入门
date: 2017-10-30 17:00:00
tags: tensorflow
categories: tensorflow
---

循环神经网络, Recurrent neural network, RNN
<!-- more -->

# RNN简介

起源于1982年的霍普菲尔德网络。

RNN的主要用途是处理和预测序列数据。应用于语音识别、语言模型、机器翻译以及时序分析等问题。

CNN中隐藏层中的节点是无连接的。RNN隐藏层的输入不仅包括输入层的输出，也包括上一时刻隐藏层的输出。

## 前向传播

损失函数为所有时刻上损失函数的总和。

使用numpy库模拟前向传播

示意图：

![]()

代码：

```python
import numpy as np

# 不同时刻的输入，此处为2个时刻
X = [1, 2]
# 初始的状态，即上一个循环体的输出
state = [0.0, 0.0]

# 循环体参数state为state的系数，input为X的系数
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 用于当前时刻从循环体输出的参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + \
        X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print("before activation: ", before_activation)
    print("state: ", state)
    print("output: ", final_output)
```

> 当循环体过多时，会发生梯度消失问题。

# LTSM

RNN中一个重要结构：长短时记忆网络, long short-term memory, LSTM

为了解决长期依赖问题(long-term dependencies)


