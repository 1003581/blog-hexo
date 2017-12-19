---
title: TensorFlow 损失函数
date: 2017-12-05 12:00:00
tags: tensorflow
categories: tensorflow
---

介绍几个常用的损失函数
<!-- more -->

参考链接：

- [神经网络Loss损失函数总结](http://blog.csdn.net/willduan1/article/details/73694826)
- [TensorFlow四种Cross Entropy算法实现和应用](http://geek.csdn.net/news/detail/126833)

# softamx cross entropy loss

softmax 交叉熵损失函数是我们常用的一种损失函数。

Softmax本身的算法很简单，就是把所有值用e的n次方计算出来，求和后算每个值占的比率，保证总和为1，一般我们可以认为Softmax出来的就是confidence也就是概率。

公式如下：

$$
J(W,b) = - \sum_i^m{y^{(i)}\log{a^{(i)}}}
$$

$$
a^{(i)} = softmax(z^{(i)}) = \frac{e^{z^{(i)}}}{\sum_i{e^{z^{(i)}}}}
$$

TensorFlow代码：

[tf.nn.softmax_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)

```python
softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```

[tf.nn.sparse_softmax_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits)

```python
sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```

这两个函数内部实现算法一样，区别是输入参数的不同，前者要求输入参数`labels`和`logits`具有相同的shape，都必须为`[batch_size, num_classes]`。而后者对`logits`的要求没变，对`labels`的要求变为了`[batch_size]`，而且值必须是从0开始编码的int32或int64，而且值范围是[0, num_class)，如果我们从1开始编码或者步长大于1，会导致某些label值超过这个范围，代码会直接报错退出。

这两个函数由于使用了Softmax，所以都必须是二分类或者多分类问题，目标标签只能有一类有效。这种标签称为`onehot-encoding`。

## sigmoid cross entropy loss

公式如下：

```python
targets * -log(sigmoid(logits)) +
    (1 - targets) * -log(1 - sigmoid(logits))
```

[tf.nn.sigmoid_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)

```python
sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```

[tf.nn.weighted_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits)

```python
weighted_cross_entropy_with_logits(
    targets,
    logits,
    pos_weight,
    name=None
)
```

这个函数的输入是`logits`和`labels`，`logits`就是神经网络模型中的最后一层，注意不需要经过`sigmoid`，两者的shape必须相同。

例如这个模型一次要判断100张图是否包含10种动物，这两个输入的shape都是[100, 10]。注释中还提到这10个分类之间是独立的、不要求是互斥，这种问题我们成为多目标，例如判断图片中是否包含10种动物，label值可以包含多个1或0个1。

对于`tf.nn.weighted_cross_entropy_with_logits`函数，可以理解为加权的`sigmoid_cross_entropy_with_logits`，正样本算出的值乘以某个系数, 公式如下：

```python
targets * -log(sigmoid(logits)) * pos_weight +
    (1 - targets) * -log(1 - sigmoid(logits))
```

# 均方差（MSE，mean squared error）

适用于回归问题的损失函数。公式如下:

$$
MSE(y, a) = \frac{\sum_{i=1}^n{(y-a)^2}}{n}
$$

# 自定义损失函数

假设要如下如下自定义的损失函数：

$$
Loss(y, a) = \sum_{i=1}^n{f(y^{(i)},a^{(i)})}, f(x,y)= \left\{ \begin {matrix} a(x-y), x<y \\ b(y-x), x>y  \end {matrix} \right.
$$

```python
loss= tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more,(y_-y)*loss_less))
```

`tf.greater(x,y)`，返回x>y的判断结果的bool型tensor，当tensor x, y的维度不一致时，采取广播（broadcasting）机制。

`tf.where(condition,x=None, y=None, name=None)`，根据condition选择x (if true) or y (if false)。