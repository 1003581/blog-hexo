---
title: TensorFlot图像预处理常用手段
date: 2017-10-27 17:00:00
tags: tensorflow
categories: tensorflow
---

图像预处理

<!-- more -->

```python
import matplotlib.pyplot as plt
import tensorflow as tf   
import numpy as np


# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg",'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    
    # 输出解码之后的三维矩阵。
    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()

```