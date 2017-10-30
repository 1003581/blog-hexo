---
title: TensorFlow 图像预处理常用手段
date: 2017-10-27 17:00:00
tags: tensorflow
categories: tensorflow
---

图像代码文档[tensorflow.org](https://www.tensorflow.org/api_guides/python/image#Working_with_Bounding_Boxes)
<!-- more -->

图像地址[qiniu](http://outz1n6zr.bkt.clouddn.com/127127207ede00bc9599d754cdf78be0.jpg)

## 单一处理

### 编解码

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
        # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 输出解码之后的三维矩阵。
    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()

    # 重新编码
    # 将数据转化为实数，以方便下面的示例程序对图像进行处理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

    encodeed_image = tf.image.encode_png(img_data)
    with tf.gfile.GFile("cat.png", "wb") as f:
        f.write(encodeed_image.eval())
```

### 大小调整

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 直接缩放成任意大小
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    plt.imshow(resized.eval())
    plt.show()

    # 使用裁剪和填充
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)
    plt.imshow(croped.eval())
    plt.show()

    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    plt.imshow(padded.eval())
    plt.show()

    # 通过比例裁剪
    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.show()
```

附`resize_images`中method的取值

- 0: 双线性插值法
- 1: 最近邻法
- 2: 双三次插值法
- 3: 面积插值法

### 图像翻转

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 上下翻转
    flipped = tf.image.flip_up_down(img_data)
    plt.imshow(flipped.eval())
    plt.show()

    # 左右翻转
    flipped = tf.image.flip_left_right(img_data)
    plt.imshow(flipped.eval())
    plt.show()

    # 左右翻转
    transposed = tf.image.transpose_image(img_data)
    plt.imshow(transposed.eval())
    plt.show()

    # 概率上下翻转
    flipped = tf.image.random_flip_up_down(img_data)
    plt.imshow(flipped.eval())
    plt.show()

    # 概率左右翻转
    flipped = tf.image.random_flip_left_right(img_data)
    plt.imshow(flipped.eval())
    plt.show()
```

### 图像色彩调整

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 亮度调整 -0.5
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 亮度调整 +0.5
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 亮度随机调整
    adjusted = tf.image.random_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 对比度调整 -5
    adjusted = tf.image.adjust_contrast(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 对比度调整 +5
    adjusted = tf.image.adjust_contrast(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 对比度随机调整
    adjusted = tf.image.random_contrast(img_data, 1, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 色相调整 0.1
    adjusted = tf.image.adjust_hue(img_data, 0.1)
    plt.imshow(adjusted.eval())
    plt.show()

    # 色相调整 0.5
    adjusted = tf.image.adjust_hue(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 色相随机调整
    adjusted = tf.image.random_hue(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 饱和度调整 -5
    adjusted = tf.image.adjust_saturation(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 饱和度调整 +5
    adjusted = tf.image.adjust_saturation(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 饱和度随机调整
    adjusted = tf.image.random_saturation(img_data, 1, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 均值为0，方差为1
    adjusted = tf.image.per_image_standardization(img_data)
    plt.imshow(adjusted.eval())
    plt.show()
```

### 处理标注框

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data, dtype=tf.float32), 0)

    # [y_min, x_min, y_max, x_max]
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    result = tf.image.draw_bounding_boxes(batched, boxes)
    result = tf.reshape(result, tf.shape(img_data))
    plt.imshow(result.eval())
    plt.show()
```

随机选择标注框并进行裁剪

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    # [y_min, x_min, y_max, x_max]
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)

    print(bbox_for_draw.eval())

    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()
```

## 实际训练中的随机处理

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 随机调整图片的色彩，定义两种顺序
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


# 对图片进行预处理，将图片转化成神经网络的输入层数据
def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机的截取图片中一个块。
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width], method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", "rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    for i in range(9):
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()

```