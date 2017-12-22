---
title: MultiNet：自主驾驶中的实时联合语义推理 论文代码
date: 2017-11-23 23:00:00
tags: 机器视觉
categories: 深度学习
---
MultiNet论文相关
<!-- more -->
论文下载地址:[原文地址](https://arxiv.org/pdf/1612.07695.pdf)、[免翻墙地址](http://outz1n6zr.bkt.clouddn.com/1612.07695.pdf)

论文Github地址：[KittiSeg](https://github.com/MarvinTeichmann/KittiSeg)

论文翻译参考：[csdn](http://blog.csdn.net/hanging_gardens/article/details/72724258)

# 程序

## Get Start

```shell
git clone https://github.com/MarvinTeichmann/KittiSeg.git
cd KittiSeg
git submodule update --init --recursive
python download_data.py --kitti_url http://kitti.is.tue.mpg.de/kitti/data_road.zip
```

下载过程

```shell
2017-11-24 13:41:12,536 INFO Downloading VGG weights.
2017-11-24 13:41:12,538 INFO Download URL: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
2017-11-24 13:41:12,538 INFO Download DIR: DATA
>> Downloading vgg16.npy 100.0%
2017-11-24 13:58:29,855 INFO Downloading Kitti Road Data.
2017-11-24 13:58:29,855 INFO Download URL: http://kitti.is.tue.mpg.de/kitti/data_road.zip
2017-11-24 13:58:29,856 INFO Download DIR: DATA
>> Downloading data_road.zip 100.0%
2017-11-24 14:46:04,978 INFO Extracting kitti_road data.
2017-11-24 14:46:09,240 INFO Preparing kitti_road data.
2017-11-24 14:46:09,244 INFO All data have been downloaded successful.
```

### 运行Demo

Windows下运行出错问题[解决](https://github.com/MarvinTeichmann/KittiSeg/issues/17)

```
python demo.py --input_image data/demo/demo.png
```

```
2017-11-27 14:12:16,096 INFO No environment variable 'TV_PLUGIN_DIR' found. Set to 'C:\Users\10217814/tv-plugins'.
2017-11-27 14:12:16,096 INFO No environment variable 'TV_STEP_SHOW' found. Set to '50'.
2017-11-27 14:12:16,096 INFO No environment variable 'TV_STEP_EVAL' found. Set to '250'.
2017-11-27 14:12:16,096 INFO No environment variable 'TV_STEP_WRITE' found. Set to '1000'.
2017-11-27 14:12:16,097 INFO No environment variable 'TV_MAX_KEEP' found. Set to '10'.
2017-11-27 14:12:16,097 INFO No environment variable 'TV_STEP_STR' found. Set to 'Step {step}/{total_steps}: loss = {loss_value:.2f}; lr = {lr_value:.2e}; {sec_per_batch:.3f} sec (per Batch); {examples_per_sec:.1f} imgs/sec'.
2017-11-27 14:12:16,107 INFO Download URL: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/KittiSeg_pretrained.zip
2017-11-27 14:12:16,108 INFO Download DIR: RUNS
>> Downloading KittiSeg_pretrained.zip 100.0%
2017-11-27 17:13:03,549 INFO Extracting KittiSeg_pretrained.zip
2017-11-27 17:13:24,930 INFO f: <_io.TextIOWrapper name='RUNS\\KittiSeg_pretrained\\model_files\\hypes.json' mode='r' encoding='cp936'>
2017-11-27 17:13:24,931 INFO Hypes loaded successfully.
2017-11-27 17:13:25,072 INFO Modules loaded successfully. Starting to build tf graph.
npy file loaded
Layer name: conv1_1
Layer shape: (3, 3, 3, 64)
2017-11-27 17:13:29,842 INFO Creating Summary for: conv1_1/filter
2017-11-27 17:13:29,890 INFO Creating Summary for: conv1_1/biases
Layer name: conv1_2
Layer shape: (3, 3, 64, 64)
2017-11-27 17:13:29,923 INFO Creating Summary for: conv1_2/filter
2017-11-27 17:13:29,947 INFO Creating Summary for: conv1_2/biases
Layer name: conv2_1
Layer shape: (3, 3, 64, 128)
2017-11-27 17:13:29,979 INFO Creating Summary for: conv2_1/filter
2017-11-27 17:13:30,004 INFO Creating Summary for: conv2_1/biases
Layer name: conv2_2
Layer shape: (3, 3, 128, 128)
2017-11-27 17:13:30,036 INFO Creating Summary for: conv2_2/filter
2017-11-27 17:13:30,061 INFO Creating Summary for: conv2_2/biases
Layer name: conv3_1
Layer shape: (3, 3, 128, 256)
2017-11-27 17:13:30,095 INFO Creating Summary for: conv3_1/filter
2017-11-27 17:13:30,119 INFO Creating Summary for: conv3_1/biases
Layer name: conv3_2
Layer shape: (3, 3, 256, 256)
2017-11-27 17:13:30,152 INFO Creating Summary for: conv3_2/filter
2017-11-27 17:13:30,175 INFO Creating Summary for: conv3_2/biases
Layer name: conv3_3
Layer shape: (3, 3, 256, 256)
2017-11-27 17:13:30,209 INFO Creating Summary for: conv3_3/filter
2017-11-27 17:13:30,232 INFO Creating Summary for: conv3_3/biases
Layer name: conv4_1
Layer shape: (3, 3, 256, 512)
2017-11-27 17:13:30,268 INFO Creating Summary for: conv4_1/filter
2017-11-27 17:13:30,293 INFO Creating Summary for: conv4_1/biases
Layer name: conv4_2
Layer shape: (3, 3, 512, 512)
2017-11-27 17:13:30,332 INFO Creating Summary for: conv4_2/filter
2017-11-27 17:13:30,356 INFO Creating Summary for: conv4_2/biases
Layer name: conv4_3
Layer shape: (3, 3, 512, 512)
2017-11-27 17:13:30,396 INFO Creating Summary for: conv4_3/filter
2017-11-27 17:13:30,422 INFO Creating Summary for: conv4_3/biases
Layer name: conv5_1
Layer shape: (3, 3, 512, 512)
2017-11-27 17:13:30,461 INFO Creating Summary for: conv5_1/filter
2017-11-27 17:13:30,485 INFO Creating Summary for: conv5_1/biases
Layer name: conv5_2
Layer shape: (3, 3, 512, 512)
2017-11-27 17:13:30,565 INFO Creating Summary for: conv5_2/filter
2017-11-27 17:13:30,589 INFO Creating Summary for: conv5_2/biases
Layer name: conv5_3
Layer shape: (3, 3, 512, 512)
2017-11-27 17:13:30,629 INFO Creating Summary for: conv5_3/filter
2017-11-27 17:13:30,655 INFO Creating Summary for: conv5_3/biases
Layer name: fc6
Layer shape: [7, 7, 512, 4096]
2017-11-27 17:13:30,965 INFO Creating Summary for: fc6/weights
2017-11-27 17:13:30,991 INFO Creating Summary for: fc6/biases
Layer name: fc7
Layer shape: [1, 1, 4096, 4096]
2017-11-27 17:13:31,067 INFO Creating Summary for: fc7/weights
2017-11-27 17:13:31,091 INFO Creating Summary for: fc7/biases
2017-11-27 17:13:31,127 INFO Creating Summary for: score_fr/weights
2017-11-27 17:13:31,151 INFO Creating Summary for: score_fr/biases
WARNING:tensorflow:From e:\GitHub\KittiSeg\incl\tensorflow_fcn\fcn8_vgg.py:114: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
2017-11-27 17:13:31,465 WARNING From e:\GitHub\KittiSeg\incl\tensorflow_fcn\fcn8_vgg.py:114: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
2017-11-27 17:13:31,488 INFO Creating Summary for: upscore2/up_filter
2017-11-27 17:13:31,525 INFO Creating Summary for: score_pool4/weights
2017-11-27 17:13:31,548 INFO Creating Summary for: score_pool4/biases
2017-11-27 17:13:31,594 INFO Creating Summary for: upscore4/up_filter
2017-11-27 17:13:31,634 INFO Creating Summary for: score_pool3/weights
2017-11-27 17:13:31,659 INFO Creating Summary for: score_pool3/biases
2017-11-27 17:13:31,705 INFO Creating Summary for: upscore32/up_filter
2017-11-27 17:13:31,778 INFO Graph build successfully.
2017-11-27 17:13:31.779807: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-11-27 17:13:31,918 INFO /u/marvin/no_backup/RUNS/KittiSeg/loss_bench/xentropy_kitti_fcn_2016_10_15_01.18/model.ckpt-15999
INFO:tensorflow:Restoring parameters from RUNS\KittiSeg_pretrained\model.ckpt-15999
2017-11-27 17:13:31,919 INFO Restoring parameters from RUNS\KittiSeg_pretrained\model.ckpt-15999
2017-11-27 17:13:35,227 INFO Weights loaded successfully.
2017-11-27 17:13:35,227 INFO Starting inference using data/demo/demo.png as input
2017-11-27 17:13:40,363 INFO
2017-11-27 17:13:40,363 INFO Raw output image has been saved to: e:\GitHub\KittiSeg\data\demo\demo_raw.png
2017-11-27 17:13:40,363 INFO Red-Blue overlay of confs have been saved to: e:\GitHub\KittiSeg\data\demo\demo_rb.png
2017-11-27 17:13:40,364 INFO Green plot of predictions have been saved to: e:\GitHub\KittiSeg\data\demo\demo_green.png
2017-11-27 17:13:40,364 INFO
2017-11-27 17:13:40,364 WARNING Do NOT use this Code to evaluate multiple images.
2017-11-27 17:13:40,364 WARNING Demo.py is **very slow** and designed to be a tutorial to show how the KittiSeg works.
2017-11-27 17:13:40,365 WARNING
2017-11-27 17:13:40,365 WARNING Please see this comment, if you like to apply demo.py tomultiple images see:
2017-11-27 17:13:40,365 WARNING https://github.com/MarvinTeichmann/KittiBox/issues/15#issuecomment-301800058
```

运行结果：

demo.png

![img](http://upload-images.jianshu.io/upload_images/5952841-fcaa7a21d79ea5c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

demo_raw.png

![img](http://upload-images.jianshu.io/upload_images/5952841-76b3e77931ab2fba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

demo_rb.png

![img](http://upload-images.jianshu.io/upload_images/5952841-13bc85984e74146d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

demo_green.png

![img](http://upload-images.jianshu.io/upload_images/5952841-16e582b1fb80de21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

代码笔记：

1. 输出日志中前几行的`No environment variable`由`incl/tensorvision/utils.py`中的325-339输出，负责读取环境变量中的一些设置。
1. `KittiSeg_pretrained.zip`由`demo.py`中的125行代码下载，大小为2.56GB，默认下载到`RUNS/`文件夹中，并解压到同一目录中。
1. 接下来读取hypes和modules.
    1. hypes是指`RUNS/KittiSeg_pretrained/model_files/hypes.json`文件
    1. modules是一个dict，存放了一些模块包，包括`KittiSeg_pretrained/model_files/` 下的`data_input.py`, `architecture.py`, `objective.py`, `solver.py`, `eval.py`.
    1. 以下代码
        ```python
        f = os.path.join(model_dir, "eval.py")
        eva = imp.load_source("evaluator_%s" % postfix, f)
        ```
        表示将`eval.py`这个文件进行导入，导入后的模块名称为`evluator_%s`，`eva`为模块变量。
1. 开始构建Tensorflow Graph。首先创建2个placeholder
    ```python
    # Create placeholder for input
    image_pl = tf.placeholder(tf.float32)
    image = tf.expand_dims(image_pl, 0)
    ```
    占位符的shape为未知
    ```
    >>> image_pl
    <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=float32>
    >>> image
    <tf.Tensor 'ExpandDims:0' shape=<unknown> dtype=float32>
    ```
1. 从module中读取Graph
    ```python
    # build Tensorflow graph using the model from logdir
    prediction = core.build_inference_graph(hypes, modules, image=image)
    
    ----------incl/tensorvision/core.py
    def build_inference_graph(hypes, modules, image):
        with tf.name_scope("Validation"):

        logits = modules['arch'].inference(hypes, image, train=False)

        decoded_logits = modules['objective'].decoder(hypes, logits,
                                                      train=False)
    return decoded_logits
    
    ----------RUNS/KittiSeg_pretrained/model_files/architecture.py
    def inference(hypes, images, train=True):
        vgg16_npy_path = os.path.join(hypes['dirs']['data_dir'], "vgg16.npy")
        vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=vgg16_npy_path)

        vgg_fcn.wd = hypes['wd']

        vgg_fcn.build(images, train=train, num_classes=2, random_init_fc8=True)

        return vgg_fcn.upscore32

    ----------RUNS/KittiSeg_pretrained/model_files/objective.py
    def decoder(hypes, logits, train):
        decoded_logits = {}
        decoded_logits['logits'] = logits
        decoded_logits['softmax'] = _add_softmax(hypes, logits)
        return decoded_logits
    ```
    上文运行`download_data.py`已经下载好了`vgg16.npy`，这是VGG-16网络实现训练好的参数。日志显示的Inference结构为：
    
    | Layer Name  | Layer Shape        |
    | ----------- | ------------------ |
    | conv1_1     | (3, 3, 3, 64)      |
    | conv1_2     | (3, 3, 64, 64)     |
    | conv2_1     | (3, 3, 64, 128)    |
    | conv2_2     | (3, 3, 128, 128)   |
    | conv3_1     | (3, 3, 128, 256)   |
    | conv3_2     | (3, 3, 256, 256)   |
    | conv3_3     | (3, 3, 256, 256)   |
    | conv4_1     | (3, 3, 256, 512)   |
    | conv4_2     | (3, 3, 512, 512)   |
    | conv4_3     | (3, 3, 512, 512)   |
    | conv5_1     | (3, 3, 512, 512)   |
    | conv5_2     | (3, 3, 512, 512)   |
    | conv5_3     | (3, 3, 512, 512)   |
    | fc6         | [7, 7, 512, 4096]  |
    | fc7         | [1, 1, 4096, 4096] |
    | score_fr    |
    | upscore2    |
    | score_pool4 |
    | upscore4    |
    | score_pool3 |
    | upscore32   |
    
1. 加载本地的权重网络变量
    ```python
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    saver = tf.train.Saver()

    # Load weights from logdir
    core.load_weights(logdir, sess, saver)
    ```
    在`RUNS/KittiSeg_pretrained`下包含了ckpt文件，`load_weights`函数会自动读取目录下的`checkpoint`文件，并得到实际的参数文件，然后`save.restore`。
1. 读取且重定义测试图像，demo中**未执行**。
    ```python
    # Load and resize input image
    image = scp.misc.imread(input_image)
    if hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        image = scp.misc.imresize(image, size=(image_height, image_width),
                                    interp='cubic')
    ```
    `scp.misc.imread`返回的类型为`<class 'numpy.ndarray'>`类型，原始shape为`(375, 1242, 3)`, 而hypes中图像大小为`(384,1248)`, 利用`cubic`三次样条插值算法进行缩放。
1. Tensorflow运行预测任务。
    ```python
    # Run KittiSeg model on image
    feed = {image_pl: image}
    softmax = prediction['softmax']
    output = sess.run([softmax], feed_dict=feed)
    ```
    softmax为输出层，输出类别为2，如下
    ```
    Tensor("Validation/decoder/Softmax:0", shape=(?, 2), dtype=float32)
    ```
    output为一个list，里面只有1个元素，该元素大小为图像大小，元素为0-1的概率，表示是目标的概率。如下：
    ```
    [array([[  9.99689460e-01,   3.10521980e-04],
       [  9.99805272e-01,   1.94725304e-04],
       [  9.99785841e-01,   2.14181622e-04],
       ...,
       [  9.99480784e-01,   5.19228633e-04],
       [  9.99274552e-01,   7.25465012e-04],
       [  9.98537183e-01,   1.46284746e-03]], dtype=float32)]
    ```
1. 将输出reshape到图像大小，`output_image`的shape为`(375, 1242)`
    ```python
    # Reshape output from flat vector to 2D Image
    shape = image.shape
    output_image = output[0][:, 1].reshape(shape[0], shape[1])
    ```
1. 将每个点的概率映射到原始图像中。`rb_image`的shape为`(375, 1242, 3)`
    ```python
    # Plot confidences as red-blue overlay
    rb_image = seg.make_overlay(image, output_image)
    ```
1. 利用阈值分割图像，用绿色标注。`green_image`的shape为`(375, 1242, 3)`
    ```python
    # Accept all pixel with conf >= 0.5 as positive prediction
    # This creates a `hard` prediction result for class street
    threshold = 0.5
    street_prediction = output_image > threshold

    # Plot the hard prediction as green overlay
    green_image = tv_utils.fast_overlay(image, street_prediction)
    ```
1. 保存图像。
    ```python
    # Save output images to disk.
    if FLAGS.output_image is None:
        output_base_name = input_image
    else:
        output_base_name = FLAGS.output_image

    raw_image_name = output_base_name.split('.')[0] + '_raw.png'
    rb_image_name = output_base_name.split('.')[0] + '_rb.png'
    green_image_name = output_base_name.split('.')[0] + '_green.png'

    scp.misc.imsave(raw_image_name, output_image)
    scp.misc.imsave(rb_image_name, rb_image)
    scp.misc.imsave(green_image_name, green_image)
    ```

### Eval

```python
python evaluate.py
```

日志输出：

```
2017-12-05 14:55:04,423 INFO No environment variable 'TV_PLUGIN_DIR' found. Set to 'C:\Users\10217814/tv-plugins'.
2017-12-05 14:55:04,424 INFO No environment variable 'TV_STEP_SHOW' found. Set to '50'.
2017-12-05 14:55:04,425 INFO No environment variable 'TV_STEP_EVAL' found. Set to '250'.
2017-12-05 14:55:04,426 INFO No environment variable 'TV_STEP_WRITE' found. Set to '1000'.
2017-12-05 14:55:04,427 INFO No environment variable 'TV_MAX_KEEP' found. Set to '10'.
2017-12-05 14:55:04,428 INFO No environment variable 'TV_STEP_STR' found. Set to 'Step {step}/{total_steps}: loss = {loss_value:.2f}; lr = {lr_value:.2e}; {sec_per_batch:.3f} sec (per Batch); {examples_per_sec:.1f} imgs/sec'.
2017-12-05 14:55:04,449 INFO f: <_io.TextIOWrapper name='hypes/KittiSeg.json' mode='r' encoding='cp936'>
2017-12-05 14:55:04,451 INFO Evaluating on Validation data.
2017-12-05 14:55:04,451 INFO f: <_io.TextIOWrapper name='RUNS\\KittiSeg_pretrained\\model_files\\hypes.json' mode='r' encoding='cp936'>
npy file loaded
Layer name: conv1_1
Layer shape: (3, 3, 3, 64)
2017-12-05 14:55:05,124 INFO Creating Summary for: conv1_1/filter
2017-12-05 14:55:05,150 INFO Creating Summary for: conv1_1/biases
Layer name: conv1_2
Layer shape: (3, 3, 64, 64)
2017-12-05 14:55:05,183 INFO Creating Summary for: conv1_2/filter
2017-12-05 14:55:05,207 INFO Creating Summary for: conv1_2/biases
Layer name: conv2_1
Layer shape: (3, 3, 64, 128)
2017-12-05 14:55:05,239 INFO Creating Summary for: conv2_1/filter
2017-12-05 14:55:05,264 INFO Creating Summary for: conv2_1/biases
Layer name: conv2_2
Layer shape: (3, 3, 128, 128)
2017-12-05 14:55:05,298 INFO Creating Summary for: conv2_2/filter
2017-12-05 14:55:05,324 INFO Creating Summary for: conv2_2/biases
Layer name: conv3_1
Layer shape: (3, 3, 128, 256)
2017-12-05 14:55:05,357 INFO Creating Summary for: conv3_1/filter
2017-12-05 14:55:05,383 INFO Creating Summary for: conv3_1/biases
Layer name: conv3_2
Layer shape: (3, 3, 256, 256)
2017-12-05 14:55:05,416 INFO Creating Summary for: conv3_2/filter
2017-12-05 14:55:05,441 INFO Creating Summary for: conv3_2/biases
Layer name: conv3_3
Layer shape: (3, 3, 256, 256)
2017-12-05 14:55:05,476 INFO Creating Summary for: conv3_3/filter
2017-12-05 14:55:05,502 INFO Creating Summary for: conv3_3/biases
Layer name: conv4_1
Layer shape: (3, 3, 256, 512)
2017-12-05 14:55:05,539 INFO Creating Summary for: conv4_1/filter
2017-12-05 14:55:05,562 INFO Creating Summary for: conv4_1/biases
Layer name: conv4_2
Layer shape: (3, 3, 512, 512)
2017-12-05 14:55:05,602 INFO Creating Summary for: conv4_2/filter
2017-12-05 14:55:05,627 INFO Creating Summary for: conv4_2/biases
Layer name: conv4_3
Layer shape: (3, 3, 512, 512)
2017-12-05 14:55:05,665 INFO Creating Summary for: conv4_3/filter
2017-12-05 14:55:05,690 INFO Creating Summary for: conv4_3/biases
Layer name: conv5_1
Layer shape: (3, 3, 512, 512)
2017-12-05 14:55:05,730 INFO Creating Summary for: conv5_1/filter
2017-12-05 14:55:05,756 INFO Creating Summary for: conv5_1/biases
Layer name: conv5_2
Layer shape: (3, 3, 512, 512)
2017-12-05 14:55:05,796 INFO Creating Summary for: conv5_2/filter
2017-12-05 14:55:05,821 INFO Creating Summary for: conv5_2/biases
Layer name: conv5_3
Layer shape: (3, 3, 512, 512)
2017-12-05 14:55:05,860 INFO Creating Summary for: conv5_3/filter
2017-12-05 14:55:05,885 INFO Creating Summary for: conv5_3/biases
Layer name: fc6
Layer shape: [7, 7, 512, 4096]
2017-12-05 14:55:06,204 INFO Creating Summary for: fc6/weights
2017-12-05 14:55:06,275 INFO Creating Summary for: fc6/biases
Layer name: fc7
Layer shape: [1, 1, 4096, 4096]
2017-12-05 14:55:06,365 INFO Creating Summary for: fc7/weights
2017-12-05 14:55:06,389 INFO Creating Summary for: fc7/biases
2017-12-05 14:55:06,423 INFO Creating Summary for: score_fr/weights
2017-12-05 14:55:06,447 INFO Creating Summary for: score_fr/biases
WARNING:tensorflow:From incl\tensorflow_fcn\fcn8_vgg.py:114: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
2017-12-05 14:55:06,647 WARNING From incl\tensorflow_fcn\fcn8_vgg.py:114: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
2017-12-05 14:55:06,671 INFO Creating Summary for: upscore2/up_filter
2017-12-05 14:55:06,710 INFO Creating Summary for: score_pool4/weights
2017-12-05 14:55:06,735 INFO Creating Summary for: score_pool4/biases
2017-12-05 14:55:06,779 INFO Creating Summary for: upscore4/up_filter
2017-12-05 14:55:06,817 INFO Creating Summary for: score_pool3/weights
2017-12-05 14:55:06,841 INFO Creating Summary for: score_pool3/biases
2017-12-05 14:55:06,885 INFO Creating Summary for: upscore32/up_filter
2017-12-05 14:55:06.951540: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2017-12-05 14:55:07,076 INFO /u/marvin/no_backup/RUNS/KittiSeg/loss_bench/xentropy_kitti_fcn_2016_10_15_01.18/model.ckpt-15999
INFO:tensorflow:Restoring parameters from RUNS\KittiSeg_pretrained\model.ckpt-15999
2017-12-05 14:55:07,077 INFO Restoring parameters from RUNS\KittiSeg_pretrained\model.ckpt-15999
2017-12-05 14:55:10,334 INFO Graph loaded succesfully. Starting evaluation.
2017-12-05 14:55:10,334 INFO Output Images will be written to: RUNS\KittiSeg_pretrained\analyse\images/
2017-12-05 14:59:26,795 INFO Evaluation Succesfull. Results:
2017-12-05 14:59:26,795 INFO     MaxF1  :  96.0821
2017-12-05 14:59:26,796 INFO     BestThresh  :  14.5098
2017-12-05 14:59:26,796 INFO     Average Precision  :  92.3620
2017-12-05 14:59:26,796 INFO     Speed (msec)  :  4477.6770
2017-12-05 14:59:26,796 INFO     Speed (fps)  :  0.2233
2017-12-05 14:59:47,150 INFO Creating output on test data.
2017-12-05 14:59:47,151 INFO f: <_io.TextIOWrapper name='RUNS\\KittiSeg_pretrained\\model_files\\hypes.json' mode='r' encoding='cp936'>
npy file loaded
Layer name: conv1_1
Layer shape: (3, 3, 3, 64)
2017-12-05 14:59:47,800 INFO Creating Summary for: conv1_1/filter
2017-12-05 14:59:47,824 INFO Creating Summary for: conv1_1/biases
Layer name: conv1_2
Layer shape: (3, 3, 64, 64)
2017-12-05 14:59:47,855 INFO Creating Summary for: conv1_2/filter
2017-12-05 14:59:47,877 INFO Creating Summary for: conv1_2/biases
Layer name: conv2_1
Layer shape: (3, 3, 64, 128)
2017-12-05 14:59:47,910 INFO Creating Summary for: conv2_1/filter
2017-12-05 14:59:47,933 INFO Creating Summary for: conv2_1/biases
Layer name: conv2_2
Layer shape: (3, 3, 128, 128)
2017-12-05 14:59:47,964 INFO Creating Summary for: conv2_2/filter
2017-12-05 14:59:47,987 INFO Creating Summary for: conv2_2/biases
Layer name: conv3_1
Layer shape: (3, 3, 128, 256)
2017-12-05 14:59:48,022 INFO Creating Summary for: conv3_1/filter
2017-12-05 14:59:48,044 INFO Creating Summary for: conv3_1/biases
Layer name: conv3_2
Layer shape: (3, 3, 256, 256)
2017-12-05 14:59:48,078 INFO Creating Summary for: conv3_2/filter
2017-12-05 14:59:48,101 INFO Creating Summary for: conv3_2/biases
Layer name: conv3_3
Layer shape: (3, 3, 256, 256)
2017-12-05 14:59:48,132 INFO Creating Summary for: conv3_3/filter
2017-12-05 14:59:48,155 INFO Creating Summary for: conv3_3/biases
Layer name: conv4_1
Layer shape: (3, 3, 256, 512)
2017-12-05 14:59:48,189 INFO Creating Summary for: conv4_1/filter
2017-12-05 14:59:48,213 INFO Creating Summary for: conv4_1/biases
Layer name: conv4_2
Layer shape: (3, 3, 512, 512)
2017-12-05 14:59:48,251 INFO Creating Summary for: conv4_2/filter
2017-12-05 14:59:48,274 INFO Creating Summary for: conv4_2/biases
Layer name: conv4_3
Layer shape: (3, 3, 512, 512)
2017-12-05 14:59:48,313 INFO Creating Summary for: conv4_3/filter
2017-12-05 14:59:48,336 INFO Creating Summary for: conv4_3/biases
Layer name: conv5_1
Layer shape: (3, 3, 512, 512)
2017-12-05 14:59:48,375 INFO Creating Summary for: conv5_1/filter
2017-12-05 14:59:48,398 INFO Creating Summary for: conv5_1/biases
Layer name: conv5_2
Layer shape: (3, 3, 512, 512)
2017-12-05 14:59:48,542 INFO Creating Summary for: conv5_2/filter
2017-12-05 14:59:48,564 INFO Creating Summary for: conv5_2/biases
Layer name: conv5_3
Layer shape: (3, 3, 512, 512)
2017-12-05 14:59:48,602 INFO Creating Summary for: conv5_3/filter
2017-12-05 14:59:48,624 INFO Creating Summary for: conv5_3/biases
Layer name: fc6
Layer shape: [7, 7, 512, 4096]
2017-12-05 14:59:48,927 INFO Creating Summary for: fc6/weights
2017-12-05 14:59:48,952 INFO Creating Summary for: fc6/biases
Layer name: fc7
Layer shape: [1, 1, 4096, 4096]
2017-12-05 14:59:49,027 INFO Creating Summary for: fc7/weights
2017-12-05 14:59:49,050 INFO Creating Summary for: fc7/biases
2017-12-05 14:59:49,082 INFO Creating Summary for: score_fr/weights
2017-12-05 14:59:49,106 INFO Creating Summary for: score_fr/biases
2017-12-05 14:59:49,150 INFO Creating Summary for: upscore2/up_filter
2017-12-05 14:59:49,184 INFO Creating Summary for: score_pool4/weights
2017-12-05 14:59:49,207 INFO Creating Summary for: score_pool4/biases
2017-12-05 14:59:49,249 INFO Creating Summary for: upscore4/up_filter
2017-12-05 14:59:49,284 INFO Creating Summary for: score_pool3/weights
2017-12-05 14:59:49,307 INFO Creating Summary for: score_pool3/biases
2017-12-05 14:59:49,350 INFO Creating Summary for: upscore32/up_filter
2017-12-05 14:59:49,544 INFO /u/marvin/no_backup/RUNS/KittiSeg/loss_bench/xentropy_kitti_fcn_2016_10_15_01.18/model.ckpt-15999
INFO:tensorflow:Restoring parameters from RUNS\KittiSeg_pretrained\model.ckpt-15999
2017-12-05 14:59:49,544 INFO Restoring parameters from RUNS\KittiSeg_pretrained\model.ckpt-15999
2017-12-05 14:59:52,804 INFO Images will be written to test_images//test_images_{green, rg}
2017-12-05 14:59:57,461 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000000.png
2017-12-05 15:00:02,348 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000001.png
2017-12-05 15:00:07,223 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000002.png
2017-12-05 15:00:12,061 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000003.png
2017-12-05 15:00:16,949 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000004.png
2017-12-05 15:00:21,853 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000005.png
2017-12-05 15:00:26,717 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000006.png
2017-12-05 15:00:31,517 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000007.png
2017-12-05 15:00:36,363 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000008.png
2017-12-05 15:00:41,237 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000009.png
2017-12-05 15:00:46,344 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000010.png
2017-12-05 15:00:51,325 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000011.png
2017-12-05 15:00:56,322 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000012.png
2017-12-05 15:01:01,255 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000013.png
2017-12-05 15:01:06,227 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000014.png
2017-12-05 15:01:11,209 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000015.png
2017-12-05 15:01:16,227 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000016.png
2017-12-05 15:01:21,133 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000017.png
2017-12-05 15:01:26,015 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000018.png
2017-12-05 15:01:30,831 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000019.png
2017-12-05 15:01:35,669 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000020.png
2017-12-05 15:01:40,528 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000021.png
2017-12-05 15:01:45,387 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000022.png
2017-12-05 15:01:50,216 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000023.png
2017-12-05 15:01:55,223 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000024.png
2017-12-05 15:02:00,105 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000025.png
2017-12-05 15:02:05,010 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000026.png
2017-12-05 15:02:09,813 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000027.png
2017-12-05 15:02:14,699 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000028.png
2017-12-05 15:02:19,589 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000029.png
2017-12-05 15:02:24,495 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000030.png
2017-12-05 15:02:29,399 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000031.png
2017-12-05 15:02:34,284 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000032.png
2017-12-05 15:02:39,152 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000033.png
2017-12-05 15:02:43,985 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000034.png
2017-12-05 15:02:48,823 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000035.png
2017-12-05 15:02:53,796 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000036.png
2017-12-05 15:02:58,849 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000037.png
2017-12-05 15:03:03,744 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000038.png
2017-12-05 15:03:08,647 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000039.png
2017-12-05 15:03:13,522 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000040.png
2017-12-05 15:03:18,384 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000041.png
2017-12-05 15:03:23,182 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000042.png
2017-12-05 15:03:28,069 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000043.png
2017-12-05 15:03:32,960 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000044.png
2017-12-05 15:03:37,831 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000045.png
2017-12-05 15:03:42,714 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000046.png
2017-12-05 15:03:47,612 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000047.png
2017-12-05 15:03:52,527 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000048.png
2017-12-05 15:03:57,440 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000049.png
2017-12-05 15:04:02,381 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000050.png
2017-12-05 15:04:07,287 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000051.png
2017-12-05 15:04:12,161 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000052.png
2017-12-05 15:04:17,087 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000053.png
2017-12-05 15:04:21,988 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000054.png
2017-12-05 15:04:26,960 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000055.png
2017-12-05 15:04:31,889 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000056.png
2017-12-05 15:04:36,739 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000057.png
2017-12-05 15:04:41,628 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000058.png
2017-12-05 15:04:46,555 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000059.png
2017-12-05 15:04:51,468 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000060.png
2017-12-05 15:04:56,434 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000061.png
2017-12-05 15:05:01,358 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000062.png
2017-12-05 15:05:06,235 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000063.png
2017-12-05 15:05:11,182 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000064.png
2017-12-05 15:05:16,128 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000065.png
2017-12-05 15:05:21,090 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000066.png
2017-12-05 15:05:26,048 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000067.png
2017-12-05 15:05:31,033 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000068.png
2017-12-05 15:05:35,943 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000069.png
2017-12-05 15:05:40,808 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000070.png
2017-12-05 15:05:45,679 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000071.png
2017-12-05 15:05:50,457 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000072.png
2017-12-05 15:05:55,265 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000073.png
2017-12-05 15:06:00,046 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000074.png
2017-12-05 15:06:04,821 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000075.png
2017-12-05 15:06:09,609 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000076.png
2017-12-05 15:06:14,362 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000077.png
2017-12-05 15:06:19,160 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000078.png
2017-12-05 15:06:23,969 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000079.png
2017-12-05 15:06:28,786 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000080.png
2017-12-05 15:06:33,557 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000081.png
2017-12-05 15:06:38,215 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000082.png
2017-12-05 15:06:42,989 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000083.png
2017-12-05 15:06:47,764 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000084.png
2017-12-05 15:06:52,520 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000085.png
2017-12-05 15:06:57,252 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000086.png
2017-12-05 15:07:02,030 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000087.png
2017-12-05 15:07:06,759 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000088.png
2017-12-05 15:07:11,461 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000089.png
2017-12-05 15:07:16,125 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000090.png
2017-12-05 15:07:20,847 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000091.png
2017-12-05 15:07:25,597 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000092.png
2017-12-05 15:07:30,367 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000093.png
2017-12-05 15:07:35,041 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000094.png
2017-12-05 15:07:39,878 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/um_road_000095.png
2017-12-05 15:07:44,772 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000000.png
2017-12-05 15:07:49,713 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000001.png
2017-12-05 15:07:54,652 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000002.png
2017-12-05 15:07:59,567 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000003.png
2017-12-05 15:08:04,506 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000004.png
2017-12-05 15:08:09,394 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000005.png
2017-12-05 15:08:14,264 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000006.png
2017-12-05 15:08:19,172 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000007.png
2017-12-05 15:08:24,038 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000008.png
2017-12-05 15:08:28,876 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000009.png
2017-12-05 15:08:33,741 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000010.png
2017-12-05 15:08:38,566 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000011.png
2017-12-05 15:08:43,371 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000012.png
2017-12-05 15:08:48,278 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000013.png
2017-12-05 15:08:53,173 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000014.png
2017-12-05 15:08:58,058 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000015.png
2017-12-05 15:09:02,945 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000016.png
2017-12-05 15:09:07,823 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000017.png
2017-12-05 15:09:12,706 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000018.png
2017-12-05 15:09:17,606 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000019.png
2017-12-05 15:09:22,511 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000020.png
2017-12-05 15:09:27,425 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000021.png
2017-12-05 15:09:32,353 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000022.png
2017-12-05 15:09:37,271 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000023.png
2017-12-05 15:09:42,127 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000024.png
2017-12-05 15:09:47,000 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000025.png
2017-12-05 15:09:51,880 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000026.png
2017-12-05 15:09:56,769 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000027.png
2017-12-05 15:10:01,659 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000028.png
2017-12-05 15:10:06,557 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000029.png
2017-12-05 15:10:11,478 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000030.png
2017-12-05 15:10:16,426 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000031.png
2017-12-05 15:10:21,363 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000032.png
2017-12-05 15:10:26,311 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000033.png
2017-12-05 15:10:31,238 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000034.png
2017-12-05 15:10:36,185 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000035.png
2017-12-05 15:10:41,134 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000036.png
2017-12-05 15:10:46,070 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000037.png
2017-12-05 15:10:51,042 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000038.png
2017-12-05 15:10:55,956 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000039.png
2017-12-05 15:11:00,887 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000040.png
2017-12-05 15:11:05,879 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000041.png
2017-12-05 15:11:10,948 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000042.png
2017-12-05 15:11:16,079 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000043.png
2017-12-05 15:11:21,366 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000044.png
2017-12-05 15:11:26,465 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000045.png
2017-12-05 15:11:31,596 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000046.png
2017-12-05 15:11:36,729 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000047.png
2017-12-05 15:11:41,803 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000048.png
2017-12-05 15:11:46,803 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000049.png
2017-12-05 15:11:51,918 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000050.png
2017-12-05 15:11:57,060 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000051.png
2017-12-05 15:12:02,089 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000052.png
2017-12-05 15:12:07,133 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000053.png
2017-12-05 15:12:12,138 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000054.png
2017-12-05 15:12:17,101 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000055.png
2017-12-05 15:12:22,004 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000056.png
2017-12-05 15:12:27,061 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000057.png
2017-12-05 15:12:32,084 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000058.png
2017-12-05 15:12:37,078 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000059.png
2017-12-05 15:12:42,055 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000060.png
2017-12-05 15:12:47,004 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000061.png
2017-12-05 15:12:51,926 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000062.png
2017-12-05 15:12:56,927 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000063.png
2017-12-05 15:13:01,903 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000064.png
2017-12-05 15:13:06,917 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000065.png
2017-12-05 15:13:11,883 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000066.png
2017-12-05 15:13:16,850 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000067.png
2017-12-05 15:13:21,859 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000068.png
2017-12-05 15:13:26,780 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000069.png
2017-12-05 15:13:31,801 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000070.png
2017-12-05 15:13:36,831 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000071.png
2017-12-05 15:13:41,825 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000072.png
2017-12-05 15:13:46,840 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000073.png
2017-12-05 15:13:51,825 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000074.png
2017-12-05 15:13:56,806 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000075.png
2017-12-05 15:14:01,863 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000076.png
2017-12-05 15:14:06,955 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000077.png
2017-12-05 15:14:12,017 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000078.png
2017-12-05 15:14:17,187 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000079.png
2017-12-05 15:14:22,356 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000080.png
2017-12-05 15:14:27,529 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000081.png
2017-12-05 15:14:32,744 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000082.png
2017-12-05 15:14:37,933 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000083.png
2017-12-05 15:14:43,074 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000084.png
2017-12-05 15:14:48,216 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000085.png
2017-12-05 15:14:53,365 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000086.png
2017-12-05 15:14:58,503 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000087.png
2017-12-05 15:15:03,648 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000088.png
2017-12-05 15:15:08,813 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000089.png
2017-12-05 15:15:13,960 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000090.png
2017-12-05 15:15:18,972 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000091.png
2017-12-05 15:15:24,061 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000092.png
2017-12-05 15:15:29,157 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/umm_road_000093.png
2017-12-05 15:15:34,265 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000000.png
2017-12-05 15:15:39,228 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000001.png
2017-12-05 15:15:44,272 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000002.png
2017-12-05 15:15:49,329 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000003.png
2017-12-05 15:15:54,296 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000004.png
2017-12-05 15:15:59,296 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000005.png
2017-12-05 15:16:04,332 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000006.png
2017-12-05 15:16:09,360 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000007.png
2017-12-05 15:16:14,318 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000008.png
2017-12-05 15:16:19,305 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000009.png
2017-12-05 15:16:24,339 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000010.png
2017-12-05 15:16:29,424 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000011.png
2017-12-05 15:16:34,434 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000012.png
2017-12-05 15:16:39,510 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000013.png
2017-12-05 15:16:44,500 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000014.png
2017-12-05 15:16:49,497 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000015.png
2017-12-05 15:16:54,575 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000016.png
2017-12-05 15:16:59,563 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000017.png
2017-12-05 15:17:04,646 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000018.png
2017-12-05 15:17:09,872 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000019.png
2017-12-05 15:17:14,897 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000020.png
2017-12-05 15:17:19,902 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000021.png
2017-12-05 15:17:24,923 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000022.png
2017-12-05 15:17:29,986 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000023.png
2017-12-05 15:17:35,160 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000024.png
2017-12-05 15:17:40,244 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000025.png
2017-12-05 15:17:45,399 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000026.png
2017-12-05 15:17:50,633 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000027.png
2017-12-05 15:17:55,675 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000028.png
2017-12-05 15:18:00,674 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000029.png
2017-12-05 15:18:05,722 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000030.png
2017-12-05 15:18:10,824 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000031.png
2017-12-05 15:18:15,823 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000032.png
2017-12-05 15:18:20,810 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000033.png
2017-12-05 15:18:25,872 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000034.png
2017-12-05 15:18:30,990 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000035.png
2017-12-05 15:18:36,192 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000036.png
2017-12-05 15:18:41,195 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000037.png
2017-12-05 15:18:46,205 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000038.png
2017-12-05 15:18:51,157 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000039.png
2017-12-05 15:18:56,218 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000040.png
2017-12-05 15:19:01,204 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000041.png
2017-12-05 15:19:06,138 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000042.png
2017-12-05 15:19:11,079 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000043.png
2017-12-05 15:19:16,025 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000044.png
2017-12-05 15:19:21,059 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000045.png
2017-12-05 15:19:26,124 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000046.png
2017-12-05 15:19:31,152 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000047.png
2017-12-05 15:19:36,268 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000048.png
2017-12-05 15:19:41,319 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000049.png
2017-12-05 15:19:46,395 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000050.png
2017-12-05 15:19:51,521 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000051.png
2017-12-05 15:19:56,704 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000052.png
2017-12-05 15:20:01,742 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000053.png
2017-12-05 15:20:06,779 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000054.png
2017-12-05 15:20:11,909 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000055.png
2017-12-05 15:20:16,943 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000056.png
2017-12-05 15:20:21,960 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000057.png
2017-12-05 15:20:26,942 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000058.png
2017-12-05 15:20:31,969 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000059.png
2017-12-05 15:20:37,024 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000060.png
2017-12-05 15:20:42,045 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000061.png
2017-12-05 15:20:47,163 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000062.png
2017-12-05 15:20:52,581 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000063.png
2017-12-05 15:20:57,870 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000064.png
2017-12-05 15:21:03,084 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000065.png
2017-12-05 15:21:08,138 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000066.png
2017-12-05 15:21:13,257 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000067.png
2017-12-05 15:21:18,286 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000068.png
2017-12-05 15:21:23,297 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000069.png
2017-12-05 15:21:28,458 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000070.png
2017-12-05 15:21:33,523 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000071.png
2017-12-05 15:21:38,423 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000072.png
2017-12-05 15:21:43,471 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000073.png
2017-12-05 15:21:48,509 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000074.png
2017-12-05 15:21:53,635 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000075.png
2017-12-05 15:21:58,703 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000076.png
2017-12-05 15:22:03,674 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000077.png
2017-12-05 15:22:08,672 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000078.png
2017-12-05 15:22:13,776 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000079.png
2017-12-05 15:22:18,827 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000080.png
2017-12-05 15:22:23,891 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000081.png
2017-12-05 15:22:28,967 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000082.png
2017-12-05 15:22:34,022 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000083.png
2017-12-05 15:22:39,076 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000084.png
2017-12-05 15:22:44,178 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000085.png
2017-12-05 15:22:49,210 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000086.png
2017-12-05 15:22:54,305 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000087.png
2017-12-05 15:22:59,279 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000088.png
2017-12-05 15:23:04,186 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000089.png
2017-12-05 15:23:09,103 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000090.png
2017-12-05 15:23:14,044 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000091.png
2017-12-05 15:23:19,067 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000092.png
2017-12-05 15:23:24,212 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000093.png
2017-12-05 15:23:29,356 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000094.png
2017-12-05 15:23:34,412 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000095.png
2017-12-05 15:23:39,554 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000096.png
2017-12-05 15:23:44,733 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000097.png
2017-12-05 15:23:49,775 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000098.png
2017-12-05 15:23:54,883 INFO Writing file: E:\GitHub\KittiSeg\RUNS\KittiSeg_pretrained\test_images/uu_road_000099.png
2017-12-05 15:23:55,479 INFO Analysis for pretrained model complete.
2017-12-05 15:23:55,479 INFO For evaluating your own models I recommend using:`tv-analyze --logdir /path/to/run`.
2017-12-05 15:23:55,480 INFO tv-analysis has a much cleaner interface.
```
