---
title: MultiNet：自主驾驶中的实时联合语义推理
date: 2017-11-23 23:00:00
tags: 机器视觉
categories: 深度学习
---

MultiNet论文相关
<!-- more -->
论文下载地址:[原文地址](https://arxiv.org/pdf/1612.07695.pdf)、[免翻墙地址](http://outz1n6zr.bkt.clouddn.com/1612.07695.pdf)

论文Github地址：[KittiSeg](https://github.com/MarvinTeichmann/KittiSeg)

论文翻译参考：[csdn](http://blog.csdn.net/hanging_gardens/article/details/72724258)

# MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving

MultiNet: 自主驾驶中的实时联合语义推理

## Abstract

While most approaches to semantic reasoning have focused on improving performance, in this paper we argue that computational times are very important in order to enable real time applications such as autonomous driving. Towards this goal, we present an approach to joint classifi-cation, detection and semantic segmentation via a unified architecture where the encoder is shared amongst the three tasks. Our approach is very simple, can be trained end-toend and performs extremely well in the challenging KITTI dataset, outperforming the state-of-the-art in the road segmentation task. Our approach is also very efficient, taking less than 100 ms to perform all tasks.

虽然大多数语义推理方法都集中于提高性能, 但在本文中, 我们认为运行时间对于实现自动驾驶等实时应用非常重要。为了实现这一目标, 我们提出了一种通过统一架构的联合classification、detection、segmentation的方法, 其中encoder由三个任务共享。我们的方法非常简单, 可以进行端到端训练, 并在KITTI数据集表上现非常出色, 超越了道路划分任务中的最先进技术。我们的方法也非常有效, 执行所有任务只需要不到100 ms。

## Introduction

Current advances in the field of computer vision have made clear that visual perception is going to play a key role in the development of self-driving cars. This is mostly due to the deep learning revolution which begun with the introduction of AlexNet in 2012 [23]. Since then, the accuracy of new approaches has been increasing at a vertiginous rate. Causes of this are the existence of more data, increased computation power and algorithmic developments. The current trend is to create deeper networks with as many layers as possible [17].

目前计算机视觉领域的前沿已经明确指出, 视觉感知将在自驾车发展中发挥关键作用。这主要是由于2012年AlexNet开始引起了深度学习革命[23]。从那时起, 新方法的准确性一直在快速提升。其原因是有了更多的数据、增大了计算量以及算法的开发。目前的趋势是尽可能多地创建更深层次的网络[17]。

While performance is extremely high, when dealing with real-world applications, running times become important. New hardware accelerators as well as compression, reduced precision and distillation methods have been exploited to speed up current networks. 

虽然性能非常好, 但在处理实际应用程序时, 运行时间也变得重要。新的硬件的加速与压缩、精度的降低与提练方法已经被利用来加速当前的网络。

In this paper we take an alternative approach and design a network architecture that can very efficiently perform classification, detection and semantic segmentation simultaneously.This is done by incorporating all three task Figure 1: Our goal: Solving street classification, vehicle detection and road segmentation in one forward pass. into a unified encoder-decoder architecture. We name our approach MultiNet. The encoder consists of the convolution and pooling layers from the VGG network [45] and is shared among all tasks. Those features are then utilized by task-specific decoders, which produce their outputs in real-time. In particular, the detection decoder combines the fast regression design introduced in Yolo [38] with the sizeadjusting ROI-Pooling of Fast-RCNN [14], achieving a better speed-accuracy ratio.

在本文中, 我们采用一种替代方法, 设计一种可以非常有效地进行classification、detection、segmentation的网络体系结构。通过将这三个任务并入统一的编码器—解码器架构来完成。以MultiNet命名我们的方法。编码器由VGG网络的卷积层和池化层组成[45], 并被所有任务共享。那些特征随后被特定任务的解码器使用, 这些解码器实时产生它们的输出。特别地, 检测解码器将Yolo中介绍的快速回归设计[38]与fast RCNN的尺寸调整ROI-Pooling[14]相结合, 实现了更好的速度—精度比。

We demonstrate the effectiveness of our approach in the challenging KITTI benchmark [13] and show state-of-theart performance in road segmentation. Importantly, our ROI-Pooling implementation can significantly improve detection performance without requiring an explicit proposal generation network. This gives our decoder a significant speed advantage compared to Faster-RCNN. Our approach is able to benefit from sharing computations, allowing us to perform inference in less than 100 ms for all tasks.

我们在KITTI数据库上测试该方法的有效性[13], 并得到道路分割中最好的性能表现。重要的是, ROI-Pooling的实施可以显著提高检测性能, 而不需要一个明确的候选生成网络。与Faster RCNN相比, 这使得我们的解码器具有显著的速度优势。我们的方法得益于共享运算, 从而执行所有任务所需时间不到100 ms。

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185805.png)

Figure 1: Our goal: Solving street classification, vehicle detection and road segmentation in one forward pass.

图1：我们的目标：在一次前向传播中解决街道分类, 车辆检测和道路分割

## Related Work 相关工作

In this section we review current approaches to the tasks that MultiNet tackles, i.e., detection, classification and semantic segmentation. We focus our attention on deep learning based approaches.

本节中, 我们回顾了目前使用MultiNet处理任务的方法, 即classification、detection、segmentation。我们着重于基于深度学习的方法。

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185846.png)

Figure 2: MultiNet architecture.

图2：MultiNet架构

> 译者注：  
> CNN Encoder : `input:1248x384x3` --(`CONV 64`)x**2**--> `1248x384x64` --(`MAX_POOL`)--> `624x192x64` --(`CONV 128`)x**2**--> `624x192x128` --(`MAX_POOL`)--> `312x96x128` --(`CONV 256`)x**3**--> `312x96x256` --(`MAX_POOL`)--> `156x48x256` --(`CONV 512`)x**3**--> `156x48x512` --(`MAX_POOL`)--> `78x24x512` --(`CONV 512`)x**3**--> `78x24x512`  
> Encoded Features: --(`MAX_POOL`)--> `39x24x512`  

### 分类

**Classification**: After the development of AlexNet [23], most modern approaches to image classification utilize deep learning. Residual networks [17] constitute the state-of-theart, as they allow to train very deep networks without problems of vanishing or exploding gradients. In the context of road classification, deep neural networks are also widely employed [31]. Sensor fusion has also been exploited in this context [43]. In this paper we use classification to guide other semantic tasks, i.e., segmentation and detection.

**分类**：在AlexNet出现后, 大多数现代图像分类方法开始利用深度学习。残差网络[17]是目前最先进的技术, 因为它们能训练非常深的网络, 而不会产生梯度消失或爆炸。在道路分类的背景下, 深度神经网络也得到广泛的应用[31]。传感器融合也在这种情况下被利用[43]。在本文中, 我们使用分类来指导其他语义任务, 即分割和检测。 

### 检测

**Detection**: Traditional deep learning approaches to object detection follow a two step process, where region proposals [25, 20, 19] are first generated and then scored using a convolutional network [15, 40]. Additional performance improvements can be gained by using convolutional neural networks (CNNs) for the proposal generation step [8, 40] or by reasoning in 3D [5, 4]. Recently, several methods have proposed to use a single deep network that is trainable end-to-end to directly perform detection [44, 38, 39, 27]. Their main advantage over proposal-based methods is that they are much faster at both training and inference time, and thus more suitable for real-time detection applications. However, so far they lag far behind in performance. In this paper we propose an end-to-end trainable detector which reduces significantly the performance gap. We argue that the main advantage of proposal-based methods is their ability to have size-adjustable features. This inspired our zoom layer that as shown in our experience results in large improvements in performance. 

**检测**：传统的深度学习目标检测方法包含两步, 首先生成候选区域[25, 20, 19], 然后使用卷积网络进行评分[15, 40]。通过CNN来对候选区域进行生成[8, 40]或通过三维推理生成可以获得更好的性能提高[5, 4]。近来, 直接利用可端对端训练的单深度网络进行检测的几种方法已经被提出[44, 38, 39, 27]。与基于候选区域的方法相比的主要优点在于, 它们在训练和推理时间上都快得多, 因此更适合实时检测的应用。然而, 到目前为止, 它们在性能上远远落后。在本文中, 我们提出了一种能够显著降低性能差距的可端对端训练的检测器。我们认为, 基于候选区域的方法的主要优点在于它们具有可调整大小的功能。这促使我们采用了rezoom层, 如实验所示, 它可以大大提高性能。

### 分割

**Segmentation**: Inspired by the successes of deep learning, CNN-based classifiers were adapted to the task of semantic segmentation. Early approaches used the inherent efficiency of CNNs to implement implicit sliding-window [16, 26]. Fully Convolutional Networks (FCNs) were proposed to model semantic segmentation using a deep learning pipeline that is trainable end-to-end. Transposed convolutions [50, 6, 21] are utilized to upsample low resolution features. A variety of deeper flavors of FCNs have been proposed since [1, 34, 41, 36]. Very good results are archived by combining FCNs with conditional random fields (CRFs) [52, 2, 3]. [52, 42] showed that mean-field inference in the CRF can be cast as a recurrent net allowing end-to-end training. Dilated convolutions were introduced in [48] to augment the receptive field size without losing resolution. The aforementioned techniques in conjunction with residual networks [17] are currently the state-of-the-art.

**分割**：受深度学习的成功的启发, 基于CNN的分类器开始应用于语义分割任务。早期方法利用CNN的固有效率来实现隐式的滑动窗口[16, 26]。全卷积网络(FCN)被用于采用可端到端训练的深度学习通道建模语义分割任务。转置卷积[50, 6, 21]用于上采样低分辨率特征。自从[1, 34, 41, 36]以来, 已经提出了各种各样的FCN。通过将FCN与条件随机场(CRF)组合得到了非常好的结果[52, 2, 3]。[52, 42]表明, CRF中平均场推理可以被投射为一个可端到端的训练经常性的网络(recurrent net)。扩展卷积(dilated convolutions)在[48]中被引入, 以增大接收域大小而不失去分辨率。上述与残差网络相结合的技术[17]是目前最先进的技术。

### 联合推理

**Joint Reasoning**: Multi-task learning techniques aim at learning better representations by exploiting many tasks. Several approaches have been proposed in the context of CNNs [30, 28] but applications have mainly been focussed on face recognition tasks [51, 47, 37]. [18] reasons jointly about classification and segmentation using an SVM in combination with dynamic programming. [46] proposed to use a CRF to solve many tasks including detection, segmentation and scene classification. In the context of deep learning, [7] proposed a model which is able to jointly perform pose estimation and object classification. To our knowledge no unified deep architecture has been proposed to solve segmentation, classification and detection.

**联合推理**：多任务学习技术旨在通过开发多任务来学习更好的表征。几种基于CNN的方法已经被提出[30, 28], 但是主要应用在面部识别任务上[51, 47, 37] 。[18]是关于使用SVM与动态规划相结合的分类和分割方法。[46]则提出使用CRF来解决多任务, 包括segmentation、detection、scene classification。在深度学习的背景下, [7]提出了联合姿态估计和目标分类的模型。据我们所知, 目前还没有提出联合的深度架构来解决classification、detection、segmentation任务。

## MultiNet for Joint Semantic Reasoning

In this paper we propose an efficient and effective feedforward architecture, which we call MultiNet, to jointly reason about semantic segmentation, image classification and object detection. Our approach shares a common encoder over the three tasks and has three branches, each implementing a decoder for a given task. We refer the reader to Fig. 2 for an illustration of our architecture. MultiNet can be trained end-to-end and joint inference over all tasks can be done in less than 100ms. We start our discussion by introducing our joint encoder, follow by the task-specific decoders.

在本文中, 我们提出了一种高效的前馈架构, 我们称之为MultiNet, 以联合的理解语义分割, 图像分类和目标检测。我们的方法在三个任务中共享一个共同的编码器, 并且具有三个分支, 每个分支是一个实现给定任务的解码器。参考图2为网络架构。MultiNet可以进行端到端的训练, 所有任务的联合推理可以在100ms内完成。我们开始讨论并根据特定任务的解码器来介绍联合编码器。

The task of the encoder is to process the image and extract rich abstract features [49] that contain all necessary information to perform accurate segmentation, detection and image classification. The encoder of MultiNet consists of the first 13 layers of the VGG16 network [45], which are applied in a fully convolutional manner to the image producing a tensor of size 39 × 12 × 512. This is the output of the 5th pooling layer, which is called pool5 in the VGG implementation [45].

编码器的任务是处理图像并提取丰富的抽象特征[49], 该特征包含了执行准确分割, 检测和图像分类的所必要的信息。MultiNet编码器由VGG16网络的前13层组成[45], 应用全卷积方式产生39×12×512大小的张量。这是第5个pooling层的输出, 在VGG中叫作pool5 [45]。

### Classification Decoder 分类解码器

The classification decoder is designed to take advantage of the encoder. Towards this goal, we apply a 1 × 1 convolution followed by a fully connected layer and a softmax layer to output the final class probabilities.

分类解码器被设计用来利用编码器的优点。为了实现这一目标, 我们应用1×1卷积, 然后用全连接层和softmax层输出最后类的概率。

### Detection Decoder 检测解码器

FastBox, our detection decoder, is designed to be a regression based detection system. We choose such a decoder over a proposal based one because it can be train end-toend, and both training and inference can be done very efficiently. Our approach is inspired by ReInspect [39], Yolo [38] and Overfeat [44]. In addition to the standard regression pipeline, we include an ROI pooling approach, which allows the network to utilize features at a higher resolution, similar to the much slower Faster-RCNN.

我们的检测解码器FastBox, 被设计为基于回归的检测系统。我们选择一种基于候选区域的解码器, 因为它可以进行端对端的训练, 并且可以非常有效地完成训练和推理。方法灵感来自ReInspect [39], Yolo [38]和Overfeat [44]。除了标准回归流程之外, 我们还包含一个ROI池化方法, 它允许网络利用更高分辨率的特征, 类似较慢的Faster-RCNN。

The first step of our decoder is to produce a rough estimate of the bounding boxes. Towards this goal, we first pass the encoded features through a 1 × 1 convolutional layer with 500 filters, producing a tensor of shape 39 × 12 × 500, which we call hidden. This tensor is processed with another 1 × 1 convolutional layer which outputs 6 channels at resolution 39 × 12. We call this tensor prediction, the values of the tensor have a semantic meaning. The first two channels of this tensor form a coarse segmentation of the image. Their values represent the confidence that an object of interest is present at that particular location in the 39 × 12 grid. The last four channels represent the coordinates of a bounding box in the area around that cell. Fig. 3 shows an image with its cells.

该解码器的第一步是产生bounding box的粗略估计。为了实现这一目标, 首先用500个滤波器的1×1卷积层传递编码的特征, 产生一个39×12×500大小的张量, 我们称之为隐藏层。随后该张量用另一个1×1卷积层处理, 输出6个分辨率为39×12的通道。我们称这个张量为prediction, 张量的值具有语义含义。该张量的前两个通道形成图像的粗分割。这些值表示感兴趣目标存在于39×12网格中的特定位置处的置信度。最后四个通道表示该单元周围区域中边界框的坐标。图3表示有cell的图像。

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185905.png)

Figure 3: Visualization of our label encoding. Blue grid: cells, Red cells: cells containing a car, Grey cells: cells in don’t care area. Green boxes: ground truth boxes.

图3：可视化我们的标签编码。蓝色网格：单元(cells)。红单元：含有汽车的单元。灰色单元：无关区域的单元。绿色框：真实值

Such prediction, however, is not very accurate. In this paper we argue that this is due to the fact that resolution has been lost by the time we arrive to the encoder output. To alleviate this problem we introduce a rezoom layer, which predicts a residual on the locations of the bounding boxes by exploiting high resolution features. This is done by concatenating subsets of higher resolution VGG features (156×48) with the hidden features (39 × 12) and applying 1 × 1 convolutions on top of this. In order to make this possible, a 39 × 12 grid needs to be generated out of the high resolution VGG features. This is achieved by applying ROI pooling [40] using the rough prediction provided by the tensor prediction. Finally, this is concatenated with the 39×12×6 features and passed through a 1×1 convolution layer to produce the residuals.

然而, 这种预测不是非常准确。在本文中, 我们认为这是由于编码器输出时的分辨率已经丢失。为了减轻这个问题, 我们引入了一个rezoom层, 它通过利用高分辨率特征来预测边界框位置上的残差。它通过将更高分辨率的VGG特征的子集(156×48)与隐藏特征(39×12)连接并在其上应用1×1卷积来完成。为了使其成为可能, 需要从高分辨率VGG特征产生39×12网格, 这些网格是通过应用ROI池化[40]使用由tensor prediction提供的粗预测来实现的。最后, 它与39×12×6特征连接, 并通过1×1卷积层以产生残差。

### Segmentation Decoder 分割解码器

The segmentation decoder follows the FCN architecture [29]. Given the encoder, we transform the remaining fullyconnected (FC) layers of the VGG architecture into 1 × 1 convolutional layers to produce a low resolution segmentation of size 39 × 12. This is followed by three transposed convolution layers [6, 21] to perform up-sampling. Skip layers are utilized to extract high resolution features from the lower layers. Those features are first processed by a 1 × 1 convolution layer and then added to the partially upsampled results.

分割解码器遵循FCN架构[29]。给定编码器, 我们将VGG架构中已有的全连接(FC)层转换为1×1的卷积层, 以产生39×12大小的低分辨率segmentation。其后是三个transposed卷积层[6, 21] 进行上采样。skip层用于从较低层提取高分辨率特征。这些特征首先由1×1卷积层处理, 然后加到部分上采样结果中。

## Training Details 训练详情

In this section we describe the loss functions we employ as well as other details of our training procedure including initialization.

在本节中, 将介绍损失函数以及训练过程中的其他细节, 包括初始化。

### Label encoding 标签编码

We use one-hot encoding for classification and segmentation. For the detection, we assigned a positive confidence if and only if it intersects with at least one bounding box. We parameterize the bounding box by the x and y coordinate of its center and the width w and height h of the box. Note that this encoding is much simpler than Faster RCNN or ReInspect.

采用单热编码(one-hot encoding)进行分类和分割。对于检测, 当且仅当它与至少一个边界框相交时, 我们才分配了一个正的的置信度。我们用其中心x、y坐标以及框的宽度w、高度h来参数化边界框。请注意, 该编码比Faster RCNN或ReInspect简单得多。

### Loss Functions

We define our loss function as the sum of the loss functions for classification, segmentation and detection. We employ cross-entropy as loss function for the classification and segmentation branches, which is defined as

将损失函数定义为分类, 分割和检测的损失函数的总和。采用交叉熵作为分类和分割分支的损失函数, 定义如下： 

$$
loss_{class}(p,q):=-\frac{1}{\vert I\vert}\sum_{i\in I}\sum_{c\in C}{q_i(c)\log p_i(c)} \qquad (1)
$$

where p is the prediction, q the ground truth and C the set of classes. We use the sum of two losses for detection: Cross entropy loss for the confidences and an L1 loss on the bounding box coordinates. Note that the L1 loss is only computed for cells which have been assigned a positive confidence label. Thus

其中p是prediction, q是ground truth, C是类的集合。我们使用两个损失的和作为detection的loss：置信度(confidence)的交叉熵损失及边界框坐标的L1损失。请注意, 只有被赋予正置信标签的cells才计算它们的L1损失。从而: 

$$
loss_{box}(p,q):=\frac{1}{I}\sum_{i \in I}\delta_{q_i}\cdot (\vert x_{p_i}-x_{q_i}\vert+\vert y_{p_i}-y_{q_i}\vert+\vert w_{p_i}-w_{q_i}\vert+\vert h_{p_i}-h_{q_i}\vert) \qquad (2)
$$

where p is the prediction, q the ground truth, C the set of classes and I is the set of examples in the mini batch.

其中p是prediction, q是ground truth, C是类的集合, I是小批次中的一组示例。

### Combined Training Strategy

Joint training is performed by merging the gradients computed by each loss on independent mini batches. This allows us to train each of the three decoders with their own set of training parameters. During gradient merging all losses are weighted equally. In addition, we observe that the detection network requires more steps to be trained than the other tasks. We thus sample our mini batches such that we alternate an update using all loss functions with two updates that only utilize the detection loss.

通过将每个损失计算的梯度合并在独立的小批量上进行联合训练。这样我们就能用自己的训练参数来训练每个解码器。在梯度合并过程中, 所有的损失都被相等地加权(all losses are weighted equally)。另外, 我们观察到, 检测网络需要比其他任务训练更多次。因此, 我们对我们的小批次进行抽样, 以便我们使用仅利用detection loss的两次更新的全损失函数来交替更新。

### Initialization

The encoder is initialized using pretrained VGG weights on ImageNet. The detection and classification decoder weights are randomly initialized using a uniform distribution in the range (−0.1,0.1). The convolutional layers of the segmentation decoder are also initialized using VGG weights and the transposed convolution layers are initialized to perform bilinear upsampling. The skip connections on the other hand are initialized randomly with very small weights (i.e. std of 1e − 4). This allows us to perform training in one step (as opposed to the two step procedure of [29]).

使用ImageNet上预先训练的VGG权重对编码器进行初始化。使用范围在(-0.1, 0.1)的单位分布随机初始化检测和分类解码器权重。分割解码器的卷积层也使用VGG权重进行初始化, 并且转置卷积层被初始化以执行双线性上采样。另一方面, skip connection以非常小的权重(即1e-4的标准)随机初始化。我们能一步进行训练(与[29]的两步程序相反)。

### Optimizer and regularization

We use the Adam optimizer [22] with a learning rate of 1e − 5 to train our MultiNet. A weight decay of 5e − 4 is applied to all layers and dropout with probability 0.5 is applied to all (inner) 1 × 1 convolutions in the decoder.

使用Adam优化器[22], 学习率为1e - 5来训练MultiNet。对所有层施加5e-4的权重衰减, 并且对解码器中所有(内部)1×1的卷积进行概率为0.5的dropout。

## Experimental Results

In this section we perform our experimental evaluation on the challenging KITTI dataset.

在KITTI数据集上进行实验评估。

| Experiment     | max steps | eval steps [k] |
| -------------- | --------- | -------------- |
| Segmentation   | 16,000    | 100            |
| Classification | 18,000    | 200            |
| Detection      | 180,000   | 1000           |
| United         | 200,000   | 1000           |

Table 1: Summary of training length.

表1：训练长度总结

### Dataset

We evaluate MultiNet in he KITTI Vision Benchmark Suite [12]. The Benchmark contains images showing a variety of street situations captured from a moving platform driving around the city of Karlruhe. In addition to the raw data, KITTI comes with a number of labels for different tasks relevant to autonomous driving. We use the road benchmark of [10] to evaluate the performance of our semantic segmentation decoder and the object detection benchmark [13] for the detection decoder. We exploit the automatically generated labels of [31], which provide us with road labels generated by combining GPS information with open-street map data.

在KITTI Vision Benchmark Suite [12]上评估MultiNet。基准测试包含许多图像, 这些图像展示了在卡尔鲁厄市由驾驶移动平台捕获的各种街道情况。除原始数据之外, KITTI还附带了许多与自主驾驶相关的不同任务的标签。使用[10]的道路基准来评估语义分割解码器的性能，使用[13]的目标检测基准来检测解码器。利用[31]自动生成的标签, 这些标签提供了通过将GPS信息与开放街道地图数据相结合生成的道路标签。

Detection performance is measured using the average precision score [9]. For evaluation, objects are divided into three categories: easy, moderate and hard to detect. The segmentation performance is measured using the MaxF1 score [10]. In addition, the average precision score is given for reference. Classification performance is evaluated by computing accuracy and precision-recall plots.

使用平均精度得分测量检测性能[9]。对于评估, 目标分为三类：容易, 中等和困难的检测。使用MaxF1分数测量分割性能[10]。此外, 平均精度得分作为参考。分类性能通过计算精度和精确召回图进行评估。

### Performance evaluation

Our evaluation is performed in two steps. First we build three individual models consisting of the VGG-encoder and the decoder corresponding to the task. Those models are tuned to achieve highest possible performance on the given task. In a second step MultiNet is trained using one encoder and three decoders in a single network. We evaluate both settings in our experimental evaluation. We report a set of plots depicting the convergence properties of our networks in Figs. 4, 6 and 8. Evaluation on the validation set is performed every k iterations during training, where k for each tasks is given in Table 1. To reduce the variance in the plots the output is smoothed by computing the median over the last 50 evaluations performed.

我们的评估分两步进行。首先, 我们构建由VGG编码器和对应三个任务的单独的解码器组成的模型。这些模型被调整到在给定任务上实现最高的性能。在第二步中, MultiNet在一个网络中使用一个编码器和三个解码器进行训练。我们在实验评估中评估这两种设置。我们展示了一组描绘该网络收敛性质的图。在训练期间每k次迭代执行验证集的评估, 其中每个任务的k在表1中给出。为了减少图中的方差, 通过计算过去执行的50次评估的中值来平滑输出。

#### Segmentation

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185933.png)

Figure 4: Convergence behavior of the segmentation decoder

图4：分割解码器的收敛性质

Our Segmentation decoder is trained using the KITTI Road Benchmark [10]. This dataset is very small, providing only 289 training images. Thus the network has to transfer as much knowledge as possible from pre-training. Note that the skip connections are the only layers which are randomly initialized and thus need to be trained from scratch. This transfer learning approach leads to very fast convergence of the network. As shown in Fig. 4 the raw scores already reach values of about 95 % after only about 4000 iterations. Training is conducted for 16,000 iterations to obtain a meaningful median score.

分割解码器使用KITTI道路基准进行训练[10]。该数据集非常小, 仅提供289个训练图像。因此, 网络必须从预训练转移尽可能多的知识。请注意, skip connection是唯一随机初始化的层, 因此需要从头进行训练。这种传输学习方法导致网络能非常快速的收敛。如图4所示, 只有4000次迭代,时 原始分数已达到近95％。训练16, 000次以获得有重要意义的中值。

| Metric            | Result  |
| ----------------- | ------- |
| MaxF1             | 95.83 % |
| Average Precision | 92.29 % |
| Speed (msec)      | 94.6 ms |
| Speed (fps)       | 10.6 Hz |

Table 2: Validation performance of the segmentation decoder

表2：分割编码器的验证性能 

| Method            | MaxF1      | AP         | Place   |
| ----------------- | ---------- | ---------- | ------- |
| FCN LC [32]       | 90.79 %    | 85.83 %    | 5th     |
| FTP [24]          | 91.61 %    | 90.96 %    | 4th     |
| DDN [33]          | 93.43 %    | 89.67 %    | 3th     |
| Up Conv Poly [35] | 93.83 %    | 90.47 %    | 2rd     |
| MultiNet          | **94.88%** | **93.71%** | **1st** |

Table 3: Summary of the URBAN ROAD scores on the public KITTIRoad Detection Leaderboard [11].

表3：KITTI Road检测排行榜上城市道路评分[11] 

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190007.png)

Figure 5: Visualization of the segmentation output. Top rows: Soft segmentation output as red blue plot. The intensity of the
plot reflects the confidence. Bottom rows hard class labels.

图5：分割输出的可视化。顶行：软分割输出为红蓝色。图的强度反映了置信度。底行：硬类标签

Table 2 shows the scores of our segmentation decoder after 16,000 iterations. The scores indicate that our segmentation decoder generalizes very well using only the data given by the KITTI Road Benchmark. No other segmentation dataset was utilized. As shown in Fig. 5, our approach is very effective at segmenting roads. Even difficult areas, corresponding to sidewalks and buildings are segmented correctly. In the confidence plots shown in top two rows of Fig. 5, it can be seen that our approach has confidence close to 0.5 at the edges of the street. This is due to the slight variation in the labels of the training set. We have submitted the results of our approach on the test set to the KITTI road leaderboard. As shown in Table 3, our result achieve first place.

表2显示了16, 000次迭代后分割解码器的得分。分数表明, 分割解码器仅使用KITTI Road Benchmark给的数据泛化得很好。没有使用其他分割数据集。如图5所示, 该方法在分割道路方面非常有效。对应于人行道和建筑物的困难区域也能正确分割。在图5上面两行所示的置信区间中, 可以看出, 我们的方法在街道边缘的置信度接近0.5。这是由于训练集标签的轻微变化。我们已经将该方法的测试结果提交给了KITTI道路排行榜。如表3所示, 我们的结果达到了第一名。 

#### Detection

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185947.png)

Figure 6: Validation scores of the detection decoder. Performance of FastBox with and without rezoom layer is shown for comparison.

图6：检测解码器的验证分数。显示具有和不具有再缩放层的FastBox的性能进行比较

Our detection decoder is trained and evaluated on the data provided by the KITTI object benchmark [13]. Fig. 6 shows the convergence rate of the validation scores. The detection decoder converges much slower than the segmentation and classification decoders. We therefore train the decoder up to iteration 180,000.

检测解码器对KITTI目标基准测试提供的数据进行了训练和评估[13]。图6显示了验证分数的收敛速度。检测解码器的收敛速度比分割和分类解码器慢得多。因此, 我们训练180, 000次。

| Task: Metric        | moderate | easy    | hard    |
| ------------------- | -------- | ------- | ------- |
| FastBox with rezoom | 83.35 %  | 92.80 % | 67.59 % |
| FastBox no rezoom   | 77.00 %  | 86.45 % | 60.82 % |

Table 4: Detection performance of FastBox.

表4：FastBox的检测性能 

|                 | FastBox  | FastBox (no rezoom) |
| --------------- | -------- | ------------------- |
| speed [msec]    | 37.49 ms | 35.75 ms            |
| speed [fps]     | 26.67 Hz | 27.96 Hz            |
| post-processing | 2.10 ms  | 2.46 ms             |

Table 5: Detection speed of FastBox. Results are measured on a Pascal Titan X.

表5：FastBox的检测速度。结果在Pascal Titan X上测量 

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190024.png)

Figure 7: Visualization of the detection output. With and without non-maximal suppression applied.

图7：检测输出的可视化。有无非最大抑制应用

FastBox can perform evaluation at very high speed: an inference step takes 37.49 ms per image. This makes FastBox particularly suitable for real-time applications. Our results indicate further that the computational overhead of the rezoom layer is negligible (see Table 5). The performance boost of the rezoom layer on the other hand is quite substantial (see Table 4), justifying the use of a rezoom layer in the final model. Qualitative results are shown in Fig. 7 with and without non-maxima suppression.

FastBox可以以非常高的速度执行评估：每个图像的推理步骤需要37.49ms。这使得FastBox特别适合实时应用。我们的结果进一步表明, rezoom layer的计算开销是可以忽略的(见表5)。另一方面, rezoom layer的性能提升是相当大的(参见表4), 在最终模型中使用rezoom layer证明了这两点。定性结果如图7具有和不具有非极大抑制。 

#### MultiNet

| Task: Metric             | seperate | 2 losses | 3 losses |
| ------------------------ | -------- | -------- | -------- |
| Segmentation: MaxF1      | 95.83%   | 94.98%   | 95.13%   |
| Detection: Moderate      | 83.35%   | 83.91%   | 84.39%   |
| Classification: Accuracy | 92.65%   | -        | 94.38%   |

Table 6: MultiNet performance: Comparison between united and seperate evaluation on the validation set.

表6：MultiNet性能：验证集合的联合和单独评估之间的比较 

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190044.png)

Figure 8: MultiNet: Comparison of Joint and Separate Training.

图8.MultiNet：联合训练和分离训练的比较

We have experimented with two versions of MultiNet. The first version is trained using two decoders, (detection and segmentation) while the second version is trained with all three decoders. Training with additional decoders significantly lowers the convergence speed of all decoders. When training with all three decoders it takes segmentation more than 30.000 and detection more than 150.000 iterations to converge, as shown in Fig. 8. Fig. 8 and Table 6 also show, that our combined training does not harm performance. On the contrary, the detection and classification tasks benefit slightly when jointly trained. This effect can be explained by transfer learning between tasks: relevant features learned from one task can be utilized in a different task.

我们已经尝试了两个版本的MultiNet。使用两个解码器(检测和分割)训练第一个版本, 而第二个版本使用所有三个解码器进行训练。使用额外的解码器进行训练明显地降低了所有解码器的收敛速度。当使用所有三个解码器进行训练时, 分割需要超过30.000次迭代，检测需要超过150.000次迭代达到收敛, 如图8所示。图8和表6还表明, 该组合训练不会影响性能。相反, 联合训练时, 检测和分类任务略有好转。这种效果可以通过任务之间的转移学习来解释：从一个任务中学到的相关特征可以用于不同的任务。

| MultiNet | Segmentation | Detection | Classification |
| -------- | ------------ | --------- | -------------- |
| 98.10 ms | 94.6 ms      | 37.5 ms   | 35.94 ms       |
| 10.2 Hz  | 10.6 Hz      | 27.7 Hz   | 27.8 Hz        |

Table 7: MultiNet inference speed: Comparision between united and seperate evaluation.

表7：MultiNet推理速度：联合和单独评估之间的比较 

MultiNet is particularly suited for real-time applications. As shown in Table 7 computational complexity benefits significantly from a shared architecture. Overall, MultiNet is able to solve all three task together in real-time.

MultiNet特别适用于实时应用。如表7, 计算复杂度显著得益于共享架构。总的来说, MultiNet能够实时的同时解决这三个任务。

Figure 9: Visualization of the MultiNet output.

图9. MultiNet输出可视化

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190113.png)

## Conclusion

In this paper we have developed a unified deep architecture which is able to jointly reason about classification, detection and semantic segmentation. Our approach is very simple, can be trained end-to-end and performs extremely well in the challenging KITTI, outperforming the state-ofthe-art in the road segmentation task. Our approach is also very efficient, taking 98.10 ms to perform all tasks. In the future we plan to exploit compression methods in order to further reduce the computational bottleneck and energy consumption of MutiNet.

在本文中, 我们开发了一个联合的深度架构, 能够共同推理分类, 检测和语义分割。我们的方法非常简单, 可以端到端训练, 并在KITTI中表现非常出色, 超越了道路分割任务中的最先进的技术。我们的方法也非常有效, 需要98.10ms执行所有任务。未来我们计划开发压缩方法, 以进一步降低MutiNet的计算瓶颈和能耗。

**Acknowledgements**: This work was partially supported by Begabtenstiftung Informatik Karlsruhe, ONR-N00014-14-1-0232, Qualcomm, Samsung, NVIDIA, Google, EPSRC and NSERC. We are thankful to Thomas Roddick for proofreading the paper.

## References

[1] V. Badrinarayanan, A. Kendall, and R. Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. CoRR, abs/1511.00561, 2015. 2  
[2] L. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. CoRR, abs/1412.7062, 2014. 2  
[3] L. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. CoRR, abs/1606.00915, 2016. 2  
[4] X. Chen, K. Kundu, Z. Zhang, H. Ma, S. Fidler, and R. Urtasun. Monocular 3d object detection for autonomous driving. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2147–2156, 2016. 2  
[5] X. Chen, K. Kundu, Y. Zhu, A. G. Berneshawi, H. Ma, S. Fidler, and R. Urtasun. 3d object proposals for accurate object class detection. In Advances in Neural Information Processing Systems, pages 424–432, 2015. 2 
[6] V. Dumoulin and F. Visin. A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285, 2016. 2, 3  
[7] M. Elhoseiny, T. El-Gaaly, A. Bakry, and A. M. Elgammal. Convolutional models for joint object categorization and pose estimation. CoRR, abs/1511.05175, 2015. 3  
[8] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov. Scalable object detection using deep neural networks. CoRR, abs/1312.2249, 2013. 2  
[9] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results. 
http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html. 4  
[10] J. Fritsch, T. Kuehnl, and A. Geiger. A new performance measure and evaluation benchmark for road detection algorithms. In International Conference on Intelligent Transportation Systems (ITSC), 2013. 4, 5 
[11] A. Geiger. Kitti road public benchmark, 2013. 5  
[12] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 2013. 4  
[13] A. Geiger, P. Lenz, and R. Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012. 1, 4, 5  
[14] R. B. Girshick. Fast R-CNN. CoRR, abs/1504.08083, 2015. 1  
[15] R. B. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. CoRR, abs/1311.2524, 2013. 2  
[16] A. Giusti, D. C. Ciresan, J. Masci, L. M. Gambardella, and J. Schmidhuber. Fast image scanning with deep max-pooling convolutional neural networks. CoRR, abs/1302.1700, 2013. 2  
[17] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015. 1, 2  
[18] M. Hoai, Z.-Z. Lan, and F. De la Torre. Joint segmentation and classification of human actions in video. In Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on, pages 3265–3272. IEEE, 2011. 3  
[19] J. H. Hosang, R. Benenson, P. Doll´ar, and B. Schiele. What makes for effective detection proposals? CoRR, abs/1502.05082, 2015. 2  
[20] J. H. Hosang, R. Benenson, and B. Schiele. How good are detection proposals, really? CoRR, abs/1406.6962, 2014. 2  
[21] D. J. Im, C. D. Kim, H. Jiang, and R. Memisevic. Generating images with recurrent adversarial networks. CoRR, abs/1602.05110, 2016. 2, 3  
[22] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. 4  
[23] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012. 1, 2  
[24] A. Laddha, M. K. Kocamaz, L. E. Navarro-Serment, and M. Hebert. Map-supervised road detection. In 2016 IEEE Intelligent Vehicles Symposium (IV), pages 118–123, June 2016. 5  
[25] C. H. Lampert, M. B. Blaschko, and T. Hofmann. Beyond sliding windows: Object localization by efficient subwindow search. In Computer Vision and Pattern Recognition, 2008. CVPR 2008. IEEE Conference on, pages 1–8. IEEE, 2008. 2  
[26] H. Li, R. Zhao, and X. Wang. Highly efficient forward and backward propagation of convolutional neural networks for pixelwise classification. CoRR, abs/1412.4526, 2014. 2  
[27] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. E. Reed. SSD: single shot multibox detector. CoRR, abs/1512.02325, 2015. 2  
[28] X. Liu, J. Gao, X. He, L. Deng, K. Duh, and Y.-Y. Wang. Representation learning using multi-task deep neural networks for semantic classification and information retrieval. In Proc. NAACL, 2015. 3  
[29] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. CVPR (to appear), Nov. 2015. 3, 4  
[30] M. Long and J. Wang. Learning multiple tasks with deep relationship networks. CoRR, abs/1506.02117, 2015. 3  
[31] W.-C. Ma, S. Wang, M. A. Brubaker, S. Fidler, and R. Urtasun. Find your way by observing the sun and other semantic cues. arXiv preprint arXiv:1606.07415, 2016. 2, 4  
[32] C. C. T. Mendes, V. Frmont, and D. F. Wolf. Exploiting fully convolutional neural networks for fast road detection. In IEEE Conference on Robotics and Automation (ICRA), May 2016. 5  
[33] R. Mohan. Deep deconvolutional networks for scene parsing, 2014. 5  
[34] H. Noh, S. Hong, and B. Han. Learning deconvolution network for semantic segmentation. 2015. 2  
[35] G. Oliveira,W. Burgard, and T. Brox. Efficient deep methods for monocular road segmentation. 2016. 5  
[36] G. Papandreou, L. Chen, K. Murphy, and A. L. Yuille. Weakly- and semi-supervised learning of a DCNN for semantic image segmentation. CoRR, abs/1502.02734, 2015. 2  
[37] R. Ranjan, V. M. Patel, and R. Chellappa. Hyperface: A deep multi-task learning framework for face detection, landmark localization, pose estimation, and gender recognition. CoRR, abs/1603.01249, 2016. 3  
[38] J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. CoRR, abs/1506.02640, 2015. 1, 2, 3  
[39] M. Ren and R. S. Zemel. End-to-end instance segmentation and counting with recurrent attention. CoRR, abs/1605.09410, 2016. 2, 3  
[40] S. Ren, K. He, R. B. Girshick, and J. Sun. Faster R-CNN: towards real-time object detection with region proposal networks. CoRR, abs/1506.01497, 2015. 2, 3  
[41] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. CoRR, abs/1505.04597, 2015. 2  
[42] A. G. Schwing and R. Urtasun. Fully connected deep structured networks. CoRR, abs/1503.02351, 2015. 2  
[43] C. Seeger, A. M¨uller, L. Schwarz, and M. Manz. Towards road type classification with occupancy grids. IVSWorkshop, 2016. 2  
[44] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. CoRR, abs/1312.6229, 2013. 2, 3  
[45] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014. 1, 3  
[46] J. Yao, S. Fidler, and R. Urtasun. Describing the scene as a whole: Joint object detection, scene classification and semantic segmentation. In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on, pages 702–709. IEEE, 2012. 3  
[47] J. Yim, H. Jung, B. Yoo, C. Choi, D. Park, and J. Kim. Rotating your face using multi-task deep neural network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 676–684, 2015. 3  
[48] F. Yu and V. Koltun. Multi-scale context aggregation by dilated convolutions. CoRR, abs/1511.07122, 2015. 2  
[49] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In European Conference on Computer Vision, pages 818–833. Springer, 2014. 3  
[50] M. D. Zeiler, D. Krishnan, G. W. Taylor, and R. Fergus. Deconvolutional networks. In Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on, pages 2528–2535. IEEE, 2010. 2  
[51] Z. Zhang, P. Luo, C. C. Loy, and X. Tang. Facial landmark detection by deep multi-task learning. In European Conference on Computer Vision, pages 94–108. Springer, 2014. 3  
[52] S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang, and P. H. S. Torr. Conditional random fields as recurrent neural networks. CoRR, abs/1502.03240, 2015. 2 

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

![](http://outz1n6zr.bkt.clouddn.com/demo.png)

demo_raw.png

![](http://outz1n6zr.bkt.clouddn.com/demo_raw.png)

demo_rb.png

![](http://outz1n6zr.bkt.clouddn.com/demo_rb.png)

demo_green.png

![](http://outz1n6zr.bkt.clouddn.com/demo_green.png)

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