---
title: MultiNet：自主驾驶中的实时联合语义推理
date: 2017-11-23 23:00:00
tags: cv
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

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185947.png)

Figure 2: MultiNet architecture.

图2：MultiNet架构

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

The classification decoder is designed to take advantage
of the encoder. Towards this goal, we apply a 1 × 1 convolution
followed by a fully connected layer and a softmax
layer to output the final class probabilities.

### Detection Decoder 检测解码器

FastBox, our detection decoder, is designed to be a regression
based detection system. We choose such a decoder
over a proposal based one because it can be train end-toend,
and both training and inference can be done very effi-
ciently. Our approach is inspired by ReInspect [39], Yolo
[38] and Overfeat [44]. In addition to the standard regression
pipeline, we include an ROI pooling approach, which
allows the network to utilize features at a higher resolution,
similar to the much slower Faster-RCNN.

The first step of our decoder is to produce a rough estimate
of the bounding boxes. Towards this goal, we first pass
the encoded features through a 1 × 1 convolutional layer
with 500 filters, producing a tensor of shape 39 × 12 × 500,
which we call hidden. This tensor is processed with another
1 × 1 convolutional layer which outputs 6 channels at resolution
39 × 12. We call this tensor prediction, the values
of the tensor have a semantic meaning. The first two channels
of this tensor form a coarse segmentation of the image.
Their values represent the confidence that an object of interest
is present at that particular location in the 39 × 12 grid.
The last four channels represent the coordinates of a bounding
box in the area around that cell. Fig. 3 shows an image
with its cells.

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185905.png)

Figure 3: Visualization of our label encoding. Blue grid:
cells, Red cells: cells containing a car, Grey cells: cells in
don’t care area. Green boxes: ground truth boxes.

Such prediction, however, is not very accurate. In this
paper we argue that this is due to the fact that resolution has
been lost by the time we arrive to the encoder output. To
alleviate this problem we introduce a rezoom layer, which
predicts a residual on the locations of the bounding boxes by
exploiting high resolution features. This is done by concatenating
subsets of higher resolution VGG features (156×48)
with the hidden features (39 × 12) and applying 1 × 1 convolutions
on top of this. In order to make this possible, a
39 × 12 grid needs to be generated out of the high resolution
VGG features. This is achieved by applying ROI pooling
[40] using the rough prediction provided by the tensor
prediction. Finally, this is concatenated with the 39×12×6
features and passed through a 1×1 convolution layer to produce
the residuals.


### Segmentation Decoder 分割解码器

The segmentation decoder follows the FCN architecture
[29]. Given the encoder, we transform the remaining fullyconnected
(FC) layers of the VGG architecture into 1 × 1
convolutional layers to produce a low resolution segmentation
of size 39 × 12. This is followed by three transposed
convolution layers [6, 21] to perform up-sampling. Skip
layers are utilized to extract high resolution features from
the lower layers. Those features are first processed by a
1 × 1 convolution layer and then added to the partially upsampled
results.

## Training Details 训练详情

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185947.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190007.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190024.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190044.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190113.png)


![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185846.png)