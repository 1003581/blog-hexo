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

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185846.png)

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

$$
loss_{class}(p,q):=-\frac{1}{\vert I\vert}\sum_{i\in I}\sum_{c\in C}{q_i(c)\log p_i(c)} \qquad (1)
$$

where p is the prediction, q the ground truth and C the set of classes. We use the sum of two losses for detection: Cross entropy loss for the confidences and an L1 loss on the bounding box coordinates. Note that the L1 loss is only computed for cells which have been assigned a positive confidence label. Thus

$$
loss_{box}(p,q):=\frac{1}{I}\sum_{i \in I}\delta_{q_i}\cdot (\vert x_{p_i}-x_{q_i}\vert+\vert y_{p_i}-y_{q_i}\vert+\vert w_{p_i}-w_{q_i}\vert+\vert h_{p_i}-h_{q_i}\vert) \qquad (2)
$$

where p is the prediction, q the ground truth, C the set of classes and I is the set of examples in the mini batch.

### Combined Training Strategy

Joint training is performed by merging the gradients computed by each loss on independent mini batches. This allows us to train each of the three decoders with their own set of training parameters. During gradient merging all losses are weighted equally. In addition, we observe that the detection network requires more steps to be trained than the other tasks. We thus sample our mini batches such that we alternate an update using all loss functions with two updates that only utilize the detection loss.

### Initialization

The encoder is initialized using pretrained VGG weights on ImageNet. The detection and classification decoder weights are randomly initialized using a uniform distribution in the range (−0.1,0.1). The convolutional layers of the segmentation decoder are also initialized using VGG weights and the transposed convolution layers are initialized to perform bilinear upsampling. The skip connections on the other hand are initialized randomly with very small weights (i.e. std of 1e − 4). This allows us to perform training in one step (as opposed to the two step procedure of [29]).

| Experiment     | max steps | eval steps [k] |
| -------------- | --------- | -------------- |
| Segmentation   | 16,000    | 100            |
| Classification | 18,000    | 200            |
| Detection      | 180,000   | 1000           |
| United         | 200,000   | 1000           |

Table 1: Summary of training length.

### Optimizer and regularization

We use the Adam optimizer [22] with a learning rate of 1e − 5 to train our MultiNet. A weight decay of 5e − 4 is applied to all layers and dropout with probability 0.5 is applied to all (inner) 1 × 1 convolutions in the decoder.

## Experimental Results

In this section we perform our experimental evaluation on the challenging KITTI dataset.

### Dataset

We evaluate MultiNet in he KITTI Vision Benchmark Suite [12]. The Benchmark contains images showing a variety of street situations captured from a moving platform driving around the city of Karlruhe. In addition to the raw data, KITTI comes with a number of labels for different tasks relevant to autonomous driving. We use the road benchmark of [10] to evaluate the performance of our semantic segmentation decoder and the object detection benchmark [13] for the detection decoder. We exploit the automatically generated labels of [31], which provide us with road labels generated by combining GPS information with open-street map data.

Detection performance is measured using the average precision score [9]. For evaluation, objects are divided into three categories: easy, moderate and hard to detect. The segmentation performance is measured using the MaxF1 score [10]. In addition, the average precision score is given for reference. Classification performance is evaluated by computing accuracy and precision-recall plots.

### Performance evaluation

Our evaluation is performed in two steps. First we build three individual models consisting of the VGG-encoder and the decoder corresponding to the task. Those models are tuned to achieve highest possible performance on the given task. In a second step MultiNet is trained using one encoder and three decoders in a single network. We evaluate both settings in our experimental evaluation. We report a set of plots depicting the convergence properties of our networks in Figs. 4, 6 and 8. Evaluation on the validation set is performed every k iterations during training, where k for each tasks is given in Table 1. To reduce the variance in the plots the output is smoothed by computing the median over the last 50 evaluations performed.

#### Segmentation

Our Segmentation decoder is trained using the KITTI Road Benchmark [10]. This dataset is very small, providing only 289 training images. Thus the network has to transfer as much knowledge as possible from pre-training. Note that the skip connections are the only layers which are randomly initialized and thus need to be trained from scratch. This transfer learning approach leads to very fast convergence of the network. As shown in Fig. 4 the raw scores already reach values of about 95 % after only about 4000 iterations. Training is conducted for 16,000 iterations to obtain a meaningful median score.

Table 2 shows the scores of our segmentation decoder after 16,000 iterations. The scores indicate that our segmentation decoder generalizes very well using only the data given by the KITTI Road Benchmark. No other segmentation dataset was utilized. As shown in Fig. 5, our approach is very effective at segmenting roads. Even difficult areas, corresponding to sidewalks and buildings are segmented correctly. In the confidence plots shown in top two rows of Fig. 5, it can be seen that our approach has confidence close to 0.5 at the edges of the street. This is due to the slight variation in the labels of the training set. We have submitted the results of our approach on the test set to the KITTI road leaderboard. As shown in Table 3, our result achieve first place.

#### Detection

Our detection decoder is trained and evaluated on the data provided by the KITTI object benchmark [13]. Fig. 6 shows the convergence rate of the validation scores. The detection decoder converges much slower than the segmentation and classification decoders. We therefore train the decoder up to iteration 180,000.

FastBox can perform evaluation at very high speed: an inference step takes 37.49 ms per image. This makes FastBox particularly suitable for real-time applications. Our results indicate further that the computational overhead of the rezoom layer is negligible (see Table 5). The performance boost of the rezoom layer on the other hand is quite substantial (see Table 4), justifying the use of a rezoom layer in the final model. Qualitative results are shown in Fig. 7 with and without non-maxima suppression.

#### MultiNet

We have experimented with two versions of
MultiNet. The first version is trained using two decoders,
(detection and segmentation) while the second version is
trained with all three decoders. Training with additional
decoders significantly lowers the convergence speed of all
decoders. When training with all three decoders it takes segmentation more than 30.000 and detection more than 150.000 iterations to converge, as shown in Fig. 8. Fig. 8
and Table 6 also show, that our combined training does not
harm performance. On the contrary, the detection and classification
tasks benefit slightly when jointly trained. This
effect can be explained by transfer learning between tasks:
relevant features learned from one task can be utilized in a
different task.

MultiNet is particularly suited for real-time applications.
As shown in Table 7 computational complexity benefits significantly
from a shared architecture. Overall, MultiNet is
able to solve all three task together in real-time.

## Conclusion

In this paper we have developed a unified deep architecture
which is able to jointly reason about classification,
detection and semantic segmentation. Our approach is very
simple, can be trained end-to-end and performs extremely
well in the challenging KITTI, outperforming the state-ofthe-art
in the road segmentation task. Our approach is also
very efficient, taking 98.10 ms to perform all tasks. In the
future we plan to exploit compression methods in order
to further reduce the computational bottleneck and energy
consumption of MutiNet.

**Acknowledgements**: This work was partially supported
by Begabtenstiftung Informatik Karlsruhe, ONR-N00014-
14-1-0232, Qualcomm, Samsung, NVIDIA, Google, EPSRC
and NSERC. We are thankful to Thomas Roddick for
proofreading the paper.

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

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185947.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190007.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190024.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190044.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_190113.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-23_185947.png)
