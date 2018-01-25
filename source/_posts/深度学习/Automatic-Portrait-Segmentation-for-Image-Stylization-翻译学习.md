---
title: Automatic Portrait Segmentation for Image Stylization 翻译学习
date: 2018-01-24 16:00:18
tags: 深度学习
categories: 深度学习
---

图像分割
<!-- more -->
# 论文资料

- [论文主页](http://xiaoyongshen.me/webpage_portrait/index.html)

# 论文翻译

Automatic Portrait Segmentation for Image Stylization

自动肖像分割的图像风格化

## 摘要 Abstract

![](http://upload-images.jianshu.io/upload_images/5952841-3a2c88230ebf2af7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 1: Our highly accurate automatic portrait segmentation method allows many portrait processing tools to be fully automatic. (a) is the input image and (b) is our automatic segmentation result. (c-e) show different automatic image stylization applications based on the segmentation result. The image is from the Flickr user “Olaf Trubel”.

图1：我们高度精确的自动人像分割方法允许许多人像处理工具完全自动化。 （a）是输入图像，（b）是我们的自动分割结果。 （c-e）根据分割结果显示不同的自动图像风格化应用程序。 该图像来自Flickr用户“Olaf Trubel”。

Portraiture is a major art form in both photography and painting. In most instances, artists seek to make the subject standout from its surrounding, for instance, by making it brighter or sharper. In the digital world, similar effects can be achieved by processing a portrait image with photographic or painterly filters that adapt to the semantics of the image. While many successful user-guided methods exist to delineate the subject, fully automatic techniques are lacking and yield unsatisfactory results. Our paper first addresses this problem by introducing a new automatic segmentation algorithm dedicated to portraits. We then build upon this result and describe several portrait filters that exploit our automatic segmentation algorithm to generate high-quality portraits.

肖像画是摄影和绘画的主要艺术形式。 在大多数情况下，艺术家都试图让这个主题脱颖而出，例如，使它更明亮或更清晰。 在数字世界中，通过使用适合图像语义的摄影或绘画过滤器处理肖像图像，可以实现类似的效果。 虽然存在许多成功的用户引导方法来描述该主题，但缺乏全自动技术并且产生不令人满意的结果。 本文首先通过引入一种专门用于人像的自动分割算法来解决这个问题。 然后我们建立在这个结果的基础上，并描述了几种利用我们的自动分割算法生成高质量肖像的肖像滤镜。

## Introduction

With the rapid adoption of camera smartphones, the self portrait image has become conspicuously abundant in digital photography. A study by Samsung UK estimated that about 30% of smart phonephotos taken were self portraits [Hal], and more recently, HTC’simaging specialist Symon Whitehorn reported that in some markets,self portraits make up 90% of smartphone photos [Mic].

随着相机智能手机的迅速普及，自拍肖像在数码摄影中的应用也日益丰富。 三星英国的一项研究估计，约30％的智能手机拍照是自画像[哈尔]，最近，HTC的成像专家Symon Whitehorn报告说，在某些市场上，自拍肖像占智能手机照片的90％[Mic]。

The bulk of these portraits are captured by casual photographers who often lack the necessary skills to consistently take great portraits, or to successfully post-process them. Even with the plethora of easy-to-use automatic image filters that are amenable to novice photographers, good portrait post-processing requires treating the subject separately from the background in order to make the subject stand out. There are many good user-guided tools for creating masks for selectively treating portrait subjects, but these tools can still be tedious and difficult to use, and remain an obstacle for casual photographers who want their portraits to look good. While many image filtering operations can be used when selectively processing portrait photos, a few that are particularly applicableto portraits include background replacement, portrait style transfer [SPB∗14], color and tone enhancement [HSGL11], and local feature editing [LCDL08]. While these can all be used to great effect with little to no user interaction, they remain inaccessible to casual photographers due to their reliance on a good selection.

这些肖像中的大部分都是被临时摄影师拍摄的，他们往往缺乏持续拍摄精美肖像的必要技能，或者成功对其进行后期处理。即使有许多易于使用的自动图像过滤器，适合新手摄影师，良好的人像后期处理需要将被摄体与背景分开处理，以使主体脱颖而出。有许多好的用户指导工具可以创建用于选择性处理肖像主题的蒙版，但是这些工具仍然是乏味且难以使用的，并且对于希望他们的肖像看起来不错的休闲摄影师来说仍然是障碍。虽然在选择性处理人像照片时可以使用许多图像滤镜操作，但是一些特别适用于人像的包括背景替换，肖像样式转换[SPB * 14]，色彩和色调增强[HSGL11]以及局部特征编辑[LCDL08]。虽然这些都可以用来很好的效果，但很少或没有用户的互动，他们仍然是由于依赖于一个很好的选择，随便摄影师无法访问。

A fully automatic portrait segmentation method is required to make these techniques accessible to the masses. Unfortunately, designing such an automatic portrait segmentation system is nontrivial. Even with access to robust facial feature detectors and smart selection techniques such as graph cuts, complicated backgroundsand backgrounds whose color statistics are similar to those of the subject readily lead to poor results. 

需要全自动的人像分割方法来使这些技术能够被大众接受。 不幸的是，设计这样的自动纵向分割系统是非常重要的。 即使能够使用强健的面部特征检测器和智能选择技术（如图形裁剪，复杂的背景和背景，其颜色统计类似于主题）也容易导致较差的结果。

In this paper, we propose a fully automatic portrait segmentation technique that takes a portrait image and produces a score map of equal resolution. This score map indicates the probability that a given pixel belongs to the subject, and can be used directly as a soft mask, or thresholded to a binary mask or trimap for use with image matting techniques. To accomplish this, we take advantage of recent advances in deep convolutional neural networks (CNNs) which have set new performance standards for sematic segmentation tasks such as Pascal VOC [EGW∗10] and Microsoft COCO [LMB∗14]. We augment one such network with portrait-specific knowledge toachieve extremely high accuracy that is more than sufficient for most automatic portrait post-processing techniques, unlocking arange of portrait editing operations previously unavailable to the novice photographer, while simultaneously reducing the amount of work required to generate these selections for intermediate and advanced users.

在本文中，我们提出了一种全自动的人像分割技术，它可以拍摄人像，并生成等分的分数图。 该分数图指示给定像素属于对象的概率，并且可以直接用作软掩膜，或者阈值化为二值掩膜或三维贴图用于图像抠像技术。 为此，我们利用最近在深度卷积神经网络（CNNs）方面的进展，为卷积分割任务设立了新的性能标准，如Pascal VOC [EGW * 10]和Microsoft COCO [LMB * 14]。 我们增加一个具有肖像特定知识的网络，以达到极高的准确度，这对于大多数自动肖像后处理技术来说已经足够了，解开了以前对于新手摄影师不可用的一些肖像编辑操作，同时减少了需要为中级和高级用户生成这些选择。

To our knowledge, our method is the first one designed for automatic portrait segmentation. The main contributions of our approach are:

- We extend the FCN-8s framework [LSD14] to leverage domain specific knowledge by introducing new portrait position and shape input channels.
- We build a portrait image segmentation dataset and benchmark for our model training and testing.
- We augment several interactive portrait editing methods with our method to make them fully automatic.

就我们所知，我们的方法是第一个为自动纵向分割而设计的方法。 我们的方法的主要贡献是：

- 我们扩展了FCN-8框架[LSD14]，通过引入新的纵向位置和形状输入通道来利用领域特定的知识。
- 我们为我们的模型训练和测试建立一幅肖像图像分割数据集和基准。
- 我们使用我们的方法增强了多种交互式肖像编辑方法，使其完全自动化。

## 2 Related Work

![](http://upload-images.jianshu.io/upload_images/5952841-4399dd9e52c6f8b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 2: Different automatic portrait segmentation results. (a) and (b) are the input and ground truth respectively. (c) is the result of applying graph-cut initialized with facial feature detector data. (d) is the result of the FCN-8s (person class). (e) is the FCN-8s network fine-tuned with our portrait dataset and reduced to two output channels which we named as PortraitFCN. (f) is our new PortraitFCN model which augments (e) with portrait-specific knowledge.

图2：不同的自动肖像分割结果。 （a）和（b）分别是输入和基本事实。 （c）是使用面部特征检测器数据应用图形切割初始化的结果。 （d）是FCN-8（人员类别）的结果。 （e）是用我们的肖像数据集精调的FCN-8s网络，并简化为两个我们命名为PortraitFCN的输出频道。 （f）是我们新的PortraitFCN模型，它增加了（e）与肖像特定的知识。

Our work is related to work in both image segmentation and image stylization. The following sections provide a brief overview on several main segmentation methodologies (interactive,learnin gbased, and image matting), as well as some background on various portrait-specific stylization algorithms.

我们的工作是关于图像分割和图像风格化的工作。 以下部分简要介绍了几种主要的分割方法（交互式，学习式和图像式）以及各种特定于人像的风格化算法的背景知识。

### 2.1. Interactive Image Selection

We divide interactive image segmentation methods into scribble-based, painting-based and boundary-based methods. In the scribble-based methods, the user specifies a number of foreground and background scribbles as boundary conditions for a variety of different optimizations including graph cut methods [BJ01,LSTS04,RKB04], geodesic distance scheme [BS07], random walks framework [Gra06] and the dense CRF method [KK11].

我们将交互式图像分割方法划分为基于涂色，基于绘画和边界的方法。 在基于涂鸦的方法中，用户指定许多前景和背景涂鸦作为用于各种不同优化的边界条件，包括图形切割方法[BJ01，LSTS04，RKB04]，测地距离方案[BS07]，随机游走框架[Gra06 ]和密集的CRF方法[KK11]。

Compared with scribble-based methods, the painting based method only needs to paint over the object the user wants to select. Popular methods and implementations include painting image selection [LSS09], and Adobe Photoshop quick selection [ADO].

与基于涂鸦的方法相比，基于绘画的方法只需要绘制用户想要选择的对象。 流行的方法和实现包括绘制图像选择[LSS09]和Adobe Photoshop快速选择[ADO]。

The object can also be selected by tracing along the boundary. For example, Snakes [KWT88] and Intelligent Scissors [MB95] compute the object boundary by tracking the user’s input rough boundaries. However, this requires accurate user interactions which can be very difficult, especially in the face of complicated boundaries.

该对象也可以通过沿边界追踪来选择。 例如，Snakes [KWT88]和Intelligent Scissors [MB95]通过跟踪用户输入的粗糙边界来计算物体边界。 但是，这需要准确的用户交互，这可能是非常困难的，尤其是在面临复杂的边界时。

Although the interactive selection methods are prevalent in image processing software, their tedious and complicated interaction limits many potentially automatic image processing applications.

尽管交互式的选择方法在图像处理软件中很普遍，但是它们繁琐而复杂的交互限制了很多潜在的自动图像处理应用。

### 2.2. CNNs for Image segmentation

A number of approaches based on deep convolutional neural networks (CNNs) have been proposed to tackle image segmentation tasks. They apply CNNs in two main ways. The first one is to learn the meaningful features and then apply classification methods to infer the pixel label. Representative methods include [AHG∗12,MYS14,FCNL13], but they are optimized to work for a lot of different classes, rather than focusing specifically on portraits. As with our FCN-8s tests, one can use their “person class” for segmentation, but the results are not accurate enough on portraits to be used for stylization.

已经提出了许多基于深度卷积神经网络（CNN）的方法来处理图像分割任务。 他们以两种主要方式应用CNN。 首先是学习有意义的特征，然后应用分类方法来推断像素标签。 代表性的方法包括[AHG * 12，MYS14，FCNL13]，但它们被优化为适用于许多不同的类别，而不是特别专注于肖像。 与我们的FCN-8s测试一样，人们可以使用他们的“人物类”进行分割，但是对于用于风格化的肖像，结果不够精确。

The second way is to directly learn a nonlinear model from the images to the label map. Long et al. [LSD14] introduce fully convolutional networks in which several well-known classification networks are “convolutionalized”. In their work, they also introduce a skip architecture in which connections from early layers to later layers were used to combine low-level and high-level featurecues. Following this framework, DeepLab [CPK∗14] and CRFas-RNN [ZJR∗15] apply dense CRF optimization to refine the CNNs predicted label map. Because deep CNNs need large-scale training data to achieve good performance, Dai et al. [DHS15] proposed the BoxSup which only requires easily obtained bounding box annotations instead of the pixel labeled data. It produced comparable results compared with the pixel labeled training data under the same CNNs settings.

第二种方法是直接从图像学习非线性模型到标签图。 Long等人 [LSD14]引入了完全卷积网络，其中几个着名的分类网络被“卷积”。 在他们的工作中，他们还引入了一种跳过架构，在这种架构中，从早期层到后期层的连接被用来组合低层和高层功能。 在此框架之后，DeepLab [CPK * 14]和CRFas-RNN [ZJR * 15]应用密集的CRF优化来细化CNN预测的标签图。 由于深度CNN需要大规模的训练数据才能取得良好的表现，Dai等 [DHS15]提出的BoxSup只需要容易获得的边界框注释而不是像素标记的数据。 与相同的CNN设置下的像素标记的训练数据相比，它产生了可比较的结果。

These CNNs were designed for image segmentation tasks and the state-of-the-art accuracy for Pascal VOC is around 70%. Although they outperform other methods, the accuracy is still insufficient for inclusion in an automatic portrait processing system.

这些CNN是为图像分割任务设计的，而Pascal VOC的最新精度约为70％。 虽然它们胜过其他方法，但准确性仍然不足以包含在自动肖像处理系统中。

### 2.3. Image Matting

Image matting is the other important technique for image selection. For natural image matting, a thorough survey can be found in [WC07]. Here we review some popular works relevant to our technique. The matting problem is ill-posed and severely under-constrained. These methods generally require initial user defined foreground and background annotations, or alternatively a trimap which encodes the foreground, background and unknown matte values. According to different formulations, the matte’s unknown pixels can be estimated by Bayesian matting [CCSS01], Poisson matting [SJTS04], Closed-form matting [LLW08], KNN matting [CLT13], etc. To evaluate the different methods, Rhemann et al. [RRW∗09] proposed a quantitative online benchmarks. For our purposes, the disadvantages of these methods is their relianceon the user to specify the trimap.

图像抠像是图像选择的另一个重要技术。 对于自然图像抠图，可以在[WC07]中找到详细的调查。 在这里我们回顾一些与我们的技术相关的热门作品。扣图问题是病态的并且受约束的。 这些方法通常需要初始的用户定义的前景和背景注释，或者是对前景，背景和未知遮罩值进行编码的trimap。 根据不同的公式，通过Bayesian matting [CCSS01]，Poisson matting [SJTS04]，Closed-matting [LLW08]，KNN matting [CLT13]等方法可以估计遮罩的未知像素。为了评估不同的方法，Rhemann等。 [RRW * 09]提出了一个定量的在线基准。 就我们的目的而言，这些方法的缺点是它们依赖于用户来指定trimap。

### 2.4. Semantic Stylization

Our portrait segmentation technique incorporates high level semantic understanding of portrait images to help it achieve state of the art segmentation results which can then be used for subject-aware portrait stylization. Here we highlight a sampling of other works which also take advantage of portrait-specific semantics for image processing and stylization. [SPB∗14] uses facial feature locations and sift flow to create robust dense mappings between user input portraits, and professional examples to allow for facial feature-accurate transfer of image style. In [LCODL08], a database of inter-facialfeature distance vectors and user attractiveness ratings is used to compute 2D warp fields which can take an input portrait, and automatically remap it to a more attractive pose and expression. And finally [CLR∗04] is able to generate high-quality non-photorelistic drawings by leveraging a semantic decomposition of the main face features and hair for generating artistic strokes.

我们的肖像分割技术结合了对肖像图像的高级语义理解，以帮助它实现最先进的分割结果，然后可以用于主题感知肖像风格化。 在这里，我们重点介绍一些其他作品的样本，这些作品还利用肖像特定语义进行图像处理和风格化。 [SPB * 14]使用面部特征位置和筛选流程在用户输入肖像之间创建鲁棒的密集映射，并提供专业示例，以实现面部特征精确的图像样式转换。 在[LCODL08]中，使用面部特征间距向量和用户吸引力评级的数据库来计算2D扭曲场，其可以采用输入肖像，并自动将其重新映射到更有吸引力的姿势和表情。 最后[CLR * 04]能够通过利用主要脸部特征和头发的语义分解来产生高质量的扣图绘画，以产生艺术笔触。

## 3. Our Motivation and Approach

Deep learning achieves state-of-the-art performance on semantic image segmentation tasks. Our automatic portrait segmentation method also applies deep learning to the problem of semantic segmentation, while leveraging portrait-specific features. Our framework is shown in Figure 3 and is detailed in Section 3.3. We start with a brief description of the fully convolutional neural network (FCN) [LSD14] upon which our technique is built.

深度学习在语义图像分割任务上达到了最先进的性能。 我们的自动肖像分割方法还将深度学习应用于语义分割问题，同时利用肖像特定的特征。 我们的框架如图3所示，详见3.3节。 我们首先简要描述我们的技术构建的完全卷积神经网络（FCN）[LSD14]。

![](http://upload-images.jianshu.io/upload_images/5952841-a60aee3f375ba623.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 3: Pipeline of our automatic portrait segmentation framework. (a) is the input image and (b) is the corresponding cropped portrait image by face detector. (d) is a template portrait image. (e) is the mean mask and normalized x- and y- coordinate. (c) shows the output with the PortraitFCN+ regression. The input to the PortraitFCN+ is the aligned mean mask , normailized x- and y-, and the portrait RGB channels.

图3：我们的自动肖像分割框架的流程。 （a）是输入图像，（b）是人脸检测器对应的裁剪后的肖像图像。 （d）是模板肖像图像。 （e）是平均掩模和归一化的x和y坐标。 （c）用PortraitFCN +回归显示输出。 PortraitFCN +的输入是对齐的平均蒙板，正常化的x和y，以及纵向的RGB通道。

### 3.1. Fully Convolutional Neutral Networks

As mentioned in the previous section, many modern semantic image segmentation frameworks are based on the fully convolutional neutral network (FCN) [LSD14] which replaces the fully connected layers of a classification network with convolutional layers. The FCN uses a spatial loss function and is formulated as a pixel regression problem against the ground-truth labeled mask. The objective function can be written as,

如前所述，许多现代语义图像分割框架都是基于完全卷积中性网络（FCN）[LSD14]，它用卷积层代替分类网络的完全连接层。 FCN使用空间损失函数，并将其表述为针对真实数据标记掩码的像素回归问题。 目标函数可以写成，

![](http://upload-images.jianshu.io/upload_images/5952841-558960a62cebb28e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

where $p$ is the pixel index of an image. $X_{\theta}(p)$ is the FCN regression function in pixel $p$ with parameter $\theta$. The loss function $e(...)$ measures the error between the regression output and the ground truth $\ell(p)$. FCNs are typically composed of the following types of layers:

其中$p$是图像的像素索引。 $X_{\theta}(p)$是参数$\theta$下$p$位置像素的FCN回归函数。 损失函数$e(...)$测量回归输出和真值$\ell(p)$之间的误差。 FCN通常由以下类型的层组成：

**Convolution Layers** This layer applies a number of convolution kernels to the previous layer. The convolution kernels are trained to extract important features from the images such as edges, corners or other informative region representations.

**卷积层** 该层将大量的卷积核应用到前一层。 训练卷积核以从图像中提取重要的特征，例如边缘，角落或其他信息区域表示。

**ReLU Layers** The ReLU is a nonlinear activation to the input. The function is $f(x) = \max(0,x)$. This nonlinearity helps the network compute nontrivial solutions on the training data.

**ReLU层** ReLU是对输入的非线性激活。 函数是$f(x) = \max(0,x)$。 这种非线性有助于网络计算训练数据的非平凡解。

**Pooling Layers** These layers compute the max or average value of a particular feature over a region in order to reduce the feature’s spatial variance.

**池化层** 这些层计算区域上特定特征的最大值或平均值，以减少特征的空间变化。

**Deconvolution Layers** Deconvolution layers learn kernels to upsample the previous layers. This layer is central in making the output of the network match the size of the input image after previous pooling layers have downsampled the layer size.

**解卷积层** 解卷积层学习卷积核去上采样前一层。 在先前的池化层下采样网络层大小后，该层是使网络的输出与输入图像的大小相匹配的重点。

**Loss Layer** This layer is used during training to measure the error (Equation 1) between the output of the network and the ground truth. For a segmentation labeling task, the loss layer is computed by the softmax function.

**损失层** 在训练期间使用该层来测量网络输出与真实值之间的误差（公式1）。 对于分割标签任务，损失层由softmax函数计算。

Weights for these layers are learned by backpropagation using stochastic gradient descent (SGD) solver.

通过使用随机梯度下降（SGD）方法的反向传播学习这些层的权重。

### 3.2. Understandings for Our Task

![](http://upload-images.jianshu.io/upload_images/5952841-8d825491d5416973.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 4:Comparisons of FCN-8s applying to portrait data. (a) isthe input image. (b) is the person class output of FCN-8s and (c) isthe output of PortraitFCN. (d) is our PortraitFCN+.

图4：应用于肖像数据的FCN-8的比较。 （a）是输入图像。 （b）是FCN-8的人类输出，（c）是PortraitFCN的输出。 （d）是我们的PortraitFCN +。

The fully convolutional network (FCN) used for this work is originally trained for semantic object segmentation on the Pascal VOC dataset for twenty class object segmentation. Although the dataset includes a person class, it still suffers from poor segmentation accuracy on our portrait image dataset as shown in Figure 4 (b). The reasons are mainly: 1) The low resolution of people in the Pascal VOC constrains the effectiveness of inference on our high resolution portrait image dataset. 2) The original model outputs multiple labels to indicate different object classes which introduces ambiguities in our task which only needs two labels. We address these two issues by labeling a new portrait segmentation dataset for fine-tuning the model and changing the label outputs to only the background and the foreground. We show results of this approach and refer to it in the paper as PortraitFCN.

用于这项工作的完全卷积网络（FCN）最初是针对二十类对象分割的Pascal VOC数据集上的语义对象分割进行训练的。 尽管数据集包含了一个人类，但是仍然会对我们的肖像图像数据集的分割精度不好，如图4（b）所示。 其原因主要是：1）Pascal VOC分辨率低，限制了我们高分辨率人像图像数据集推理的有效性。 2）原始模型输出多个标签来指示不同的对象类别，这在我们的任务中引入了模糊性，其仅需要两个标签。 我们通过标记新的纵向分割数据集来解决这两个问题，以便微调模型并将标签输出更改为只有背景和前景。 我们展示了这种方法的结果，并在论文中将其称为PortraitFCN。

Although PortraitFCN improves the accuracy for our task as shown in Figure 4 (c), it is still has issues with clothing and background regions. A big reason for this is the translational invariance that is inherent in CNNs. Subsequent convolution and pooling layers incrementally trade spatial information for semantic information. While this is desirable for tasks such as classification, it means that we lose information that allows the network to learn, for example, the pixels that are far above and to the right of the face in 4 (c) are likely background.

虽然PortraitFCN如图4（c）所示提高了我们的任务的准确性，但是在服装和背景区域仍然存在问题。 其中一个重要原因就是CNN固有的平移不变性。 随后的卷积和池化层逐步交换空间信息以获取语义信息。 虽然这对分类这样的任务来说是可取的，但这意味着我们失去了允许网络学习的信息，例如，4（c）中上方和脸部右侧看上去应是背景。

To mitigate this, we propose the PortraitFCN+ model, described next, which injects spatial information extracted from the portrait, back into the FCN.

为了减轻这一点，我们提出了PortraitFCN +模型，下面将介绍，它将从肖像中提取的空间信息注入到FCN中。

### 3.3. Our Approach

Our approach incorporates portrait-specific knowledge into the model learned by our CNNs. To accomplish this, we leverage robust facial feature detectors [SLC09] to generate auxiliary position and shape channels. These channels are then included as inputs along with the portrait color information into the first convolutional layer of our network.

我们的方法将用CNNs学习到的特定肖像的知识注入到了模型中。 为了做到这一点，我们利用鲁棒的面部特征检测器[SLC09]来生成辅助的位置通道和形状通道。 然后将这些通道作为输入，并与肖像的颜色信息一同包含到网络的第一个卷积层。（译者注：如图3下方所示的网络结构第一层，由5个Channels组成，分别是R、G、B、Mean Mask、Normalized x and y，后两个分别为位置通道和形状通道。）

**Position Channels** The objective of these channels is to encode the pixel positions relative to the face. The input image pixel position only gives limited information about the portrait because the subjects are framed differently in each picture. This motivates us to provide two additional channels to the network, the _normalized x and y_ channels where _x_ and _y_ are the pixel coordinates. We define them by first detecting facial feature points [SLC09] and estimating a homography transform $\tau$ between the fitted features and a canonical pose as shown in Figure 3 (d). We defined the normalized _x_ channel as $\tau(x_{img})$ where $x_{img}$ is the _x_ coordinate of the pixels with its zero in face center in the image. We define the normalized _y_ channel similarly. Intuitively, this procedure expresses the position of each pixel in a coordinate system centered on the face and scaled according to the face size.

**位置通道** 这些通道的目的是编码相对于脸部的像素位置。 输入图像的像素位置只给出了关于肖像的有限信息，因为目标在不同的图像中具有不同的外框。 这鼓励我们为网络提供两个额外的通道， _normalized x and y_ 通道，其中 _x_ 和 _y_ 是像素坐标。 我们通过首先检测面部特征点[SLC09]并估计拟合特征和典型姿态之间的单应变换 $\tau$ 来定义它们，如图3（d）所示。 我们将归一化的 _x_ 通道定义为 $\tau(x_{img})$ ，其中 $x_{img}$ 是图像中脸部中心为零的像素的x坐标。 我们同样类似地定义标准化的 _y_ 通道。 直观地说，该过程表示在以脸部为中心的坐标系中各个像素的位置，并根据面部尺寸进行了缩放。

**Shape Channel** In addition to the position channel, we found that adding a shape channel further improves segmentation. A typical portrait includes the subject’s head and some amount of the shoulders, arms, and upper body. By including a channel in which a subject-shaped region is aligned with the actual portrait subject, we are explicitly providing a feature to the network which should be a reasonable initial estimate of the final solution. To generate this channel, we first compute an aligned average mask from our training dataset. For each training portrait-mask pair {Pi,Mi}, we transform Mi using a homography Ti which is estimated from the facial feature points of Pi and a canonical pose. We compute the mean of these transformed masks as:

**形状通道** 除了位置通道，我们发现添加一个形状通道进一步改善了分割。 典型的肖像包括主体的头部和肩膀、手臂以及上半身。 通过包含一个形状区域目标与真实肖像目标对齐的通道，我们明确地向网络提供一个特征，该特征应该是最终结果的合理初始估计。 为了生成这个通道，我们首先从我们的训练数据集中计算一个对齐的均值掩膜。 对于每个进行训练的肖像-掩膜对{Pi，Mi}，我们使用单应性Ti来变换Mi，该Ti是从Pi的面部特征点和典型姿态的估计得到的。 我们计算这些变换后的掩膜的平均值为：

![](http://upload-images.jianshu.io/upload_images/5952841-4969950985983028.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

where wi is a matrix indicating whether the pixel in Mi is outside the image after the transform Ti. The value will be 1 if the pixel is inside the image, otherwise, it is set as 0. The operator ◦ denotes element-wise multiplication. This mean mask M which has been aligned to a canonical pose can then be similarly transformed to align with the facial feature points of the input portrait.

其中wi是一个矩阵，指示Mi中的像素经过Ti变换后是否处于图像之外。 如果像素位于图像内，则值为1，否则设为0。 运算符◦表示元素乘法。 这意味着已经和典型姿态对齐的掩膜M接下来可以类似地变换为与输入肖像的脸部特征点对齐。

Figure 3 shows our PortraitFCN+ automatic portrait segmentation system including the additional position and shape input channels. As shown in Figure 4, our method outperforms all other tested approaches. We will quantify the importance of the position and shape channels in Section 5.1.

图3显示了我们的PortraitFCN +自动人像分割系统，包括附加的位置和形状输入通道。 如图4所示，我们的方法胜过所有其他测试方法。 我们将在5.1节中量化位置和形状通道的重要性。

## 4. Data and Model Training

Since there is no portrait image dataset for segmentation, we labeled a new one for our model training and testing. In this section we detail the data preparation and training schemes.

由于没有用于分割的肖像图像数据集，我们为我们的模型训练和测试标记了一个新的数据集。 在本节中，我们将详细介绍数据准备和训练计划。

![](http://upload-images.jianshu.io/upload_images/5952841-c7044280e4bc6c6b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 5: Some example portrait images with different variations in our dataset.

图5：在我们的数据集中有不同变化的一些示例肖像图像。

**Data Preparation** We collected 1800 portrait images from Flickr and manually labeled them with Photoshop quick selection. We captured a range of portrait types but biased the Flickr searches toward natural self portraits that were captured with mobile frontfacing cameras. These are challenging images that represent the typical cases that we would like to handle. We then ran a face detector on each image, and automatically scaled and cropped the image to 600×800 according the bounding box of the face detection result as shown in Figure 3(a) and (b). This process excludes images for which the face detector failed. Some of the portrait images in our dataset are shown in Figure 5 and display large variations in age, color, background, clothing, accessories, head position, hair style, etc. We include such large variations in our dataset to make our model more robust to challenging inputs. We split the 1800 labeled images into a 1500 image training dataset and a 300 image testing/validation dataset. Because more data tends to produce better results, we augmented our training dataset by perturbing the rotations and scales of our original training images. We synthesize four new scales {0.6,0.8,1.2,1.5} and four new rotations {−45◦,−22◦,22◦,45◦}. We also apply four different gamma transforms to get more color variation. The gamma values are {0.5,0.8,1.2,1.5}. With these transforms, we generate more than 19,000 training images.

**数据准备**我们从Flickr收集了1800幅肖像图像，并用Photoshop快速选择进行了手动标记。我们捕捉到了一系列肖像类型，但是将Flickr搜索偏向于用移动前摄像头拍摄的自然肖像。这些是具有挑战性的图像，代表了我们想要处理的典型案例。然后，我们在每幅图像上运行一个人脸检测器，并根据人脸检测结果的边界框自动缩放并裁剪成600×800，如图3（a）和（b）所示。此过程排除脸部检测器失败的图像。我们的数据集中的一些肖像图像如图5所示，并且在年龄，颜色，背景，服装，饰品，头部位置，发型等方面显示出很大的变化。我们在数据集中包含如此大的变化以使我们的模型对输入更加地鲁棒。我们将1800个标记的图像分成1500个图像训练数据集和一个300个图像测试/验证数据集。由于更多的数据往往会产生更好的结果，我们通过扰动我们的原始训练图像的旋转和尺度来增强我们的训练数据集。我们综合了四个新的尺度{0.6,0.8,1.2,1.5}和四个新的旋转{-45°，-22°，22°，45°}。我们也应用四种不同的伽马变换来获得更多的颜色变化。伽玛值是{0.5,0.8,1.2,1.5}。通过这些转换，我们生成了超过19,000个训练图像。

**Model Training** We setup our model training and testing experiment in Caffe [JSD∗14].With the model illustrated in Figure 3, we use a stochastic gradient descent (SGD) solver with softmax loss function. We start with a FCN-8s model which pre-trained on the PASCAL VOC 2010 20-class object segmentation dataset. While it is preferable to incrementally fine-tune, starting with the topmost layer and working backward, we have to fine-tune the entire network since our pre-trained model does not contain weights for the aligned mean mask and x and y channels in the first convolutional layer.We initialize these unknown weights with random values and fine-tune with a learning rate of 10−4. As is common practice in fine-tuning neural networks, we select this learning rate by trying several rates and visually inspecting the loss as shown in Figure 6. We found that too small and too large learning rate did not successfully converge or over fitting. 

**Running Time for Training and Testing** We conduct training and testing on a single Nvidia Titan X GPU. Our model training requires about one day to learn a good model with about 40,000 Caffe SGD iterations. For the testing phase, the running time on a 600×800 color image is only 0.2 second on the same GPU. We also run our experiment on the Intel Core i7-5930K CPU which takes 4 seconds using the MKL-optimized build of Caffe.

## 5. Results and Applications

Our method achieved substantial performance improvements over other methods for the task of automatic portrait segmentation. We provide a detailed comparison to other approaches. A number of applications are also conducted because of the high performance segmentation accuracy.
