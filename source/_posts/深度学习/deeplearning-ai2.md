---
title: deeplearning.ai2
date: 2017-11-06 23:30:00
tags: 深度学习
categories: 深度学习
---

课程在[网易云课堂](https://study.163.com/provider/2001053000/index.htm)上免费观看
<!-- more -->
# 卷积神经网络

## 第一周 卷积神经网络

### 理论

PADDING：VALID SAME

valid padding: NxN->N-f+1 x N-f+1

为何卷积核的尺寸是奇数：

1. 计算机视觉的惯例，有名的边缘检测算法都是奇数。
1. 一个中心点的话，会比较好描述模版处于哪个位置。
1. SAME会比较自然的填充。
1. 会计算像素之间亚象素的过渡

原图为n×n，卷积核为f×f，步长为s，PADDING大小为p，处理后图像大小为 (n+2p-f)/s+1 × (n+2p-f)/s+1，除法为向下整除 

单层卷积神经网络

If layer $l$ is a convolution layer:

$f^{[l]}$ = filter size  
$p^{[l]}$ = padding  
$s^{[l]}$ = stride  
$n_c^{[l]}$ = number of filters  

Input: $n_H^{[l-1]} \times n_W^{[l-1]} \times n_c^{[l-1]}$  
Output: $n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$  

**卷积运算中，会先矩阵相乘，然后加上偏置，然后进入激活函数，得到卷积结果。**

**池化层没有参数**

$$
n_{H/W}^{[l]}=\lfloor \frac{n_{H/W}^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \rfloor
$$

$$
A^{[l]} \rightarrow m \times n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}
$$

Each filter is: $f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$  
Activations: $a^{[l]} \rightarrow n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$  
Weights: $f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}$  
biases: $n_c^{[l]} \rightarrow (1,1,1,n_c^{[l]})$ 

池化层的超参数

f: filter size 常用2、3  
s: stride  常用2    

为什么使用卷积？

- 参数共享 parameter sharing
    - 垂直边缘特征检测器适用于图像全部区域
- 稀疏连接 sparsity of connections
    - 在每一层中，每个输出值仅仅依赖于很小的一块输入

## 第二周 深度卷积神经网络

### 笔记

#### Classic networks 经典网络

##### LeNet-5

![img](http://upload-images.jianshu.io/upload_images/5952841-81ca9cbad2d6a00f.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

1. `32x32x1` --(`conv f=5 s=1 VALID`)--> `28x28x6`
1. --(`avg-pool f=2 s=2`)--> `14x14x6`
1. --(`conv f=5 s=1 VALID`)--> `10x10x16`
1. --(`avg-pool f=2 s=2`)--> `5x5x16`
1. `5x5x16=400` --(`fc`)--> `120`
1. --(`fc`)--> `84`
1. --(`softmax`)--> `10 objects`

- 大约60K，6万个参数
- 论文中网络使用了Sigmoid和Tanh
- 经典论文中还在池化后加入了激活函数

##### AlexNet

![img](http://upload-images.jianshu.io/upload_images/5952841-4a65eaf72b047c42.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

1. `227x227x3` --(`conv f=11 s=4 VALID`)--> `55x55x96`
1. --(`max-pool f=3 s=2`)--> `27x27x96`
1. --(`conv f=5 SAME`)--> `27x27x256`
1. --(`max-pool f=3 s=2`)--> `13x13x256`
1. --(`conv f=3 SAME`)--> `13x13x384`
1. --(`conv f=3 SAME`)--> `13x13x384`
1. --(`conv f=3 SAME`)--> `13x13x256`
1. --(`max-pool f=3 s=2`)--> `6x6x256`
1. `6x6x256=9216` --(`fc`)--> `4096`
1. --(`fc`)--> `4096`
1. --(`softmax`)--> `1000 objects`


- 大约60M，6千万个参数 
- 论文中使用了ReLU激活函数
- 经典ALexNet中包含局部响应归一化层，用于将通道之间相同位置上的像素进行归一化。

![img](http://upload-images.jianshu.io/upload_images/1689929-063fb60285b6ed42.png?imageMogr2/auto-orient/strip%7CimageView2/2)

##### VGG-16

![img](http://upload-images.jianshu.io/upload_images/5952841-ca64c8f9427eff24.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

只用了2种网络：`CONV(f=3 s=1 SAME)` `MAX-POOL(f=2 s=2)`

1. `224x224x3` --(`CONV 64`)x2--> `224x224x64`
1. --(`MAX-POOL`)--> `112x112x64`
1. --(`CONV 128`)x2--> `112x112x128`
1. --(`MAX-POOL`)--> `56x56x128`
1. --(`CONV 256`)x3--> `56x56x256`
1. --(`MAX-POOL`)--> `28x28x256`
1. --(`CONV 512`)x3--> `28x28x512`
1. --(`MAX-POOL`)--> `14x14x512`
1. --(`CONV 512`)x3--> `14x14x512`
1. --(`MAX-POOL`)--> `7x7x512`
1. `7x7x512=25088` --(`fc`)--> `4096`
1. --(`fc`)--> `4096`
1. --(`softmax`)--> `1000 objects`

- VGG-16中16代表总共含有16层（卷积+FC）。
- 大约含有138M，1.38亿个参数

#### 残差网络 ResNet(152 layers)

![img](http://upload-images.jianshu.io/upload_images/5952841-8230bb999744d22c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![img](http://upload-images.jianshu.io/upload_images/5952841-421c3c2fdd9d1615.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

正常网络 plain network

$$
a^{[l+2]}=g(z^{[l+2]})
$$

残差网络 residual network

$$
a^{[l+2]}=g(z^{[l+2]} + a^{[l]})
$$

当$W^{[l+2]}$,$b^{[l+2]}$接近为0，使用ReLU时，$a^{[l+2]}=g(a^{[l]})=a^{[l]}$，这称之为学习恒等式，中间一层不会对网络的性能造成影响，而且有时会还学习到一些有用的信息。

残差块的矩阵加法要求维度相同，故需要添加一个矩阵，$W_s$，即$a^{[l+2]}=g(z^{[l+2]} + W_s a^{[l]})$，该参数属于学习参数。

![img](http://upload-images.jianshu.io/upload_images/5952841-7402517207ae0344.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 1x1卷积

在每个像素上的深度上的全连接运算。可以用来改变通道深度，或者对每个像素分别添加了非线性变换。

Network in Network

#### Inception

一个Inception模块，帮你解决使用什么尺寸的卷积层和何时使用池化层。

![img](http://upload-images.jianshu.io/upload_images/5952841-ab356377d7ef393d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了解决计算成本问题，引入1x1卷积进行优化计算。

![img](http://upload-images.jianshu.io/upload_images/5952841-8c7a2c347246c03d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

事实证明，只要合理构建瓶颈层，不仅不会降低网络性能，还会降低计算成本。

具体模块

![img](http://upload-images.jianshu.io/upload_images/5952841-a17e711a97eae00d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

具体网络

![img](http://upload-images.jianshu.io/upload_images/5952841-a3afa54b3f16a255.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 迁移学习

冻结一部分网络，自己训练一部分网络，并替换输出层的softmax

![img](http://upload-images.jianshu.io/upload_images/5952841-a1fb8b637b7939ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 数据增强

- 常用操作
    - 镜像操作
    - 随机修剪、裁剪
- 颜色偏移
    - 颜色通道分别加减值，改变RGB
    - PCA颜色增强算法

从硬盘中读取数据并且进行数据增强可以在CPU的线程中实现，并且可以与训练过程并行化。

## 第三周 目标检测

### 笔记

图像分类（图像中只有一个目标）->

目标定位（图像中只有一个目标）->

目标检测（图像中多个目标）

#### 目标定位

左上角(0,0)，右下角(1,1)

神经网络不仅输出类别，还输出bounding box (bx,by),(bh,bw)

输入图像如下，红框为标记位置。

![img](http://upload-images.jianshu.io/upload_images/5952841-1743d706ccb72cad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此分类任务中包含4个类别：

1. 行人pedestrian
1. 车辆car
1. 摩托车motorcycle
1. 无目标background

则Label y的维度为，其中$P_c$为是否存在目标，若不存在，则为0，(bx,by),(bh,bw)为目标的位置，c1,c2,c3为属于哪一类。

$$
y = \left[ \begin {matrix}
P_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3
\end {matrix} \right]
= \left[ \begin {matrix}
1 \\ 0.5 \\ 0.7 \\ 0.3 \\ 0.4 \\ 0 \\ 1 \\ 0
\end {matrix} \right]
$$

损失函数为分段函数，当$y_1$为0时，只考虑$y_1$的损失即可。当$y_1$为1时，需要考虑全部维度，各个维度采用不同的损失函数，如 $P_c$ 采用Logistic损失函数，bounding box (bx,by),(bh,bw)采用平方根，c1,c2,c3采用softmax中的对数表示方式。

#### 特征点检测

若想输出人脸中的眼角特征点的位置，则在神经网络的输出中添加4个数值即可。

比如人脸中包含64个特征点，则神经网络的输出层中添加64x2个输出。

#### 滑动窗口目标检测

首先需要训练裁剪过后的小图片。

然后针对输入的大图片，利用滑动窗口的技术对每个窗口进行检测。

将窗口放大，再次遍历整个图像。

将窗口再放大，再次遍历整个图像。

滑动窗口技术计算成本过高，

#### CNN中的滑动窗口

将网络中的FC转化为卷积层，实际效果一样。

![img](http://upload-images.jianshu.io/upload_images/5952841-163c6ca8ca8b5222.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

整个大图像做卷积运算。

![img](http://upload-images.jianshu.io/upload_images/5952841-82d270859df8497c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 边界框预测

YOLO算法（You Only Look Once）

将整个大图像划分为3x3、19x19这样的格子，然后修改Label Y，每个小格子中，若目标对象的中心点位于该格内，则该格Label Y中的$P_c$为1。相邻格子就算包含了目标对象的一部分，$P_c$也为0

#### 交并比

评价目标定位的指标

Intersection over Union(IoU)

交集面积/并集面积

一般认为，如果IoU >= 0.5，则认为是正确

#### 非最大值抑制

选定一份概率最大的矩形，然后抑制（减小其概率）与之交并比比较高的矩形。

#### Anchor Boxes

一个格子检测多个目标

#### YOLO算法

1. 将图像划分为3x3=9个格子，然后每个格子中包含2个anchor box，那么输出Y的维度为3x3x2x8。即$y=[p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3,p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3]^T$。对于没有目标的格子，输出为$y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]^T$，对于有一个车辆的格子，输出为$y=[0,0,0,0,0,0,0,0,1,b_x,b_y,b_h,b_w,0,1,0]^T$
1. 对于9个格子中的每一个，都会输出2个预测框。
1. 去掉预测值低的框。
1. 对于每一个类别（行人、车辆、摩托），运行非最大值抑制去获得最终预测结果。

#### RPN网络

Region proposal 候选区域

R-CNN 区域CNN

不再使用滑动窗口卷积，而是选择一些候选区域进行卷积，使用图像分割（Segmentation）算法选出候选区域。对区域进行卷积分类比较缓慢。算法不断优化。

Fast R-CNN

使用滑动窗口的卷积实现去分类所有的候选区域。但是区域候选区域的算法依然缓慢。

Faster R-CNN

使用卷积网络去检测候选区域，速度比Fast R-CNN快

## 第四周 特殊应用：人脸识别和神经风格转变

### 笔记

#### 术语

人脸检测face recognition和活体检测liveness detection

人脸验证face verification，1:1问题，验证name和face是否一一对应

人脸识别face recognition，1:k问题，一个人在不在这个库中。多次运行人脸验证。

#### one-shot learning

需要用这个人的一张照片去识别这个人，样本只有一个。

一种方法是将100个员工的人脸照片当作训练集，然后输出softmax 100个分类，但是这样识别效果并不好，且每加入一个新员工，都需要重新训练。

正确的方法是让深度学习网络学习一个相似函数similarity function，输入为2幅图像，输出为2幅图像之间的差异值。

#### Siamese Network

假设一个图像x1,通过一个卷积网络，得到了一个128维的向量$a^{[l]}$，不需要把$a^{[l]}$通过softmax，而是将这128维向量作为该图像的编码，称之为$f(x_1)$。

比较2幅图像的编码，判断他们的差异值。$d(x_1,x_2)=||f(x_1)-f(x_2)||^2_2$，差异小表示为同一个人，差异大为不同的人。

这样的网络称之为**Siamese Network Architecture**

#### Triplet 损失

三元组损失函数

需要同时看三组图像，Anchor图像、Positive图像、Negative图像。A、P、N

A和P是同一个人，A和N不是同一个人

$$
l(A,P,N)=\max(||f(A)-f(P)||^2-||f(A)-f(N)||^2+\alpha, 0)
$$

其中，$\alpha$是间隔值，比如取0.2。如果不设该值，若编码函数一直输出0，也会符合损失函数。

$$
J = \sum^m_{i=1}l(A^{i},P^{i},N^{i})
$$

训练集中必须包含一个人的至少2张照片，而实际使用时，可以one-shot，只用一张也可以。

A、P图像需成对使用，训练时使用难识别的图像。

Schroff 2015 FaceNet

#### 面部验证与二分类

输入为2幅图像， 输出为0或者1。同样使用上一节的编码。
