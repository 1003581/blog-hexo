---
title: 图像处理岗位面试题搜罗汇总
date: 2017-09-14 15:59:00
tags: 
- 面试
- 机器视觉
categories: 机器学习
---

图像处理岗位面试题搜罗汇总
<!-- more -->

## Matlab编程

### Matlab 中读、写及显示一幅图像的命令各是什么？

imread(), imwrite(), imshow()

### Matlab 与VC++混合编程有哪几种方式？

Matlab引擎方式(Matlab后台程序为服务器，VC前端为客户端，C/S结构)、Matlab编译器（将Matlab源代码编译为C++可以调用的库文件）及COM组件（Matlab生成COM组件，VC调用）

### Matlab运算中 `.*`和 `*` 的区别？

`.*`表示矩阵元素分别相乘，要求两个矩阵具有相同的shape。`*`表示矩阵相乘。

## 图像处理基础部分

### Intel指令集中MMX,SSE,SSE2,SSE3和SSE4指的是什么？

MMX（Multi Media eXtension，多媒体扩展指令集）是一些整数并行运算指令。

SSE（Streaming SIMD Extensions，单指令多数据流扩展）是一系列浮点并行运算指令。

### 并行计算有哪些实现方式？

单指令多数据流SIMD、对称多处理机SMP、大规模并行处理机MPP、工作站机群COW、分布共享存储DSM多处理机。

### 彩色图像、灰度图像、二值图像和索引图像区别？

彩色图像：RGB图像。灰度图像：0-255像素值。二值图像：0和1，用于掩膜图像。

索引图像：在灰度图像中，自定义调色板，自定义输出256种颜色值。

### 常用边缘检测有哪些算子，各有什么特性？

1. **Sobel算子**：典型的基于一阶导数的边缘检测算子，对于像素的位置的影响做了加权，可以降低边缘模糊程度。
    
    不足：没有将图像的主体与背景严格地区分开来, 没有基于图像灰度进行处理.

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$
1. **Isotropic Sobel**：各向同性Sobel(Isotropic Sobel)算子。各向同性Sobel算子和普通Sobel算子相比，它的位置加权系数更为准确，在检测不同方向的边沿时梯度的幅度一致。

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 & +1 \\ -\sqrt2 & 0 & +\sqrt2 \\ -1 & 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & -\sqrt2 & -1 \\ 0 & 0 & 0 \\ +1 & +\sqrt2 & +1 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$
1. **Roberts算子**：一种利用局部差分算子寻找边缘的算子。

    不足：对噪声敏感,无法抑制噪声的影响。

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 \\ +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & +1 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$

    检测对角线方向梯度时：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 \\ 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} 0 & -1 \\ +1 & 0 \end{matrix} \right]
    G = \sqrt{ {G_x}^2 + {G_y}^2 }
    $$
1. **Prewitt算子**：Sobel是该算子的改进版。

    卷积核和像素更新公式如下：

    $$
    G_x=\left[ \begin{matrix} -1 & 0 & +1 \\ -1 & 0 & +1 \\ -1 & 0 & +1 \end{matrix} \right]
    G_y=\left[ \begin{matrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ +1 & +1 & +1 \end{matrix} \right]
    G_x'=\left[ \begin{matrix} -1 & -1 & 0 \\ -1 & 0 & +1 \\ 0 & +1 & +1 \end{matrix} \right]
    G_y'=\left[ \begin{matrix} 0 & 1 & 1 \\ -1 & 0 & 1 \\ -1 & -1 & 0 \end{matrix} \right]
    $$
1. **Laplacian算子**：拉普拉斯算子,各向同性算子，二阶微分算子,只适用于无噪声图象,存在噪声情况下，使用Laplacian算子检测边缘之前需要先进行低通滤波。

    卷积核和像素更新公式如下：
    
    $$
    R=\left[ \begin{matrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{matrix} \right]
    G=\left\{ \begin{matrix} 1 & |R(x,y)| \ge T \\ 0 & others \end{matrix} \right.
    $$
1. **Canny算子**：一个具有滤波，增强，检测的多阶段的优化算子。先利用高斯平滑滤波器来平滑图像以除去噪声，采用一阶偏导的有限差分来计算梯度幅值和方向，然后再进行非极大值抑制。
1. **Laplacian of Gaussian(LoG)算子**：先对图像做高斯滤波，再做Laplacian算子检测。

### 简述BP神经网络

BP(back propagation)神经网络，输入X，通过隐藏节点的非线性变换后，输出信号Y，通过误差分析，来调整隐藏节点的W和b。

### AdBoost的基本原理？

AdBoost是一个广泛使用的BOOSTING算法，其中训练集上依次训练弱分类器，每次下一个弱分类器是在训练样本的不同权重集合上训练。权重是由每个样本分类的难度确定的。分类的难度是通过分类器的输出估计的。

## C/C++部分

### 关键字static的作用是什么？

1. 在函数体，一个被声明为静态的变量在这一函数被调用过程中维持其值不变。
1. 在模块内（但在函数体外），一个被声明为静态的变量可以被模块内所用函数访问，但不能被模块外其它函数，它是一个本地的全局变量。
1. 在模块内，一个被声明为静态的函数只可被这一模块的它函数调用。那就是，这个函数被限制在声明它的模块的本地范围内使用。