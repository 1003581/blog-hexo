---
title: deeplearning.ai笔记
date: 2017-11-06 23:30:00
tags: 深度学习
categories: 深度学习
---

课程在[网易云课堂](https://study.163.com/provider/2001053000/index.htm)上免费观看，作业题如下：加粗为答案。
<!-- more -->

# 神经网络和深度学习

[网址](https://mooc.study.163.com/course/2001281002?tid=2001392029#/info)

## 第一周 深度学习概论

## 第二周 Logistic Regression

### Logistic回归公式推导

样本个数 $m$, 训练样本个数 $m_{train}$, 同理 $m_{test}$,$m_{valid}$

#### 单个样本

##### forward propagate

input x的shape为$(n_x, 1)$，label y的shape为$(1, 1)$

$$
x = 
\left[
    \begin{matrix} 
    x_1 \\ 
    \vdots \\ 
    x_n 
    \end{matrix} 
\right]
\quad
y = \left[
    \begin{matrix}
    y
    \end{matrix}
    \right]
$$

weights w的shape为$(n_x, 1)$，biases b的shape为$(1, 1)$

$$
w =
\left[
    \begin{matrix}
    w_1 \\
    \vdots \\
    w_n
    \end{matrix}
\right]
\quad
b =
\left[
    \begin{matrix}
    b
    \end{matrix}
\right]
$$

z的shape为$(1, 1)$

$$
\begin {aligned}
z &= w^Tx+b \\
&=\left[
    \begin{matrix}
    w_1 &
    \cdots &
    w_n
    \end{matrix}
\right]
\left[
    \begin{matrix} 
    x_1 \\ 
    \vdots \\ 
    x_n 
    \end{matrix} 
\right]
+
b
\end {aligned}
$$

output $\hat{y}$的shape为$(1, 1)$

$$
\begin {aligned}
\hat{y} &= a \\
&= \sigma(z) \\
&= \frac{1}{1+e^{-z}}
\end {aligned}
$$

损失函数$l(\hat{y},y)$的shape为$(1, 1)$

$$
\begin {aligned}
l(\hat{y},y) &= l(a,y) \\
&= -(y\log{a}+(1-y)\log{(1-a)})
\end {aligned}
$$

##### backward propagate

损失函数对$\hat{y}$和$a$的偏导

$$
\begin {aligned}
d_a &= \frac{\partial l}{\partial a}(-y\log{a}-(1-y)\log{(1-a)}) \\
&= -\frac{y}{a} + \frac{1-y}{1-a}
\end {aligned}
$$

损失函数对$z$的偏导

$$
\begin {aligned}
d_z &= \frac{\partial l}{\partial a} \frac{\partial a}{\partial z} \\
&= (\frac{y}{a} + \frac{1-y}{1-a}) * \frac{\partial a}{\partial z}(\frac{1}{1+e^{-z}}) \\
&= (\frac{y}{a} + \frac{1-y}{1-a}) * (\frac{e^{-z}}{(1+e^{-z})^2}) \\
&= (\frac{y}{a} + \frac{1-y}{1-a}) * (\frac{1}{1+e^{-z}} * \frac{e^{-z}}{1+e^{-z}}) \\
&= (\frac{y}{a} + \frac{1-y}{1-a}) * (a * (1-a)) \\
&= a-y
\end {aligned}
$$

损失函数对$d_w$的偏导

$$
\begin {aligned}
d_{w_1} &= \frac{\partial l}{\partial z} \frac{\partial z}{\partial w_1} \\
&= (a-y) * \frac{\partial z}{\partial w_1}(w_1x_1+\cdots+w_nx_n+b) \\
&= x_1 * (a-y) \\

d_w &= \left[
    \begin{matrix}
    d_{w_1} \\
    \vdots \\
    d_{w_n}
    \end{matrix}
    \right] \\
&= 
\left[
    \begin{matrix}
    x_1 \\
    \vdots \\
    x_n
    \end{matrix}
\right] (a-y)
\end {aligned}
$$

损失函数对$d_b$的偏导

$$
\begin {aligned}
d_b &= \frac{\partial l}{\partial z} \frac{\partial z}{\partial b} \\
&= (a-y) * \frac{\partial z}{\partial b}(w_1x_1+\cdots+w_nx_n+b) \\
&= a-y
\end {aligned}
$$

根据导数对梯度进行更新的计算公式

$$
\begin {aligned}
w &= w - \alpha d_w \\
&= \left[
    \begin{matrix}
    w_1 \\
    \vdots \\
    w_n
    \end{matrix}
    \right] 
    - \alpha 
    \left[
    \begin{matrix}
    d_{w_1} \\
    \vdots \\
    d_{w_n}
    \end{matrix}
    \right] \\
b &= b - \alpha d_b
\end {aligned}
$$

#### m个样本

##### forward propagate

input X的shape为$(n_x, m)$，label Y的shape为$(1, m)$

$$
X = 
\left[
    \begin{matrix} 
    x^{(1)} & \cdots & x^{(m)}
    \end{matrix} 
\right] =
\left[
    \begin{matrix} 
    x^{(1)}_1 & \cdots & x^{(m)}_1\\ 
    \vdots & \ddots & \vdots \\ 
    x^{(1)}_n & \cdots & x^{(m)}_n
    \end{matrix} 
\right]
\quad
Y = \left[
    \begin{matrix}
    y^{(1)} & \cdots & y^{(m)}
    \end{matrix}
    \right]
$$

weights w的shape为$(n_x, 1)$，biases b的shape为$(1, 1)$ **和单个样本一样**

$$
w =
\left[
    \begin{matrix}
    w_1 \\
    \vdots \\
    w_n
    \end{matrix}
\right]
\quad
b =
\left[
    \begin{matrix}
    b
    \end{matrix}
\right]
$$

z的shape为$(1, m)$

$$
\begin {aligned}
Z &= w^TX+b \\
&=\left[
    \begin{matrix}
    w_1 &
    \cdots &
    w_n
    \end{matrix}
\right]
\left[
    \begin{matrix} 
    x^{(1)}_1 & \cdots & x^{(m)}_1\\ 
    \vdots & \ddots & \vdots \\ 
    x^{(1)}_n & \cdots & x^{(m)}_n
    \end{matrix} 
\right]
+
b \\
& =
\left[
    \begin{matrix} 
    z^{(1)} & \cdots & z^{(m)}
    \end{matrix} 
\right]
\end {aligned}
$$

output $\hat{Y}$的shape为$(1, m)$

$$
\begin {aligned}
\hat{Y} &= A\\
&= \sigma(Z) \\
&= \sigma(\left[
    \begin{matrix} 
    z^{(1)} & \cdots & z^{(m)}
    \end{matrix} 
\right]) \\
&= 
\left[
    \begin{matrix} 
    \frac{1}{1+e^{-z^{(1)}}} & \cdots & \frac{1}{1+e^{-z^{(m)}}}
    z^{(1)} & \cdots & z^{(m)}
    \end{matrix} 
\right] \\
&= \left[
    \begin{matrix} 
    a^{(1)} & \cdots & a^{(m)}
    \end{matrix} 
\right] \\
&= \left[
    \begin{matrix} 
    \hat{y}^{(1)} & \cdots & \hat{y}^{(m)}
    \end{matrix} 
\right]
\end {aligned}
$$

损失函数$J(w,b)$的shape为$(1, 1)$

$$
\begin {aligned}
J(w,b) &=
\frac{1}{m} \sum{m \atop i=1} l(\hat{y}^{(i)},y^{(i)}) \\
&= -\frac{1}{m} \sum{m \atop i=1} [y^{(i)}\log{a^{(i)}}+(1-y^{(i)})\log{(1-a^{(i)})}]
\end {aligned}
$$

##### backward propagate

损失函数对$d_Z$的偏导

$$
\begin {aligned}
d_{z^{(1)}} &= \frac{\partial J(w,b)}{\partial a^{(1)}} \frac{\partial a^{(1)}}{\partial z^{(1)}} \\
&= -\frac{1}{m} \frac{\partial l(a^{(1)}, y^{(1)})}{\partial a^{(1)}} \frac{\partial a^{(1)}}{\partial z^{(1)}} \\
&= -\frac{1}{m} d_{z^{(1)}} \\
&= a^{(1)}-y^{(1)} \\

d_Z &= 
\left[
\begin {matrix}
d_{z^{(1)}} & \cdots & d_{z^{(m)}}
\end {matrix}
\right] \\
&= 
\left[
\begin {matrix}
a^{(1)}-y^{(1)} & \cdots & a^{(m)}-y^{(m)}
\end {matrix}
\right]
\end {aligned}
$$

损失函数对$d_{w_1}$的偏导

$$
\begin {aligned}
\frac{\partial J(w,b)}{\partial w_1} 
&= \frac{1}{m} \sum{m \atop i=1} \frac{\partial l}{\partial w_1} l(\hat{y}^{(i)},y^{(i)}) \\
&= \frac{1}{m} \sum{m \atop i=1} d_{w_1^{(i)}} \\
&= \frac{1}{m} \sum{m \atop i=1} x_1^{(i)} * (a^{(i)}-y^{(i)})
\end {aligned}
$$

损失函数对$d_W$的偏导

$$
\begin {aligned}
\frac{\partial J(w,b)}{\partial W}  &= \left[
    \begin{matrix}
    \frac{\partial J(w,b)}{\partial w_1}  \\
    \vdots \\
    \frac{\partial J(w,b)}{\partial w_n} 
    \end{matrix}
    \right] \\
&= \frac{1}{m} 
\left[
\begin{matrix}
\sum{m \atop i=1} x_1^{(i)} * (a^{(i)}-y^{(i)}) \\
\vdots \\
\sum{m \atop i=1} x_n^{(i)} * (a^{(i)}-y^{(i)})
\end{matrix}
\right]
\\
&= \frac{1}{m} 
\left[
    \begin{matrix}
    x_1^{(1)} * (a^{(1)}-y^{(1)}) + \cdots + x_1^{(m)} * (a^{(m)}-y^{(m)}) \\
    \vdots \\
    x_n^{(1)} * (a^{(1)}-y^{(1)}) + \cdots + x_n^{(m)} * (a^{(m)}-y^{(m)})
    \end{matrix}
\right] \\
&= \frac{1}{m}
\left[
    \begin{matrix}
    x_1^{(1)} & \cdots & x_1^{(m)} \\
    \vdots \\
    x_n^{(1)} & \cdots & x_n^{(m)}
    \end{matrix}
\right]
\left[
    \begin{matrix}
    a^{(1)}-y^{(1)} \\
    \vdots \\
    a^{(m)}-y^{(m)}
    \end{matrix}
\right] \\
&= \frac{1}{m} X d_Z^T
\end {aligned}
$$

损失函数对$d_b$的偏导

$$
\begin {aligned}
d_b 
&= \frac{\partial J(w,b)}{\partial b} \\
&= \frac{\partial J(w,b)}{\partial Z} \frac{\partial Z}{\partial b} \\
&= \frac{1}{m} \sum{m \atop i=1} d_Z \\
&= \frac{1}{m} \sum{m \atop i=1} a^{(i)}-y^{(i)} \\
\end {aligned}
$$

根据导数对梯度进行更新的计算公式

$$
\begin {aligned}
W &= W - \alpha d_W \\
&= \left[
    \begin{matrix}
    w_1 \\
    \vdots \\
    w_n
    \end{matrix}
    \right] 
    - \alpha 
    \left[
    \begin{matrix}
    d_{w_1} \\
    \vdots \\
    d_{w_n}
    \end{matrix}
    \right] \\
b &= b - \alpha d_b
\end {aligned}
$$

## 第三周 浅层神经网络

### 公式推导

右上角`[]`表示层数，右上角`()`表示样本数

$a^{[0]}=X$ 第0层为输入层

$a^{[1]}_2$ 表示第一层中第二个神经元

$g(1)$为第一层网络的激活函数

#### forward propagate

输入X的shape为$(n^{[0]},m)$, Y的shape为$(1,m)$

$$
X = 
\left[
\begin {matrix}
x^{(1)}_1 & \cdots & x^{(m)}_1 \\
\vdots & \ddots & \vdots \\
x^{(1)}_n & \cdots & x^{(m)}_n
\end {matrix}
\right]

\quad

Y =
\left[
\begin {matrix}
y^{(1)} & \cdots & y^{(m)}
\end {matrix}
\right]
$$

权重矩阵$W^{[1]}$的shape为 $(n^{[1]}, n^{[0]})$, 偏置$b^{[1]}$的shape为 $(n^{[1]}, 1)$

$$
\begin {aligned}
W^{[1]} &=
\left[
\begin {matrix}
w_{1,1} & \cdots & w_{1,n^{[0]}} \\
\vdots & \ddots & \vdots \\
w_{n^{[1]},1} & \cdots & w_{n^{[1]},n^{[0]}}
\end {matrix}
\right] \\

b^{[1]} &=
\left[
\begin {matrix}
b_{1} \\
\vdots \\
b_{n^{[1]}}
\end {matrix}
\right]
\end {aligned}
$$

第一层神经元$Z^{[1]}$、$A^{[1]}$的shape为$(n^{[1]},m)$

$$
\begin {aligned}
Z^{[1]} 
&= W^{[1]} X + b^{[1]} \\
&= 
\left[
\begin {matrix}
w_{1,1} & \cdots & w_{1,n^{[0]}} \\
\vdots & \ddots & \vdots \\
w_{n^{[1]},1} & \cdots & w_{n^{[1]},n^{[0]}}
\end {matrix}
\right]
\left[
\begin {matrix}
x^{(1)}_1 & \cdots & x^{(m)}_1 \\
\vdots & \ddots & \vdots \\
x^{(1)}_n & \cdots & x^{(m)}_n
\end {matrix}
\right]
+
\left[
\begin {matrix}
b_{1} \\
\vdots \\
b_{n^{[1]}}
\end {matrix}
\right] \\
&= 
\left[
\begin {matrix}
z_{1}^{[1] (1)} & \cdots & z_{1}^{[1] (m)} \\
\vdots & \ddots & \vdots \\
z_{n^{[1]}}^{[1] (1)} & \cdots & z_{n^{[1]}}^{[1] (m)}
\end {matrix}
\right] \\

A^{[1]} &=
g(1)(Z^{[1]})
\end {aligned}
$$

权重矩阵$W^{[2]}$的shape为 $(n^{[2]}, n^{[1]})$, 偏置$b^{[2]}$的shape为 $(n^{[2]}, 1)$

$$
\begin {aligned}
W^{[2]} &=
\left[
\begin {matrix}
w_{1,1} & \cdots & w_{1,n^{[1]}} \\
\vdots & \ddots & \vdots \\
w_{n^{[2]},1} & \cdots & w_{n^{[2]},n^{[1]}}
\end {matrix}
\right] \\

b^{[2]} &=
\left[
\begin {matrix}
b_{1} \\
\vdots \\
b_{n^{[2]}}
\end {matrix}
\right]
\end {aligned}
$$

第二层神经元$Z^{[2]}$、$A^{[2]}$的shape为$(n^{[2]},m)$

$$
\begin {aligned}
Z^{[2]} 
&= W^{[2]} A^{[1]} + b^{[2]} \\
&= 
\left[
\begin {matrix}
z_{1}^{[2] (1)} & \cdots & z_{1}^{[2] (m)} \\
\vdots & \ddots & \vdots \\
z_{n^{[1]}}^{[2] (1)} & \cdots & z_{n^{[1]}}^{[2] (m)}
\end {matrix}
\right] \\

A^{[2]} &=
g(2)(Z^{[2]})
\end {aligned}
$$

损失函数为

$$
\begin {aligned}
J(W^{[1]},b^{[1]},W^{[2]},b^{[2]})
&= \frac{1}{m} \sum{m \atop i=1} l(\hat{y}^{(i)},y^{(i)}) \\
&= \frac{1}{m} \sum{m \atop i=1} l(a^{[2] (i)},y^{(i)})
\end {aligned}
$$

#### backward propagate

当$g(2)$为sigmoid函数时

$$
\begin {aligned}
d_{Z^{[2]}} &= A^{[2]} - Y ||shape(n^{[2]}, m)\\
d_{W^{[2]}} &= \frac{1}{m} d_{Z^{[2]}} A^{[1]T} ||shape(n^{[2]}, n^{[1]})\\
d_{b^{[2]}} &= \frac{1}{m} np.sum(d_{Z^{[2]}}, axis=1, keepdims=True) ||shape(n^{[2]}, 1)\\
d_{Z^{[1]}} &= W^{[2]T}d_{Z^{[2]}} * g^{[1]'}(Z^{[1]}) || shape(n^{[1]}, m)\\
d_{W^{[1]}} &= \frac{1}{m} d_{Z^{[1]}} X^T ||shape(n^{[1]}, n^{[0]}) \\
d_{b^{[1]}} &= \frac{1}{m} np.sum(d_{Z^{[1]}}, axis=1, keepdims=True) ||shape(n^{[1]}, 1) \\
\end {aligned}
$$

四种激活函数及其导数

$$a=g(z)=sigmoid(z) = \frac{1}{1+e^{-z}}$$
$$g'(z)=sigmoid(z)' = a(1-a)$$
$$a=g(z)=tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}$$
$$g'(z)=tanh(z)' = 1-a^2$$
$$a=g(z)=relu(z) = max(0, z)$$
$$g'(z)=relu(z)' = \begin{cases} 0, &z<0\cr 1, &z \geq 0 \end{cases}$$
$$a=g(z)=leakyRelu(z) = max(0.01z, z)$$
$$g'(z)=leakyRelu(z)' = \begin{cases} 0.01, &z<0\cr 1, &z \geq 0 \end{cases}$$

## 第四周 深层神经网络

### 公式

$$
\begin {aligned}
Z^{[l]} &= W^{[l]}A^{[l-1]} + b^{[l]} \\
A^{[l]} &= g^{[l]}(Z^{[l]})
\end {aligned}
$$

其中，$l$为层数，总层数为$L$，$l=0$表示输入层$X$，$l=L$表示输出层

$W^{[l]}$的shape为$(n^{[l]}, n^{[l-1]})$

$Z^{[l]}$ $A^{[l]}$的shape为$(n^{[l]}, m)$

$b^{[l]}$的shape为$(n^{[l]}, 1)$

$$
\begin {aligned}
\mathrm{d}Z^{[l]} &= \mathrm{d}A^{[l]} * g^{[l]'}(Z^{[l]}) \\
\mathrm{d}W^{[l]} &= \frac{1}{m} \mathrm{d}Z^{[l]} A^{[l-1]T} \\
\mathrm{d}b^{[l]} &= \frac{1}{m} np.sum(\mathrm{d}Z^{[l]}, axis=1, keepdims=True) \\
\mathrm{d}A^{[l-1]} &= W^{[l]T}\mathrm{d}Z^{[l]}
\end {aligned}
$$

# 改善深层神经网络：超参数调试、正则化以及优化

[网址](https://mooc.study.163.com/course/2001281003?tid=2001391036#/info)

## 第一周 深度学习的实用层面

### 训练/开发/测试集

对于100万以上数据 train 98% dev/valid 1% test 1%

### 偏差bias/方差variance

训练集上的高偏差?

加深网络、换网络模型

验证集上的高方差?

更多的数据、正则化

### 正则化-L2

L2正则化，$J(w,b)=\frac{1}{m} \sum{m \atop i=1}{l(a^{(i)},y^{(i)})}+\frac{\lambda}{2m}||w||^2$

$\lambda$表示正则化参数，python编程时用`lambd`表示。

$||w||^2$表示权重矩阵中所有权重值的平方和。

对上式子进行求导，会得到$\mathrm{d}W^{[l]}=(from backpropa)+\frac{\lambda}{m}W^{[l]}$

权重更新公式为$W^{[l]}=W^{[l]}-\mathrm{d}W^{[l]}=(1-\frac{a\lambda}{m})W^{[l]}-a(from backpropa)$

权重会不断的下降，所以也称之为权重衰减。weight decay

$\lambda$越大，$Z$越小，tanh或者sigmoid激活函数越接近于线性，整个神经网络会向线性方向发展，这样就会避免过拟合。

### 正则化-dropout

Inverted dropout

```python
d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep-prob
a3 = np.multiply(d3, a3)
a3 /= keep-prob
```

### 其他正则化方法

数据扩增，包含翻转、旋转、缩放、扭曲等。

early stopping，在中间点停止迭代过程。

### 输入归一化

将输入归一化为正太分布

$$
\begin {aligned}
\mu &= \frac{1}{m}\sum{m \atop i=1}x^{(i)} \\
x &= x - \mu \\
\sigma^2 &= \frac{1}{m}\sum{m \atop i=1}x^{(i)2} \\
x &= x / \sigma^2
\end {aligned}
$$

**使得代价函数更加圆滑，梯度更加合理**

### 梯度消失和梯度爆炸

vanishing/exploding gradients

W^10000，W<1，消失 >1，爆炸

### 权重初始化

[机器学习的模型（e.g. logistic regression, RBM）中为什么加入bias?](https://www.zhihu.com/question/24300697)

对于relu神经元，$W^{[l]}=np.random.randn(shape)*np.sqrt(\frac{a}{n^{[l-1]}})$

对于tanh神经元，会乘以$np.sqrt(\frac{1}{n^{[l-1]}})$或者$np.sqrt(\frac{2}{n^{[l-1]}+n^{[l]}})$，被称之为Xavierc初始化。

### 梯度检查

grad check

不要在训练中使用，仅仅debug

如果检查失败，检查bug

不要忘记正则化

不要使用dropout

### 防止过拟合的方法

1. 降低模型复杂度
1. 扩充样本，数据增强
    1. 随机裁剪
    1. 随机加光照
    1. 随机左右翻转
1. early stopping
1. dropout
1. weight penality L1&L2

### 数据集少怎么办

1. 图像平移
1. 图像旋转
1. 图像镜像
1. 图像亮度变化
1. 裁剪
1. 缩放
1. 图像模糊

## 第二周 优化算法

### Mini-batch

64 128 512 1024

不会稳定的想最小值发展，不会收敛

### 动量梯度下降

指数加权平均，滑动平均模型

$$
\begin {aligned}
v_{\mathrm{d}W} &= \beta v_{\mathrm{d}W} + (1-\beta)\mathrm{d}W \\
v_{\mathrm{d}b} &= \beta v_{\mathrm{d}b} + (1-\beta)\mathrm{d}b \\
W &= W - \alpha v_{\mathrm{d}W} \\
b &= b - \alpha v_{\mathrm{d}b} \\
\end {aligned}
$$

包含两个超参数：学习率$\alpha$和滑动衰减率$\beta$，$\beta$一般取0.9或者0.99，越多模型越稳定。

Momentum是为了对冲mini-batch带来的抖动。

### RMSprop

$$
\begin {aligned}
S_{\mathrm{d}W} &= \beta S_{\mathrm{d}W} + (1-\beta)(\mathrm{d}W)^2 \\
S_{\mathrm{d}b} &= \beta S_{\mathrm{d}b} + (1-\beta)(\mathrm{d}b)^2 \\
W &= W - \alpha \frac{\mathrm{d}W}{\sqrt{S_{\mathrm{d}W}}+\varepsilon} \\
b &= b - \alpha \frac{\mathrm{d}b}{\sqrt{S_{\mathrm{d}b}}+\varepsilon} \\
\end {aligned}
$

$\varepsilon$为了阻止除以极小值，一般取`e-8`

RMSprop是为了对hyper-parameter进行归一。直观理解是将摆动大的梯度进行缩小。

### Adam优化算法

Adaptive Moment Estimation 结合了动量和RMSprop

mini-batch中计算出每次迭代过程$t$的$\mathrm{d}W$和$\mathrm{d}b$后，Adam优化算法公式如下：

$$
\begin {aligned}
V_{\mathrm{d}W} = \beta_1 V_{\mathrm{d}W} + (1-\beta_1)\mathrm{d}W
&\quad
V_{\mathrm{d}b} = \beta_1 V_{\mathrm{d}b} + (1-\beta_1)\mathrm{d}b \\
S_{\mathrm{d}W} = \beta_2 S_{\mathrm{d}W} + (1-\beta_2)(\mathrm{d}W)^2
&\quad
S_{\mathrm{d}b} = \beta_2 S_{\mathrm{d}b} + (1-\beta_2)(\mathrm{d}b)^2 \\
V^{corrected}_{\mathrm{d}W} = \frac{V_{\mathrm{d}W}}{1-{\beta_1}^t} 
&\quad
V^{corrected}_{\mathrm{d}b} = \frac{V_{\mathrm{d}b}}{1-{\beta_1}^t} \\
S^{corrected}_{\mathrm{d}W} = \frac{S_{\mathrm{d}W}}{1-{\beta_2}^t} 
&\quad
S^{corrected}_{\mathrm{d}b} = \frac{S_{\mathrm{d}b}}{1-{\beta_2}^t} \\
W = W - \alpha \frac{V^{corrected}_{\mathrm{d}W}}{\sqrt{S^{corrected}_{\mathrm{d}W}}+\varepsilon}
&\quad
b = b - \alpha \frac{V^{corrected}_{\mathrm{d}b}}{\sqrt{S^{corrected}_{\mathrm{d}b}}+\varepsilon} \\
\end {aligned}
$$

第三行和第四行公式为偏差修正

- $\alpha$:全局学习率
- $\beta_1$:默认0.9
- $\beta_2$:默认0.999
- $\varepsilon$:默认$10^{-8}$

### 学习率衰减

学习率随着时间而慢慢变小，初始学习率，衰减率

### 局部最优问题

鞍点saddle point----损失函数中的0梯度点

平滑段使得训练变慢

## 第三周 超参数调试、Batch正则化和程序框架

### 超参

按重要程度排名

- 学习率$\alpha$——最重要
    - 对对数轴上均匀取值$[10^a,10^b]$,比如a,b的取值为[-4,-1]
- 隐藏层神经元个数`#hidden units`——第二重要
- mini-batch size——第二重要
- moment beta——第二重要
    - 0.9意味着取过去10个数字的平均值，0.999以为着取过去1000个数字的平均值
    - 针对$1-\beta$对对数轴上均匀取值$[10^a,10^b]$,比如a,b的取值为[-3,-1]
- 隐藏层数——第三重要
- 学习率衰减值——第三重要
- Adam算法中的$\beta_1,\beta_2,\varepsilon$，一般取默认值——不重要

超参搜索

- 随机选择超参组合
    - 在各个参数的合理范围内随机取值
    - 有助于发现潜在的最优值
- 由粗到精的搜索

### Batch归一化

将每一层的Z[l]归一化，在激活之前。可以加快训练速度

$$
\begin {aligned}
\mu &= \frac{1}{m}\sum{z^{(i)}} \\
\sigma^2 &= \frac{1}{m}\sum{(z^{(i)}-\mu)^2} \\
z^{(i)}_{norm} &=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}} \\
\tilde{z}^{(i)} &= \gamma z^{(i)}_{norm} + \beta
\end {aligned}
$$

### Softmax层

激活函数公式

$$
\begin {aligned}
Z^{[l]} &= W^{[l]}A^{[l-1]}+b^{[l]} \\
T^{[l]} &= e^{Z^{[l]}} \\
A^{[l]} &= \frac{T^{[l]}}{\sum{T^{[l]}}}
\end {aligned}
$$

Softmax相比于Hardmax

Softmax: $[0.1, 0.2, 0.7]^T$，Hardmax: $[0, 0, 1]^T$，温和的给出概率，而不是直接定死

损失函数定义：

$$
l(a,y) = - \sum^{n^L}_{j=1}y_j\log{a_j}
$$

对损失函数的导数如下：

$$
\frac{\partial J(W,b)}{\partial {Z^L}} = \hat{Y} - Y
$$

# 结构化机器学习项目

## 机器学习(ML)策略1

### 正交化 orthogonalization

- Fit training set well on cost function
    - 按钮1：bigger network
    - 按钮2：不同的优化算法
    - ......
- Fit dev set well on cost function
    - 按钮1：正则化
    - 按钮2：更大的训练集
- Fit test set well on cost function
    - 按钮1：更大的验证集
- Performs well in real world
    - 更改验证集
    - 更改损失函数

不推荐使用early stopping，因为这会同时影响训练和验证过程，不是非常正交。

### 单一数字评估指标

在评估分类器时，使用查准率Precision和查全率Recall是比较合理的。

| |真实值为1|真实值为0|
|---|---|---|
|预测值为1|真阳性True Positive|假阳性False Positive
|预测值为0|假阴性False Negative|真阴性True Negative

准确率，查准率，Precision （检索出的相关信息量/检索出的信息总量）x100%  

$$
\begin {aligned}
Precison(\%)
&= \frac{TruePositive}{NumberOfPredictedPositive} \times 100\% \\
&= \frac{TruePositive}{TruePositive+FalsePositive} \times 100\% \\
\end {aligned}
$$

召回率，查全率，Recall （检索出的相关信息量/系统中的相关信息总量）x100%

$$
\begin {aligned}
Recall(\%)
&= \frac{TruePositive}{NumberOfPredictedActuallyPositive} \times 100\% \\
&= \frac{TruePositive}{TruePositive+TrueNegative} \times 100\% \\
\end {aligned}
$$

在机器学习中，不推荐使用两个指标来衡量，此时最好使用F1 score，公式如下，理解为PR的调和平均值，也大致理解为PR的算术平均数。

$$
s = \frac{2}{\frac{1}{P}+\frac{1}{R}}
$$

在众多的指标中指定单一实数指标，比如平均值，最值等。

### 满足指标和优化指标

优化指标Optimizing metric——准确度，一个

满足指标Satisficing metric——运行时间（阈值），多个

### 训练/开发/测试数据

开发集和评估指标，决定了靶心。

开发集和测试集需要是一个分布

### 调整靶心

通过权重修改损失函数来调节错误率。

### 可避免偏差

训练错误和贝叶斯（人类表现）的差距叫可避免偏差

训练错误和开发错误的差距叫做方差

### 人类水平表现 human-level performance

贝叶斯误差的替代品 

## 机器学习(ML)策略2

### 进行误差分析

比如一个猫分类器，手动检查测试集中的错误例子，若在100个中，只有5个狗被识别成为了猫，则不值得处理狗的问题。若有50个狗被识别错误，则值得处理。

这称之为性能上限。

需要分析被错误识别的原因，包括种类错误、模糊、滤镜等。然后决定接下来去解决哪个方面的误差分析。

### 清除标注错误的数据

深度学习算法本身对随机的错误训练样本有一定的鲁棒性。

关注于最大的错误的一边。

### 快速搭建你的第一个系统，并进行迭代

1. 快速设立训练集、测试集和指标，这样可以决定你的目标所在。
1. 快速搭建深度学习模型，然后进行训练，观察结果如何。
1. 分析偏差方差，分析误差。决定决定做什么和下一步优先做什么。

# 卷积神经网络

## 第一周 卷积神经网络

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

### Classic networks 经典网络

#### LeNet-5

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

#### AlexNet

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

#### VGG-16

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

### 残差网络 ResNet(152 layers)

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

残差网络的优势：

1. 深度网络不是越深越好，网络越深，性能反而越差，这是因为梯度涣散问题。残差网络解决了这一问题。
1. 残差网络可以认为是浅层网络和深层网络的结合体，哪个生效用哪个。

### 1x1卷积

在每个像素上的深度上的全连接运算。可以用来改变通道深度，或者对每个像素分别添加了非线性变换。

Network in Network

### Inception

一个Inception模块，帮你解决使用什么尺寸的卷积层和何时使用池化层。

![img](http://upload-images.jianshu.io/upload_images/5952841-ab356377d7ef393d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了解决计算成本问题，引入1x1卷积进行优化计算。

![img](http://upload-images.jianshu.io/upload_images/5952841-8c7a2c347246c03d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

事实证明，只要合理构建瓶颈层，不仅不会降低网络性能，还会降低计算成本。

具体模块

![img](http://upload-images.jianshu.io/upload_images/5952841-a17e711a97eae00d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

具体网络

![img](http://upload-images.jianshu.io/upload_images/5952841-a3afa54b3f16a255.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 迁移学习

冻结一部分网络，自己训练一部分网络，并替换输出层的softmax

![img](http://upload-images.jianshu.io/upload_images/5952841-a1fb8b637b7939ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 数据增强

- 常用操作
    - 镜像操作
    - 随机修剪、裁剪
- 颜色偏移
    - 颜色通道分别加减值，改变RGB
    - PCA颜色增强算法

从硬盘中读取数据并且进行数据增强可以在CPU的线程中实现，并且可以与训练过程并行化。

## 第三周 目标检测

图像分类（图像中只有一个目标）->

目标定位（图像中只有一个目标）->

目标检测（图像中多个目标）

### 目标定位

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

### 特征点检测

若想输出人脸中的眼角特征点的位置，则在神经网络的输出中添加4个数值即可。

比如人脸中包含64个特征点，则神经网络的输出层中添加64x2个输出。

### 滑动窗口目标检测

首先需要训练裁剪过后的小图片。

然后针对输入的大图片，利用滑动窗口的技术对每个窗口进行检测。

将窗口放大，再次遍历整个图像。

将窗口再放大，再次遍历整个图像。

滑动窗口技术计算成本过高，

### CNN中的滑动窗口

将网络中的FC转化为卷积层，实际效果一样。

![img](http://upload-images.jianshu.io/upload_images/5952841-163c6ca8ca8b5222.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

整个大图像做卷积运算。

![img](http://upload-images.jianshu.io/upload_images/5952841-82d270859df8497c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 边界框预测

YOLO算法（You Only Look Once）

将整个大图像划分为3x3、19x19这样的格子，然后修改Label Y，每个小格子中，若目标对象的中心点位于该格内，则该格Label Y中的$P_c$为1。相邻格子就算包含了目标对象的一部分，$P_c$也为0

### 交并比

评价目标定位的指标

Intersection over Union(IoU)

交集面积/并集面积

一般认为，如果IoU >= 0.5，则认为是正确

### 非最大值抑制

选定一份概率最大的矩形，然后抑制（减小其概率）与之交并比比较高的矩形。

### Anchor Boxes

一个格子检测多个目标

### YOLO算法

1. 将图像划分为3x3=9个格子，然后每个格子中包含2个anchor box，那么输出Y的维度为3x3x2x8。即$y=[p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3,p_c,b_x,b_y,b_h,b_w,c_1,c_2,c_3]^T$。对于没有目标的格子，输出为$y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]^T$，对于有一个车辆的格子，输出为$y=[0,0,0,0,0,0,0,0,1,b_x,b_y,b_h,b_w,0,1,0]^T$
1. 对于9个格子中的每一个，都会输出2个预测框。
1. 去掉预测值低的框。
1. 对于每一个类别（行人、车辆、摩托），运行非最大值抑制去获得最终预测结果。

### RPN网络

预先进行Region proposal 候选区域提取

[RCNN,Fast RCNN,Faster RCNN 总结](http://shartoo.github.io/RCNN-series/ "RCNN,Fast RCNN,Faster RCNN 总结")

#### R-CNN 

[RCNN算法详解](http://blog.csdn.net/shenxiaolu1984/article/details/51066975)

相比传统算法（HOG+SVM），RCNN（区域CNN）优势如下：

1. 速度。经典的目标检测算法使用滑动窗法依次判断所有可能的区域。RCNN则预先提取一系列较可能是物体的候选区域，之后仅在这些候选区域上提取特征，进行判断。
1. 特征提取。经典的目标检测算法在区域中提取人工设定的特征（Haar，HOG）。RCNN则需要训练深度网络进行特征提取。

![R-CNN ](http://upload-images.jianshu.io/upload_images/5952841-11077d2fb07d9b86.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

不再使用滑动窗口卷积，而是选择一些候选区域进行卷积，使用图像分割（Segmentation）算法选出候选区域。对区域进行卷积分类比较缓慢。算法不断优化。

1. 使用候选区域提取算法selective search大约提取2000个矩形框。（是一种分割后合并的算法）
1. 对这2000个矩形框与groud truth进行IOU比较，大于阈值则认为是目标区域。
1. 将这2000个矩形进行缩放到CNN的输入大小227*227
1. 将2000个图像分别通过CNN提取特征。
1. 利用SVM进行特征分类。

#### Fast R-CNN

[Fast RCNN算法详解](http://blog.csdn.net/shenxiaolu1984/article/details/51036677)

相比RCNN，Fast RCNN有如下优化：

1. 速度优化。RCNN对2000个区域分别做CNN特征提取，而这些区域很多都是重叠的，所有包含大量的重复计算。Fast R-CNN将整个图像归一化后传入深度网络，消除了重复计算。
1. 空间缩小。RCNN中独立的分类器和回归器需要大量特征作为训练样本。 Fast R-CNN把类别判断和位置精调统一用深度网络实现（softmax和regressor），不再需要额外存储。

![Fast RCNN](http://upload-images.jianshu.io/upload_images/5952841-59cd6ebdeab287d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

算法具体要点：

1. Conv feature map。将整个图像放入深度网络后，得到了总的feature map（图中中间的Deep ConvNet箭头所指的大map），然后结合2000个候选区域，得到了2000个子feature map（图中中间的RoI projection所指的灰色部分，是大map中的一部分）。
1. ROI pooling。 因为这些feature后续需要通过全连接层，所以需要尺寸一致，所以需要将不同大小的feature map归一化到相同的大小。具体是先分割区域，然后max pooling。
1. ROI feature vector。每个候选区域经过ROI pooling layer和2个FC后，得到了大小相同的 feature vector，这些vector分成2部分，一个进行全连接之后用来做softmax回归，用来进行分类，另一个经过全连接之后用来做bbox回归。

#### Faster R-CNN

从RCNN到Fast RCNN，再到本文的Faster RCNN，目标检测的四个基本步骤（候选区域生成，特征提取，分类，位置精修）终于被统一到一个深度网络框架之内。所有计算没有重复，完全在GPU中完成，大大提高了运行速度。 

![image.png](http://upload-images.jianshu.io/upload_images/5952841-d050f513da3db453.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Faster R-CNN可以简单认为是RPN+Fast RCNN

![Faster R-CNN](http://upload-images.jianshu.io/upload_images/5952841-2039458e798cb3d3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

优点如下：

1. 使用RPN网络代替Faster RCNN和RCNN中的区域提取算法selective search。
1. FASTER-RCNN创造性地采用卷积网络自行产生建议框，并且和目标检测网络共享卷积网络，使得建议框数目从原有的约2000个减少为300个，且建议框的质量也有本质的提高.

## 第四周 特殊应用：人脸识别和神经风格转变

### 术语

人脸检测face recognition和活体检测liveness detection

人脸验证face verification，1:1问题，验证name和face是否一一对应

人脸识别face recognition，1:k问题，一个人在不在这个库中。多次运行人脸验证。

### one-shot learning

需要用这个人的一张照片去识别这个人，样本只有一个。

一种方法是将100个员工的人脸照片当作训练集，然后输出softmax 100个分类，但是这样识别效果并不好，且每加入一个新员工，都需要重新训练。

正确的方法是让深度学习网络学习一个相似函数similarity function，输入为2幅图像，输出为2幅图像之间的差异值。

### Siamese Network

假设一个图像x1,通过一个卷积网络，得到了一个128维的向量$a^{[l]}$，不需要把$a^{[l]}$通过softmax，而是将这128维向量作为该图像的编码，称之为$f(x_1)$。

比较2幅图像的编码，判断他们的差异值。$d(x_1,x_2)=||f(x_1)-f(x_2)||^2_2$，差异小表示为同一个人，差异大为不同的人。

这样的网络称之为**Siamese Network Architecture**

### Triplet 损失

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

### 面部验证与二分类

输入为2幅图像， 输出为0或者1。同样使用上一节的编码。
