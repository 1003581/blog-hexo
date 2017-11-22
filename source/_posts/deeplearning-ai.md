---
title: deeplearning.ai
date: 2017-11-06 23:30:00
tags: 深度学习
categories: 机器学习
---

课程在[网易云课堂](https://study.163.com/provider/2001053000/index.htm)上免费观看，作业题如下：加粗为答案。

<!-- more -->

# 神经网络和深度学习

[网址](https://mooc.study.163.com/course/2001281002?tid=2001392029#/info)

## 第一周 深度学习概论

10个选择题，原见[Github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/Quiz-week1-Introduction%20to%20deep%20learning.pdf)

1. What does the analogy “AI is the new electricity” refer to?
    1. **Similar to electricity starting about 100 years ago, AI is transforming multiple industries.**
    1. Through the “smart grid”, AI is delivering a new wave of electricity.
    1. AI runs on computers and is thus powered by electricity, but it is letting computers do things not possible before.
    1. AI is powering personal devices in our homes and offices, similar to electricity.
1. Which of these are reasons for Deep Learning recently taking off? (Check the three options that apply.)
    1. Deep learning has resulted in significant improvements in important applications such as online advertising, speech recognition, and image recognition.
    1. **We have access to a lot more data.**
    1. **We have access to a lot more computational power.**
    1. Neural Networks are a brand new field.
1. Recall this diagram of iterating over different ML ideas. Which of the statements below are true? (Check all that apply.)   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212040.png)
    1. **Being able to try out ideas quickly allows deep learning engineers to iterate more quickly.**
    1. **Faster computation can help speed up how long a team takes to iterate to a good idea.**
    1. It is faster to train on a big dataset than a small dataset.
    1. **Recent progress in deep learning algorithms has allowed us to train good models faster (even without changing the CPU/GPU hardware).**
1. When an experienced deep learning engineer works on a new problem, they can usually use insight from previous problems to train a good model on the first try, without needing to iterate multiple times through different models. True/False?
    1. True
    1. **False**
1. Which one of these plots represents a ReLU activation function?
    1. Figure 1:   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212525.png)
    1. Figure 2:   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212535.png)
    1. **Figure 3:**   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212541.png)
    1. Figure 4:   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212548.png)
1. Images for cat recognition is an example of “structured” data, because it is represented as a structured array in a computer. True/False?
    1. True
    1. **False**
1. A demographic dataset with statistics on different cities' population, GDP per capita, economic growth is an example of “unstructured” data because it contains data coming from different sources. True/False?
    1. True
    1. **False**
1. Why is an RNN (Recurrent Neural Network) used for machine translation, say translating English to French? (Check all that apply.)
    1. **It can be trained as a supervised learning problem.**
    1. It is strictly more powerful than a Convolutional Neural Network (CNN).
    1. **It is applicable when the input/output is a sequence (e.g., a sequence of words).**
    1. RNNs represent the recurrent process of Idea->Code->Experiment->Idea->....
1. In this diagram which we hand-drew in lecture, what do the horizontal axis (x-axis) and vertical axis (y-axis) represent?  
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212556.png)
    1.  - x-axis is the performance of the algorithm
        - y-axis (vertical axis) is the amount of data.
    1.  - x-axis is the input to the algorithm
        - y-axis is outputs.
    1.  - x-axis is the amount of data
        - y-axis is the size of the model you train.
    1.  - **x-axis is the amount of data**
        - **y-axis (vertical axis) is the performance of the algorithm.**
1. Assuming the trends described in the previous question's figure are accurate (and hoping you got the axis labels right), which of the following are true? (Check all that apply.)
    1. Decreasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.
    1. Decreasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.
    1. **Increasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.**
    1. **Increasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.**

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

### Neural-Network-Basics

10个选择题，原见[Github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/Quiz-week2-Coursera%20_%20Online%20Courses%20From%20Top%20Universities.pdf)

1. What does a neuron compute?
    1. A neuron computes an activation function followed by a linear function (z = Wx + b)
    1. **A neuron computes a linear function (z = Wx + b) followed by an activation function**
    1. A neuron computes a function g that scales the input x linearly (Wx + b)
    1. A neuron computes the mean of all features before applying the output to an activation function
1. Which of these is the "Logistic Loss"?
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = -(y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)}))$ **True**
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = |y^{(i)} - \hat{y}^{(i)}|$
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = \max(0, y^{(i)} - \hat{y}^{(i)})$
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = |y^{(i)} - \hat{y}^{(i)}|^2$
1. Suppose img is a (32,32,3) array, representing a 32x32 image with 3 color channels red, green and blue. How do you reshape this into a column vector?
    1. **`x = img.reshape((32*32*3,1))`**
    1. `x = img.reshape((32*32,3))`
    1. `x = img.reshape((1,32*32,*3))`
    1. `x = img.reshape((3,32*32))`
1. Consider the two following random arrays "a" and "b":
    ```python
    a = np.random.randn(2, 3) # a.shape = (2, 3)
    b = np.random.randn(2, 1) # b.shape = (2, 1)
    c = a + b
    ```
    What will be the shape of "c"?
    1. c.shape = (2, 1)
    1. c.shape = (3, 2)
    1. **c.shape = (2, 3)**
    1. The computation cannot happen because the sizes don't match. It's going to be "Error"!
1. Consider the two following random arrays "a" and "b":
    ```python
    a = np.random.rand(4, 3) # a.shape = (4, 3)
    b = np.random.rand(3, 2) # a.shape = (3, 2)
    c = a*b
    ```
    What will be the shape of "c"?
    1. c.shape = (4, 2)
    1. c.shape = (4, 3)
    1. **The computation cannot happen because the sizes don't match. It's going to be "Error"!**
    1. c.shape = (3, 3)
1. Suppose you have $n_x$ input features per example. Recall that $X=[x^{(2)}x^{(m)}...x^{(1)}]$. What is the dimension of X?
    1. (m, 1)
    1. **($n_x$, m)**
    1. (m, $n_x$)
    1. (1, m)
1. Recall that "np.dot(a,b)" performs a matrix multiplication on a and b, whereas "a*b" performs an element-wise multiplication.  
    Consider the two following random arrays "a" and "b":
    ```python
    a = np.random.randn(12288, 150) # a.shape = (12288, 150)
    b = np.random.randn(150, 45) # b.shape = (150, 45)
    c = np.dot(a, b)
    ```
    What is the shape of c?
    1. The computation cannot happen because the sizes don't match. It's going to be "Error"!
    1. c.shape = (12288, 150)
    1. **c.shape = (12288, 45)**
    1. c.shape = (150,150)
1. Consider the following code snippet:
    ```python
    # a.shape = (3, 4)
    # b.shape = (4, 1)

    for i in range(3):
        for j in range(4):
            c[i][j] = a[i][j] + b[j]
    ```
    How do you vectorize this?
    1. c = a.T + b
    1. c = a.T + b.T
    1. **c = a + b.T**
    1. c = a + b
1. Consider the following code:
    ```python
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 1)
    c = a * b
    ```
    What will be c? (If you’re not sure, feel free to run this in python to find out).
    1. **This will invoke broadcasting, so b is copied three times to become (3,3), and ∗ is an element-wise product so c.shape will be (3, 3)**
    1. This will invoke broadcasting, so b is copied three times to become (3, 3), and ∗ invokes a matrix multiplication operation of two 3x3 matrices so c.shape will be (3, 3)
    1. This will multiply a 3x3 matrix a with a 3x1 vector, thus resulting in a 3x1 vector. That is, c.shape = (3,1).
    1. It will lead to an error since you cannot use “*” to operate on these two matrices. You need to instead use np.dot(a,b)
1. Consider the following computation graph.  
    ![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_094905.png)  
    What is the output J?
    1. `J = (c - 1)*(b + a)`
    1. **`J = (a - 1) * (b + c)`**
    1. `J = a*b + b*c + a*c`
    1. `J = (b - 1) * (c + a)`

### Logistic-Regression-with-a-Neural-Network-mindset

相关数据集和输出见[github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week2/Logistic%20Regression%20as%20a%20Neural%20Network/my-Logistic%2BRegression%2Bwith%2Ba%2BNeural%2BNetwork%2Bmindset%2Bv3.ipynb)

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

### Shallow Neural Networks

10个选择题，原见[Github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/Quiz-week3-Coursera%20_%20Online%20Courses%20From%20Top%20Universities.pdf)

1. Which of the following are true? (Check all that apply.)
    1. **$a^{[2]}$ denotes the activation vector of the $2^{nd}$ layer.**
    1. **$a^{[2] (12)}$ denotes the activation vector of the $2^{nd}$ layer for the $12^{th}$ training example.**
    1. $X$ is a matrix in which each row is one training example.
    1. $a^{[2] (12)}$ denotes activation vector of the $12^{th}$ layer on the $2^{nd}$ training example.
    1. **$X$ is a matrix in which each column is one training example.**
    1. **$a^{[2]}_4$ is the activation output by the $4^{th}$ neuron of the $2^{nd}$ layer**
    1. $a^{[2]}_4$ is the activation output of the $2^{nd}$ layer for the $4^{th}$ training example
1. The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False?
    1. **True**
    1. False
1. Which of these is a correct vectorized implementation of forward propagation for layer $l$, where $1\leq{}l\leq{}L$?
    1.  - $Z^{[l]}=W^{[l]}A^{[l]}+b^{[l]}$
        - $A^{[l+1]}=g^{[l+1]}(Z^{[l]})$
    1.  - $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$ **True**
        - $A^{[l]}=g^{[l]}(Z^{[l]})$ **True**
    1.  - $Z^{[l]}=W^{[l]}A^{[l]}+b^{[l]}$
        - $A^{[l+1]}=g^{[l]}(Z^{[l]})$
    1.  - $Z^{[l]}=W^{[l-1]}A^{[l]}+b^{[l-1]}$
        - $A^{[l]}=g^{[l]}(Z^{[l]})$
1. You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer?
    1. ReLU
    1. Leaky ReLU
    1. **sigmoid**
    1. tanh
1. Consider the following code:
    ```python
    A = np.random.randn(4, 3)
    B = np.sum(A, axis = 1, keepdims = True)
    ```
    What will be B.shape? (If you’re not sure, feel free to run this in python to find out).
    1. **(4, 1)**
    1. (4, )
    1. (1, 3)
    1. (, 3)
1. Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements is true?
    1. **Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons.**
    1. Each neuron in the first hidden layer will perform the same computation in the first iteration. But after one iteration of gradient descent they will learn to compute different things because we have “broken symmetry”.
    1. Each neuron in the first hidden layer will compute the same thing, but neurons in different layers will compute different things, thus we have accomplished “symmetry breaking” as described in lecture.
    1. The first hidden layer’s neurons will perform different computations from each other even in the first iteration; their parameters will thus keep evolving in their own way.
1. Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?
    1. True
    1. **False**
1. You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen?
    1. **This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow.**
    1. This will cause the inputs of the tanh to also be very large, causing the units to be “highly activated” and thus speed up learning compared to if the weights had to start from small values.
    1. It doesn’t matter. So long as you initialize the weights randomly gradient descent is not affected by whether the weights are large or small.
    1. This will cause the inputs of the tanh to also be very large, thus causing gradients to also become large. You therefore have to set α to be very small to prevent divergence; this will slow down learning.
1. Consider the following 1 hidden layer neural network:  
    ![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_095200.png)
    Which of the following statements are True? (Check all that apply).
    1. $W^{[1]}$ will have shape (2, 4)
    1. **$b^{[1]}$ will have shape (4, 1)**
    1. **$W^{[1]}$ will have shape (4, 2)**
    1. $b^{[1]}$ will have shape (2, 1)
    1. **$W^{[2]}$ will have shape (1, 4)**
    1. $b^{[2]}$ will have shape (4, 1)
    1. $W^{[2]}$ will have shape (4, 1)
    1. **$b^{[2]}$ will have shape (1, 1)**
1. In the same network as the previous question, what are the dimensions of $Z^{[1]}$ and $A^{[1]}$?
    1. $Z^{[1]}$ and $A^{[1]}$ are (1, 4)
    1. $Z^{[1]}$ and $A^{[1]}$ are (4, 2)
    1. **$Z^{[1]}$ and $A^{[1]}$ are (4, m)**
    1. $Z^{[1]}$ and $A^{[1]}$ are (4, 1)
    
### Planar data classification with one hidden layer

相关数据集和输出见[github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week3/my-Planar%2Bdata%2Bclassification%2Bwith%2Bone%2Bhidden%2Blayer.ipynb)

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

### Key concepts on Deep Neural Network

1. What is the "cache" used for in our implementation of forward propagation and backward propagation?
    1. **We use it to pass variables computed during forward propagation to the corresponding backward propagation step. It contains useful values for backward propagation to compute derivatives.**
    1. We use it to pass variables computed during backward propagation to the corresponding forward propagation step. It contains useful values for forward propagation to compute activations.
    1. It is used to cache the intermediate values of the cost function during training.
    1. It is used to keep track of the hyperparameters that we are searching over, to speed up computation.
1. Among the following, which ones are "hyperparameters"? (Check all that apply.)
    1. **size of the hidden layers $n^{[l]}$**
    1. **number of layers $L$ in the neural network**
    1. **learning rate $α$**
    1. activation values $a^{[l]}$
    1. **number of iterations**
    1. weight matrices $W^{[l]}$
    1. bias vectors $b^{[l]}$
1. Which of the following statements is true?
    1. **The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.**
    1. The earlier layers of a neural network are typically computing more complex features of the input than the deeper layers.
1. Vectorization allows you to compute forward propagation in an $L$-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False?
    1. **True**
    1. False
1. Assume we store the values for $n^{[l]}$ in an array called layers, as follows: layer_dims = [$n_x$, 4,3,2,1]. So layer 1 has four hidden units, layer 2 has 3 hidden units and so on. Which of the following for-loops will allow you to initialize the parameters for the model?
    1. Code1
        ```python
        for (i in range(1, len(layer_dims)/2)):
            parameter['w' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i], 1) *　0.01
        ```
    1. Code2 
        ```python
        for (i in range(1, len(layer_dims)/2)):
            parameter['w' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i-1], 1) *　0.01
        ```
    1. Code3 
        ```python
        for (i in range(1, len(layer_dims))):
            parameter['w' + str(i)] = np.random.randn(layers[i-1], layers[i]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i], 1) *　0.01
        ```
    1. **Code4**
        ```python
        for (i in range(1, len(layer_dims))):
            parameter['w' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i], 1) *　0.01
        ```
1. Consider the following neural network.  
    ![](...)  
    How many layers does this network have?
    1. **The number of layers $L$ is 4. The number of hidden layers is 3.**
    1. The number of layers $L$ is 3. The number of hidden layers is 3.
    1. The number of layers $L$ is 4. The number of hidden layers is 4.
    1. The number of layers $L$ is 5. The number of hidden layers is 4.
1. During forward propagation, in the forward function for a layer $l$ you need to know what is the activation function in a layer (Sigmoid, tanh, ReLU, etc.). During backpropagation, the corresponding backward function also needs to know what is the activation function for layer $l$, since the gradient depends on it. True/False?
    1. **True**
    1. False
1. There are certain functions with the following properties:  
    (i) To compute the function using a shallow network circuit, you will need a large network (where we measure size by the number of logic gates in the network), but (ii) To compute it using a deep network circuit, you need only an exponentially smaller network. True/False?
    1. **True**
    1. False
1. Consider the following 2 hidden layer neural network:  
    ![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_095336.png)  
    Which of the following statements are True? (Check all that apply).
    1. **W[1] will have shape (4, 4)**
    1. **b[1] will have shape (4, 1)**
    1. W[1] will have shape (3, 4)
    1. b[1] will have shape (3, 1)
    1. **W[2] will have shape (3, 4)**
    1. b[2] will have shape (1, 1)
    1. W[2] will have shape (3, 1)
    1. **b[2] will have shape (3, 1)**
    1. W[3] will have shape (3, 1)
    1. **b[3] will have shape (1, 1)**
    1. **W[3] will have shape (1, 3)**
    1. b[3] will have shape (3, 1)
1. Whereas the previous question used a specific network, in the general case what is the dimension of W^{[l]}, the weight matrix associated with layer $l$?
    1. **W[l] has shape (n[l],n[l−1])**
    1. W[l] has shape (n[l+1],n[l])
    1. W[l] has shape (n[l],n[l+1])
    1. W[l] has shape (n[l−1],n[l])

### Building your Deep Neural Network - Step by Step

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week4/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/my-Building%2Byour%2BDeep%2BNeural%2BNetwork%2B-%2BStep%2Bby%2BStep.ipynb)

### Deep Neural Network - Application

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week4/Deep%20Neural%20Network%20Application%20Image%20Classification/my-Deep%2BNeural%2BNetwork%2B-%2BApplication.ipynb)

# 改善深层神经网络：超参数调试、正则化以及优化

[网址](https://mooc.study.163.com/course/2001281003?tid=2001391036#/info)

## 第一周 深度学习的实用层面

### 笔记

#### 训练/开发/测试集

对于100万以上数据 train 98% dev/valid 1% test 1%

#### 偏差bias/方差variance

训练集上的高偏差?

加深网络、换网络模型

验证集上的高方差?

更多的数据、正则化

#### 正则化-L2

L2正则化，$J(w,b)=\frac{1}{m} \sum{m \atop i=1}{l(a^{(i)},y^{(i)})}+\frac{\lambda}{2m}||w||^2$

$\lambda$表示正则化参数，python编程时用`lambd`表示。

$||w||^2$表示权重矩阵中所有权重值的平方和。

对上式子进行求导，会得到$\mathrm{d}W^{[l]}=(from backpropa)+\frac{\lambda}{m}W^{[l]}$

权重更新公式为$W^{[l]}=W^{[l]}-\mathrm{d}W^{[l]}=(1-\frac{a\lambda}{m})W^{[l]}-a(from backpropa)$

权重会不断的下降，所以也称之为权重衰减。weight decay

$\lambda$越大，$Z$越小，tanh或者sigmoid激活函数越接近于线性，整个神经网络会向线性方向发展，这样就会避免过拟合。

#### 正则化-dropout

Inverted dropout

```python
d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep-prob
a3 = np.multiply(d3, a3)
a3 /= keep-prob
```

#### 其他正则化方法

数据扩增，包含翻转、旋转、缩放、扭曲等。

early stopping，在中间点停止迭代过程。

#### 输入归一化

将输入归一化为正太分布

$$
\begin {aligned}
\mu &= \frac{1}{m}\sum{m \atop i=1}x^{(i)} \\
x &= x - \mu \\
\sigma^2 &= \frac{1}{m}\sum{m \atop i=1}x^{(i)2} \\
x &= x / \sigma^2
\end {aligned}
$$

使得代价函数更加圆滑，梯度更加合理

#### 梯度消失和梯度爆炸

vanishing/exploding gradients

W^10000，W<1，消失 >1，爆炸

#### 权重初始化

[机器学习的模型（e.g. logistic regression, RBM）中为什么加入bias?](https://www.zhihu.com/question/24300697)

对于relu神经元，$W^{[l]}=np.random.randn(shape)*np.sqrt(\frac{a}{n^{[l-1]}})$

对于tanh神经元，会乘以$np.sqrt(\frac{1}{n^{[l-1]}})$或者$np.sqrt(\frac{2}{n^{[l-1]}+n^{[l]}})$，被称之为Xavierc初始化。

#### 梯度检查

grad check

不要在训练中使用，仅仅debug

如果检查失败，检查bug

不要忘记正则化

不要使用dropout

### Practical aspects of deep learning

1. If you have 10,000,000 examples, how would you split the train/dev/test set?
    1. 60% train . 20% dev . 20% test
    1. **98% train . 1% dev . 1% test**
    1. 33% train . 33% dev . 33% test
1. The dev and test set should:
    1. **Come from the same distribution**
    1. Come from different distributions
    1. Be identical to each other (same (x,y) pairs)
    1. Have the same number of examples
1. If your Neural Network model seems to have high variance, what of the following would be promising things to try?
    1. **Add regularization**        
    1. Make the Neural Network deeper
    1. Increase the number of units in each hidden layer
    1. Get more test data
    1. **Get more training data**
1. You are working on an automated check-out kiosk for a supermarket, and are building a classifier for apples, bananas, and oranges. Suppose your classifier obtains a training set error of 0.5%, and a dev set error of 7%, Which of the following are promising things to try to improve your classfier? (Check all that apply.)
    1. **Increase the regularization parameter lambda**
    1. Decrease the regularization parameter lambda
    1. **Get more training data**
    1. Use a bigger neural network
1. What is weight decay?
    1. Gradual corruption of the weights in the neural network if it is trained on noisy data.
    1. A technique to avoid vanishing gradient by imposing a ceiling on the values of the weights.
    1. **A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.**
    1. The process of gradually decreasing the learning rate during training.
1. What happens when you increase the regularization hyperparameter lambda?
    1. **Weights are pushed toward becoming smaller (closer to 0)**
    1. Weights are pushed toward becoming bigger (further from 0)
    1. Doubling lambda should roughly result in doubling the weights
    1. Gradient descent taking bigger steps with each iteration (proportional to lambda)
1. With the inverted dropout technique, at test time:
    1. You apply dropout (randomly eliminating units) and do not keep the 1/keep_prob factor in the calculations used in training
    1. You do not apply dropout (do not randomly eliminate units), but keep the 1/keep_prob factor in the calculations used in training.
    1. **You do not apply dropout (do not randomly eliminate units) and do not keep the 1/keep_prob factor in the calculations used in training**
    1. You apply dropout (randomly eliminating units) but keep the 1/keep_prob factor in the calculations used in training.
1. Increasing the parameter keep_prob from (say) 0.5 to 0.6 will likely cause the following: (Check the two that apply)
    1. Increasing the regularization effect
    1. **Reducing the regularization effect**
    1. Causing the neural network to end up with a higher training set error
    1. **Causing the neural network to end up with a lower training set error**
1. Which of these techniques are useful for reducing variance (reducing overfitting)? (Check all that apply.)
    1. **Data augmentation**
    1. Gradient Checking
    1. Exploding gradient
    1. **L2 regularization**
    1. **Dropout**
    1. Vanishing gradient
    1. Xavier initialization
1. Why do we normalize the inputs x?
    1. It makes it easier to visualize the data
    1. It makes the parameter initialization faster
    1. Normalization is another word for regularization--It helps to reduce variance
    1. **It makes the cost function faster to optimize**

### Regularization

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/2_Improving%20Deep%20Neural%20Networks/week1_Regularization/my-Regularization.ipynb)

### Initialization

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/2_Improving%20Deep%20Neural%20Networks/week1_initialization/my-Initialization.ipynb)

### Gradient_Checking

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/2_Improving%20Deep%20Neural%20Networks/week1_Gradient_Checking/my-Gradient%2BChecking.ipynb)

## 第二周 优化算法

### 理论

#### Mini-batch

64 128 512 1024

不会稳定的想最小值发展，不会收敛

#### 动量梯度下降

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

#### RMSprop

$$
\begin {aligned}
S_{\mathrm{d}W} &= \beta S_{\mathrm{d}W} + (1-\beta)(\mathrm{d}W)^2 \\
S_{\mathrm{d}b} &= \beta S_{\mathrm{d}b} + (1-\beta)(\mathrm{d}b)^2 \\
W &= W - \alpha \frac{\mathrm{d}W}{\sqrt{S_{\mathrm{d}W}}+\varepsilon} \\
b &= b - \alpha \frac{\mathrm{d}b}{\sqrt{S_{\mathrm{d}b}}+\varepsilon} \\
\end {aligned}
$$

$\varepsilon$为了阻止除以极小值，一般取`e-8`

RMSprop是为了对hyper-parameter进行归一。直观理解是将摆动大的梯度进行缩小。

#### Adam优化算法

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

#### 学习率衰减

学习率随着时间而慢慢变小，初始学习率，衰减率

#### 局部最优问题

鞍点saddle point----损失函数中的0梯度点

平滑段使得训练变慢

### Optimization algorithms

1. Which notation would you use to denote the 3rd layer’s activations when the input is the 7th example from the 8th minibatch?
    1. a[8]{3}(7)
    1. a[3]{7}(8)
    1. a[8]{7}(3)
    1. **a[3]{8}(7)**
1. Which of these statements about mini-batch gradient descent do you agree with?
    1. Training one epoch (one pass through the training set) using minibatch gradient descent is faster than training one epoch using batch gradient descent.
    1. **One iteration of mini-batch gradient descent (computing on a single mini-batch) is faster than one iteration of batch gradient descent.**
    1. You should implement mini-batch gradient descent without an explicit for-loop over different mini-batches, so that the algorithm processes all mini-batches at the same time(vectorization).
1. Why is the best mini-batch size usually not 1 and not m, but instead something in-between?
    1. **If the mini-batch size is m, you end up with batch gradient descent, which has to process the whole training set before making progress.**
    1. If the mini-batch size is m, you end up with stochastic gradient descent, which is usually slower than mini-batch gradient descent.
    1. **If the mini-batch size is 1, you lose the benefits of vectorization across examples in the mini-batch.**
    1. If the mini-batch size is 1, you end up having to process the entire training set before making any progress.
1. Suppose your learning algorithm’s cost , plotted as a function of the number of iterations, looks like this:  
    ![](http://outz1n6zr.bkt.clouddn.com/201711170922.PNG)  
    Which of the following do you agree with?
    1. Whether you’re using batch gradient descent or mini-batch gradient descent, this looks acceptable.
    1. **If you’re using mini-batch gradient descent, this looks acceptable. But if you’re using batch gradient descent, something is wrong.**
    1. Whether you’re using batch gradient descent or mini-batch gradient descent, something is wrong.
    1. If you’re using mini-batch gradient descent, something is wrong. But if you’re using batch gradient descent, this looks acceptable.
1. Suppose the temperature in Casablanca over the first three days of January are the same:  
    Jan 1st: $\theta_1 = 10^{\circ}C$  
    Jan 2nd: $\theta_2 = 10^{\circ}C$  
    (We used Fahrenheit in lecture, so will use Celsius here in honor of the metric world.)  
    Say you use an exponentially weighted average with $\beta=0.5$ to track the temperature: $v_0=0,v_t=\beta v_{t-1}+(1-\beta)\theta_t$. If $v_2$ is the value computed after day 2 without bias correction, and $v_2^{corrected}$ is the value you compute with bias correction. What are these values? (You might be able to do this without a calculator, but you don't actually need one. Remember what is bias correction doing.)
    1. $v_2=7.5,v_2^{corrected}=7.5$
    1. $v_2=10,v_2^{corrected}=7.5$
    1. $v_2=10,v_2^{corrected}=10$
    1. $v_2=7.5,v_2^{corrected}=10$ **True**
1. Which of these is NOT a good learning rate decay scheme? Here, t is the epoch number.
    1. $\alpha=\frac{1}{1+2*t}\alpha_0$
    1. $\alpha=e^t\alpha_0$ **True**
    1. $\alpha=0.95^t\alpha_0$
    1. $\alpha=\frac{1}{\sqrt{t}}\alpha_0$
1. You use an exponentially weighted average on the London temperature dataset. You use the following to track the temperature: $v_t=\beta v_{t-1}+(1-\beta)\theta_t$. The red line below was computed using $\beta=0.9$. What would happen to your red curve as you vary $\beta$? (Check the two that apply)  
    ![](http://outz1n6zr.bkt.clouddn.com/201711170944.PNG)
    1. Decreasing $\beta$ will shift the red line slightly to the right.
    1. **Increasing $\beta$ will shift the red line slightly to the right.**
    1. Decreasing $\beta$ will create more oscillation within the red line.
    1. Increasing $\beta$ will create more oscillations within the red line.
1. Consider this figure:  
    ![](http://outz1n6zr.bkt.clouddn.com/201711170947.PNG)  
    These plots were generated with gradient descent; with gradient descent with momentum ($\beta$ = 0.5) and gradient descent with momentum ($\beta$ = 0.9). Which curve corresponds to which algorithm?
    1. (1) is gradient descent with momentum (small $\beta$). (2) is gradient descent. (3) is gradient descent with momentum (large $\beta$)
    1. (1) is gradient descent. (2) is gradient descent with momentum (large $\beta$) . (3) is gradient descent with momentum (small $\beta$)
    1. **(1) is gradient descent. (2) is gradient descent with momentum (small $\beta$). (3) is gradient descent with momentum (large $\beta$)**
    1. (1) is gradient descent with momentum (small $\beta$), (2) is gradient descent with momentum (small $\beta$), (3) is gradient descent
1. Suppose batch gradient descent in a deep network is taking excessively long to find a value of the parameters that achieves a small value for the cost function $J(W^{[1]},b^{[1]},...,W^{[L]},b^{[L]})$. Which of the following techniques could help find parameter values that attain a small value for $J$? (Check all that apply)
    1. Try initializing all the weights to zero
    1. **Try tuning the learning rate $\alpha$**
    1. **Try better random initialization for the weights**
    1. **Try using Adam**
    1. **Try mini-batch gradient descent**
1. Which of the following statements about Adam is False?
    1. **Adam should be used with batch gradient computations, not with mini-batches.**
    1. We usually use “default” values for the hyperparameters and in Adam ($\beta_1=0.9$,$\beta_2=0.999$,$\varepsilon=10^{-8}$)
    1. The learning rate hyperparameter $\alpha$ in Adam usually needs to be tuned.
    1. Adam combines the advantages of RMSProp and momentum



## 第三周 超参数调试、Batch正则化和程序框架

### 笔记

#### 超参

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

#### Batch归一化

将每一层的Z[l]归一化，在激活之前。可以加快训练速度

$$
\begin {aligned}
\mu &= \frac{1}{m}\sum{z^{(i)}} \\
\sigma^2 &= \frac{1}{m}\sum{(z^{(i)}-\mu)^2} \\
z^{(i)}_{norm} &=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}} \\
\tilde{z}^{(i)} &= \gamma z^{(i)}_{norm} + \beta
\end {aligned}
$$

#### Softmax层

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

### Hyperparameter tuning, Batch Normalization, Programming Frameworks

1. If searching among a large number of hyperparameters, you should try values in a grid rather than random values, so that you can carry out the search more systematically and not rely on chance. True or False?
    1. True
    1. **False**
1. Every hyperparameter, if set poorly, can have a huge negative impact on training, and so all hyperparameters are about equally important to tune well. True or False?
    1. True
    1. **False**
1. During hyperparameter search, whether you try to babysit one model (“Panda” strategy) or train a lot of models in parallel (“Caviar”) is largely determined by:
    1. Whether you use batch or mini-batch optimization
    1. The presence of local minima (and saddle points) in your neural network
    1. **The amount of computational power you can access**
    1. The number of hyperparameters you have to tune
1. If you think $\beta$(hyperparameter for momentum) is between on 0.9 and 0.99, which of the following is the recommended way to sample a value for beta?
    1. Code1
        ```python
        r = np.random.rand()
        beta = r*0.09 + 0.9
        ```
    1. **Code2**
        ```python
        r = np.random.rand()
        beta = 1-10**(- r - 1)
        ```
    1. Code3
        ```python
        r = np.random.rand()
        beta = 1-10**(- r + 1)
        ```
    1. Code4
        ```python
        r = np.random.rand()
        beta = r*0.9 + 0.09
        ```
1. Finding good hyperparameter values is very time-consuming. So typically you should do it once at the start of the project, and try to find very good hyperparameters so that you don’t ever have to revisit tuning them again. True or false?
    1. True
    1. **False**
1. In batch normalization as presented in the videos, if you apply it on the $l$th layer of your neural network, what are you normalizing?
    1. $z^{[l]}$**True**
    1. $W^{[l]}$
    1. $a^{[l]}$
    1. $b^{[l]}$
1. In the normalization formula $z^{(i)}_{norm} =\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}}$, why do we use epsilon?
    1. To have a more accurate normalization
    1. **To avoid division by zero**
    1. In case μ is too small
    1. To speed up convergence
1. Which of the following statements about $\gamma$ and $\beta$ in Batch Norm are true?
    1. **They can be learned using Adam, Gradient descent with momentum, or RMSprop, not just with gradient descent.**
    1. β and γ are hyperparameters of the algorithm, which we tune via random sampling.
    1. **They set the mean and variance of the linear variable z[ l] of a given layer.**
    1. There is one global value of $\gamma \in R$ and one global value of $\beta \in R$ for each layer, and applies to all the hidden units in that layer.
    1. The optimal values are γ = $\sqrt{\sigma^2+\varepsilon}$, and β
1. After training a neural network with Batch Norm, at test time, to evaluate the neural network on a new example you should:
    1. Use the most recent mini-batch’s value of μ and σ2 to perform the needed normalizations
    1. If you implemented Batch Norm on mini-batches of (say) 256 examples, then to evaluate on one test example, duplicate that example 256 times so that you’re working with a mini-batch the same size as during training.
    1. **Perform the needed normalizations, use μ and σ2 estimated using an exponentially weighted average across mini-batches seen during training.**
    1. Skip the step where you normalize using and since a single test example cannot be normalized.
1. Which of these statements about deep learning programming frameworks are true? (Check all that apply)
    1. **Even if a project is currently open source, good governance of the project helps ensure that the it remains open even in the long term, rather than become closed or modified to benifit only one company.**
    1. **A programming framework allows you to code up deep learning algorithms with typically fewer lines of code than a lower-level language such as Python.**

### TensorFlow Tutorial

[ipynb]()

# 结构化机器学习项目

## 机器学习(ML)策略1

### 笔记

#### 正交化 orthogonalization

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

#### 单一数字评估指标

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

#### 满足指标和优化指标

优化指标Optimizing metric——准确度，一个

满足指标Satisficing metric——运行时间（阈值），多个

#### 训练/开发/测试数据

开发集和评估指标，决定了靶心。

开发集和测试集需要是一个分布

#### 调整靶心

通过权重修改损失函数来调节错误率。

#### 可避免偏差

训练错误和贝叶斯（人类表现）的差距叫可避免偏差

训练错误和开发错误的差距叫做方差

#### 人类水平表现 human-level performance

贝叶斯误差的替代品 

## 机器学习(ML)策略2

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

![](http://outz1n6zr.bkt.clouddn.com/lenet5.PNG)

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

![](http://outz1n6zr.bkt.clouddn.com/alexnet.PNG)

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

![](http://upload-images.jianshu.io/upload_images/1689929-063fb60285b6ed42.png?imageMogr2/auto-orient/strip%7CimageView2/2)

##### VGG-16

![](http://outz1n6zr.bkt.clouddn.com/vgg16.PNG)

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

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_112752.png)

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_112910.png)

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

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_142806.png)

#### 1x1卷积

在每个像素上的深度上的全连接运算。可以用来改变通道深度，或者对每个像素分别添加了非线性变换。

Network in Network

#### Inception

一个Inception模块，帮你解决使用什么尺寸的卷积层和何时使用池化层。

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_144756.png)

为了解决计算成本问题，引入1x1卷积进行优化计算。

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_144817.png)

事实证明，只要合理构建瓶颈层，不仅不会降低网络性能，还会降低计算成本。

具体模块

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_145952.png)

具体网络

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_150301.png)

#### 迁移学习

冻结一部分网络，自己训练一部分网络，并替换输出层的softmax

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_154149.png)

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

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_161948.png)

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

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_170438.png)

整个大图像做卷积运算。

![](http://outz1n6zr.bkt.clouddn.com/2017-11-22_171833.png)

#### 边界框预测

YOLO算法（You Only Look Once）

将整个大图像划分为3x3、19x19这样的格子，然后修改Label Y，每个小格子中，若目标对象的中心点位于该格内，则该格Label Y中的$P_c$为1。相邻格子就算包含了目标对象的一部分，$P_c$也为0

## 第四周 特殊应用：人脸识别和神经风格转变