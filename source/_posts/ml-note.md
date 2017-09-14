---
title: 机器学习入门笔记
date: 2017-09-14 16:05:13
tags: ML
categories: 机器学习
---

## 安装octave绘制3D函数图像

<!-- more -->

[http://www.shareditor.com/blogshow?blogId=28](http://www.shareditor.com/blogshow?blogId=28)

```
apt-get install octave
```

## 用scikit-learn求解一元线性回归问题

[原文](http://www.shareditor.com/blogshow?blogId=54)

### scikit-learn的一元线性回归

安装[scikit-learn](http://scikit-learn.org/stable/install.html)

```
pip install -U scikit-learn numpy scipy
```

[LinearRegression文档](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

代码`scikit_learn_linear_model_demo.py`

```python
import numpy as np
from sklearn.linear_model import LinearRegression

x = [[1],[2],[3],[4],[5],[6]]
y = [[1],[2.1],[2.9],[4.2],[5.1],[5.8]]
model = LinearRegression()
model.fit(x, y)
predicted = model.predict(13)
print predicted
```

输出

```
[[ 12.82666667]]
```

### 画一元线性图像

安装

```
apt-get install -y python-dev python-tk
pip install matplotlib
```

代码`plot_linear.py`

```python
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties()

plt.figure()
plt.title('linear sample')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0, 10, 0, 10])
plt.grid(True)
x = [[1],[2],[3],[4],[5],[6]]
y = [[1],[2.1],[2.9],[4.2],[5.1],[5.8]]
plt.plot(x, y, 'k.')
plt.show()
```

> plot()函数的第三个参数非常简单：k表示卡其色khaki，g表示绿色green，r表示红色red，'.'表示点，'-'表示线

预测加画图代码

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = [[1],[2],[3],[4],[5],[6]]
y = [[1],[2.1],[2.9],[4.2],[5.1],[5.8]]
model = LinearRegression()
model.fit(x, y)
x2 = [[0], [2.5], [5.3], [9.1]]
y2 = model.predict(x2)

plt.figure()
plt.title('linear sample')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0, 10, 0, 10])
plt.grid(True)
plt.plot(x, y, 'k.')
plt.plot(x2, y2, 'g-')
plt.show()
```

### 方差

```python
import numpy as np
np.var([1,2,3,4,5,6], ddof=1)
```

> 结果为3.5，ddof是无偏估计校正技术

### 协方差

```python
import numpy as np
np.cov([1,2,3,4,5,6], [1,2.1,2.9,4.2,5.1,5.8])[0][1]
```

### 参数查看

b = 方差/协方差

输入`print model.coef_`

### 模型评估

```python
model.score(x2, y2)
```

## 用scikit-learn求解多元线性回归问题

[原文]()

### 多元线性回归模型

y = x0 + a1 * x1 + a2 * x2 + ...

若方程为 y = 1 + 2 * x1 + 3 * x2

输入 X = [[1,1,1],[1,1,2],[1,2,1]]

计算得到 y = [[6],[9],[8]]

代码`scikit_learn_multvariable_linear_model_demo.py`

```python
from numpy.linalg import inv
from numpy import dot, transpose

X = [[1,1,1],[1,1,2],[1,2,1]]
y = [[6],[9],[8]]

print dot(inv(dot(transpose(X),X)), dot(transpose(X),y))
```

> transpose是求转置，dot是求矩阵乘积，inv是求矩阵的逆

等同于以下代码

```python
from numpy.linalg import lstsq
print lstsq(X, y)[0]
```

> lstsq就是least square最小二乘

结果

```
[[ 1.]
 [ 2.]
 [ 3.]]
```

### 用scikit-learn求解多元线性回归

```python
from sklearn.linear_model import LinearRegression

X = [[1,1,1],[1,1,2],[1,2,1]]
y = [[6],[9],[8]]

model = LinearRegression()
model.fit(X, y)
x2 = [[1,3,5]]
y2 = model.predict(x2)
print y2
```

输出

```
[[ 22.]]
```

## 用matplotlib绘制精美的图表

[原文](http://www.shareditor.com/blogshow?blogId=55)

安装见上文

### 绘制一元函数图像y=ax+b

代码`single_variable.py`

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np

plt.figure() # 实例化作图变量
plt.title('single variable') # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.axis([0, 5, 0, 10]) # x轴范围0-5，y轴范围0-10
plt.grid(True) # 是否绘制网格线
xx = np.linspace(0, 5, 10) # 在0-5之间生成10个点的向量
plt.plot(xx, 2*xx, 'g-') # 绘制y=2x图像，颜色green，形式为线条
plt.show() # 展示图像
```

### 绘制正弦曲线y=sin(x)

代码`sinx.py`

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np

plt.figure() # 实例化作图变量
plt.title('single variable') # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.axis([-12, 12, -1, 1]) # x轴范围-12到12，y轴范围-1到1
plt.grid(True) # 是否绘制网格线
xx = np.linspace(-12, 12, 1000) # 在-12到12之间生成1000个点的向量
plt.plot(xx, np.sin(xx), 'g-', label="$sin(x)$") # 绘制y=sin(x)图像，颜色green，形式为线条
plt.plot(xx, np.cos(xx), 'r--', label="$cos(x)$") # 绘制y=cos(x)图像，颜色red，形式为虚线
plt.legend() # 绘制图例
plt.show() # 展示图像
```

> [legend位置](https://matplotlib.org/api/legend_api.html)

### 绘制多轴图

代码`multi_axis.py`

```
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np

def draw(plt):
    plt.axis([-12, 12, -1, 1]) # x轴范围-12到12，y轴范围-1到1
    plt.grid(True) # 是否绘制网格线
    xx = np.linspace(-12, 12, 1000) # 在-12到12之间生成1000个点的向量
    plt.plot(xx, np.sin(xx), 'g-', label="$sin(x)$") # 绘制y=sin(x)图像，颜色green，形式为线条
    plt.plot(xx, np.cos(xx), 'r--', label="$cos(x)$") # 绘制y=cos(x)图像，颜色red，形式为虚线
    plt.legend() # 绘制图例

plt.figure() # 实例化作图变量
plt1 = plt.subplot(2,2,1) # 两行两列中的第1张图
draw(plt1)
plt2 = plt.subplot(2,2,2) # 两行两列中的第2张图
draw(plt2)
plt3 = plt.subplot(2,2,3) # 两行两列中的第3张图
draw(plt3)
plt4 = plt.subplot(2,2,4) # 两行两列中的第4张图
draw(plt4)

plt.show() # 展示图像
```

### 绘制3D图像

代码`plot_3d.py`

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 500) # theta旋转角从-4pi到4pi，相当于两圈
z = np.linspace(0, 2, 500) # z轴从下到上,从-2到2之间画100个点
r = z # 半径设置为z大小
x = r * np.sin(theta) # x和y画圆
y = r * np.cos(theta) # x和y画圆
ax.plot(x, y, z, label='curve')
ax.legend()

plt.show()
```

### 3D散点图

代码`plot_3d_scatter.py`

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
xx = np.linspace(0, 5, 10) # 在0-5之间生成10个点的向量
yy = np.linspace(0, 5, 10) # 在0-5之间生成10个点的向量
zz1 = xx
zz2 = 2*xx
zz3 = 3*xx
ax.scatter(xx, yy, zz1, c='red', marker='o') # o型符号
ax.scatter(xx, yy, zz2, c='green', marker='^') # 三角型符号
ax.scatter(xx, yy, zz3, c='black', marker='*') # 星型符号
ax.legend()

plt.show()
```

### 绘制3D表面

代码`plot_3d_surface.py`

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

Z = X**2+Y**2

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()
```

## 用scikit-learn求解多项式回归问题

[原文](http://www.shareditor.com/blogshow?blogId=56)

### 住房价格样本

样本 | 面积(平方米) | 价格(万元)
--- | --- | ---
1 | 50 | 150
2 | 100 | 200
3 | 150 | 250
4 | 200 | 280
5 | 250 | 310
6 | 300 | 330

### 做图像

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np

plt.figure() # 实例化作图变量
plt.title('single variable') # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.axis([30, 400, 100, 400])
plt.grid(True) # 是否绘制网格线

xx = [[50],[100],[150],[200],[250],[300]]
yy = [[150],[200],[250],[280],[310],[330]]
plt.plot(xx, yy, 'k.')
plt.show() # 展示图像
```

### 线性回归

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

plt.figure() # 实例化作图变量
plt.title('single variable') # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.axis([30, 400, 100, 400])
plt.grid(True) # 是否绘制网格线

xx = [[50],[100],[150],[200],[250],[300]]
yy = [[150],[200],[250],[280],[310],[330]]
plt.plot(xx, yy, 'k.')

model = LinearRegression()
model.fit(xx, yy)
print model.coef_
x2 = [[30], [400]]
y2 = model.predict(x2)
plt.plot(x2, y2, 'g-')

plt.show() # 展示图像
```

### 采用多项式回归

二次多项式

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

plt.figure() # 实例化作图变量
plt.title('single variable') # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.axis([30, 400, 100, 400])
plt.grid(True) # 是否绘制网格线

X = [[50],[100],[150],[200],[250],[300]]
y = [[150],[200],[250],[280],[310],[330]]
X_test = [[250],[300]] # 用来做最终效果测试
y_test = [[310],[330]] # 用来做最终效果测试
plt.plot(X, y, 'k.')

model = LinearRegression()
model.fit(X, y)
X2 = [[30], [400]]
y2 = model.predict(X2)
plt.plot(X2, y2, 'g-')

xx = np.linspace(30, 400, 100) # 设计x轴一系列点作为画图的x点集
quadratic_featurizer = PolynomialFeatures(degree=2) # 实例化一个二次多项式特征实例
X_train_quadratic = quadratic_featurizer.fit_transform(X) # 用二次多项式对样本X值做变换
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1)) # 把训练好X值的多项式特征实例应用到一系列点上,形成矩阵
regressor_quadratic = LinearRegression() # 创建一个线性回归实例
regressor_quadratic.fit(X_train_quadratic, y) # 以多项式变换后的x值为输入，代入线性回归模型做训练
print regressor_quadratic.coef_
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-') # 用训练好的模型作图

print '一元线性回归 r-squared', model.score(X_test, y_test)
X_test_quadratic = quadratic_featurizer.transform(X_test)
print '二次回归     r-squared', regressor_quadratic.score(X_test_quadratic, y_test)

plt.show() # 展示图像
```

三次多项式

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

plt.figure() # 实例化作图变量
plt.title('single variable') # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.axis([30, 400, 100, 400])
plt.grid(True) # 是否绘制网格线

X = [[50],[100],[150],[200],[250],[300]]
y = [[150],[200],[250],[280],[310],[330]]
X_test = [[250],[300]] # 用来做最终效果测试
y_test = [[310],[330]] # 用来做最终效果测试
plt.plot(X, y, 'k.')

model = LinearRegression()
model.fit(X, y)
X2 = [[30], [400]]
y2 = model.predict(X2)
plt.plot(X2, y2, 'g-')

xx = np.linspace(30, 400, 100) # 设计x轴一系列点作为画图的x点集
quadratic_featurizer = PolynomialFeatures(degree=2) # 实例化一个二次多项式特征实例
X_train_quadratic = quadratic_featurizer.fit_transform(X) # 用二次多项式对样本X值做变换
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1)) # 把训练好X值的多项式特征实例应用到一系列点上,形成矩阵
regressor_quadratic = LinearRegression() # 创建一个线性回归实例
regressor_quadratic.fit(X_train_quadratic, y) # 以多项式变换后的x值为输入，代入线性回归模型做训练
print regressor_quadratic.coef_
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-') # 用训练好的模型作图

cubic_featurizer = PolynomialFeatures(degree=3)
X_train_cubic = cubic_featurizer.fit_transform(X)
regressor_cubic = LinearRegression()
regressor_cubic.fit(X_train_cubic, y)
xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_cubic.predict(xx_cubic))

print '一元线性回归 r-squared', model.score(X_test, y_test)
X_test_quadratic = quadratic_featurizer.transform(X_test)
print '二次回归     r-squared', regressor_quadratic.score(X_test_quadratic, y_test)
X_test_cubic = cubic_featurizer.transform(X_test)
print '三次回归     r-squared', regressor_cubic.score(X_test_cubic, y_test)

plt.show() # 展示图像
```

## 用随机梯度下降法(SGD)做线性拟合

scikit-learn的线性回归模型都是通过最小化成本函数来计算参数的，通过矩阵乘法和求逆运算来计算参数。当变量很多的时候计算量会非常大，因此我们改用梯度下降法，批量梯度下降法每次迭代都用所有样本，快速收敛但性能不高，随机梯度下降法每次用一个样本调整参数，逐渐逼近，效率高，本节我们来利用随机梯度下降法做拟合。

### 梯度下降法

梯度下降就好比从一个凹凸不平的山顶快速下到山脚下，每一步都会根据当前的坡度来找一个能最快下来的方向。随机梯度下降英文是Stochastic gradient descend(SGD)，在scikit-learn中叫做SGDRegressor。

### 样本实验

依然用上一节的房价样本

```python
X = [[50],[100],[150],[200],[250],[300]]
y = [[150],[200],[250],[280],[310],[330]]
```

代码`sgd_regressor.py`

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

plt.figure() # 实例化作图变量
plt.title('single variable') # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.grid(True) # 是否绘制网格线

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X = [[50],[100],[150],[200],[250],[300]]
y = [[150],[200],[250],[280],[310],[330]]
X = X_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)
X_test = [[40],[400]] # 用来做最终效果测试
X_test = X_scaler.transform(X_test)

plt.plot(X, y, 'k.')

model = SGDRegressor()
model.fit(X, y.ravel())
y_result = model.predict(X_test)
plt.plot(X_test, y_result, 'g-')

plt.show() # 展示图像
```

效果不好，扩大样本

```python
X = [[50],[100],[150],[200],[250],[300],[50],[100],[150],[200],[250],[300],[50],[100],[150],[200],[250],[300],[50],[100],[150],[200],[250],[300],[50],[100],[150],[200],[250],[300],[50],[100],[150],[200],[250],[300],[50],[100],[150],[200],[250],[300],[50],[100],[150],[200],[250],[300]]
y = [[150],[200],[250],[280],[310],[330],[150],[200],[250],[280],[310],[330],[150],[200],[250],[280],[310],[330],[150],[200],[250],[280],[310],[330],[150],[200],[250],[280],[310],[330],[150],[200],[250],[280],[310],[330],[150],[200],[250],[280],[310],[330],[150],[200],[200],[200],[200],[200]]
```

## 用scikit-learn做特征提取

[原文](http://www.shareditor.com/blogshow?blogId=58)

### 分类变量的特征提取

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [{'city': '北京'},{'city': '天津'}, {'city': '上海'}]
print(onehot_encoder.fit_transform(instances).toarray())
```

输出

```
[[ 0.  1.  0.]
 [ 0.  0.  1.]
 [ 1.  0.  0.]]
```

### 文字特征提取

文字特征无非这几种：有这个词还是没有、这个词的TF-IDF。

第一种为词库表示法

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
        'UNC played Duke in basketball',
        'Duke lost the basketball game' ]
vectorizer = CountVectorizer()
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_
```

输出

```
[[1 1 0 1 0 1 0 1]
 [1 1 1 0 1 0 1 0]]
{u'duke': 1, u'basketball': 0, u'lost': 4, u'played': 5, u'game': 2, u'unc': 7, u'in': 3, u'the': 6}
```

> 数值为1表示词表中的这个词出现，为0表示未出现
> 词表中的数值表示单词的坐标位置

第二种情况TF-IDF表示词的重要性，代码

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
        'The dog ate a sandwich and I ate a sandwich',
        'The wizard transfigured a sandwich' ]
vectorizer = TfidfVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
```

输出

```
[[ 0.75458397  0.37729199  0.53689271  0.          0.        ]
 [ 0.          0.          0.44943642  0.6316672   0.6316672 ]]
{u'sandwich': 2, u'wizard': 4, u'dog': 1, u'transfigured': 3, u'ate': 0}
```

> 值最高的是第一个句子中的ate，因为它在这一个句子里出现了两次
> 值最低的自然是本句子未出现的单词

### 数据标准化

数据标准化就是把数据转成均值为0，是单位方差的。比如对如下矩阵做标准化：

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from sklearn import preprocessing
import numpy as np
X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
    ])
print(preprocessing.scale(X))
```

输出

```
[[ 0.         -0.70710678 -1.38873015  0.52489066  0.59299945 -1.35873244]
 [ 0.         -0.70710678  0.46291005  0.87481777  0.81537425  1.01904933]
 [ 0.          1.41421356  0.9258201  -1.39970842 -1.4083737   0.33968311]]
```

## 二元分类效果的评估方法

[原文](http://www.shareditor.com/blogshow?blogId=59)

效果评估是模型选择和算法设计的重要步骤，知道评估优劣才能选择最佳的模型和算法，本节介绍一些有关评估方法的定义，凡是在统计或大数据领域都用得到。

- 真阳性 true positives, TP
- 真阴性 true negatives, TN
- 假阳性 false positives, FP
- 假阴性 false negatives, FN

- 准确率 分类器预测正确性的比例，可以通过LogisticRegression.score() 来计算准确率
- 精确率 分类器预测出的脏话中真的是脏话的比例 P=TP/(TP+FP)
- 召回率 也叫灵敏度。所有真的脏话被分类器正确找出来的比例。 R=TP/(TP+FN)

- 综合评价指标 F-measure，精确率和召回率的调和均值。精确率和召回率都不能从差的分类器中区分出好的分类器，综合评价指标平衡了精确率和召回率。1/F+1/F=1/P+1/R 即 F=2*PR/(P+R)
- 误警率 假阳性率，所有阴性样本中分类器识别为阳性的样本所占比例 F=FP/(TN+FP)
- ROC(Receiver Operating Characteristic) ROC曲线画的是分类器的召回率与误警率(fall-out)的曲线
- AUC(Area Under Curve) ROC曲线下方的面积,它把ROC曲线变成一个值,表示分类器随机预测的效果 scikit-learn画ROC曲线和AUC值的方法如下：

## 用scikit-learn的网格搜索快速找到最优模型参数

[原文](http://www.shareditor.com/blogshow?blogId=60)

任何一种机器学习模型都附带很多参数，不同场景对应不同的最佳参数，手工尝试各种参数无疑浪费很多时间，scikit-learn帮我们实现了自动化，那就是网格搜索。

### 网格搜索

网格指的是不同参数不同取值交叉后形成的一个多维网格空间。比如参数a可以取1、2，参数b可以取3、4，参数c可以取5、6，那么形成的多维网格空间就是：一共2*2*2=8种情况。

网格搜索就是遍历这8种情况进行模型训练和验证，最终选择出效果最优的参数组合

### 用法举例

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 构造样本，这块得多构造点，不然会报class不足的错误，因为gridsearch会拆分成小组
X = []
X.append("fuck you")
X.append("fuck you all")
X.append("hello everyone")
X.append("fuck me")
X.append("hello boy")
X.append("fuck you")
X.append("fuck you all")
X.append("hello everyone")
X.append("fuck me")
X.append("hello boy")
X.append("fuck you")
X.append("fuck you all")
X.append("hello everyone")
X.append("fuck me")
X.append("hello boy")
X.append("fuck you")
X.append("fuck you all")
X.append("hello everyone")
X.append("fuck me")
X.append("hello boy")
X.append("fuck you")
X.append("fuck you all")
X.append("hello everyone")
X.append("fuck me")
X.append("hello boy")

y = [1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1]

# 这是执行的序列，gridsearch是构造多进程顺序执行序列并比较结果
# 这里的vect和clf名字自己随便起，但是要和parameters中的前缀对应
pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
    ])

# 这里面的max_features必须是TfidfVectorizer的参数, 里面的取值就是子进程分别执行所用
parameters = {
        'vect__max_features': (3, 5),
        }

# accuracy表示按精确度判断最优值
grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1, scoring = 'accuracy', cv = 3)
grid_search.fit(X, y)

print '最佳效果: %0.3f' % grid_search.best_score_
print '最优参数组合: '
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
```

输出

```
Fitting 3 folds for each of 2 candidates, totalling 6 fits
[Parallel(n_jobs=-1)]: Done   6 out of   6 | elapsed:    0.0s finished
最佳效果: 0.800
最优参数组合: 
    vect__max_features: 3
```

## 用scikit-learn做聚类分析

[原文](http://www.shareditor.com/blogshow?blogId=61)

线性回归和逻辑回归都是监督学习方法，聚类分析是非监督学习的一种，可以从一批数据集中探索信息，比如在社交网络数据中可以识别社区，在一堆菜谱中识别出菜系。本节介绍K-means聚类算法。

### K-means

k是一个超参数，表示要聚类成多少类。K-means计算方法是重复移动类的重心，以实现成本函数最小化。

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/2224c88869e0627a6f76ee18d107c2d740029c10.png)

### 试验

构造一些样本用户试验，如下：

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np

# 生成2*10的矩阵，且值均匀分布的随机数
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))

# 顺序连接两个矩阵，形成一个新矩阵,所以生成了一个2*20的矩阵，T做转置后变成20*2的矩阵,刚好是一堆(x,y)的坐标点
X = np.hstack((cluster1, cluster2)).T

plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')
plt.show()
```

通过k-means做聚类，输出重心点，代码：

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# 生成2*10的矩阵，且值均匀分布的随机数
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))

# 顺序连接两个矩阵，形成一个新矩阵,所以生成了一个2*20的矩阵，T做转置后变成20*2的矩阵,刚好是一堆(x,y)的坐标点
X = np.hstack((cluster1, cluster2)).T

plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 'ro')

plt.show()
```

### 肘部法则

现实情况是多个点并不像上面这么聚类清晰，你说不清它应该聚类成2、3、4个点，因此我们需要通过分别计算k=(2,3,4)的聚类结果，并比较他们的成本函数值，随着k的增大，成本函数值会不断降低，只有快速降低的那个k值才是最合适的k值，如下：

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 生成2*10的矩阵，且值均匀分布的随机数
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(1.5, 2.5, (2, 10))
cluster3 = np.random.uniform(1.5, 3.5, (2, 10))
cluster4 = np.random.uniform(3.5, 4.5, (2, 10))

# 顺序连接两个矩阵，形成一个新矩阵,所以生成了一个2*20的矩阵，T做转置后变成20*2的矩阵,刚好是一堆(x,y)的坐标点
X1 = np.hstack((cluster1, cluster2))
X2 = np.hstack((cluster3, cluster4))
X = np.hstack((X1, X2)).T

K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # 求kmeans的成本函数值
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.figure()
plt.grid(True)
plt1 = plt.subplot(2,1,1)
# 画样本点
plt1.plot(X[:,0],X[:,1],'k.');
plt2 = plt.subplot(2,1,2)
# 画成本函数值曲线
plt2.plot(K, meandistortions, 'bx-')
plt.show()
```

从曲线上可以看到，随着k的增加，成本函数值在降低，但降低的变化幅度不断在减小，因此急速降低才是最合适的，这里面也许3是比较合适的，你也许会有不同看法

通过这种方法来判断最佳K值的方法叫做肘部法则，你看图像像不像一个人的胳膊肘？

## 神经网络模型的原理

[原文](http://www.shareditor.com/blogshow?blogId=91)

深度学习本质上就是机器学习的一个topic，是深度人工神经网络的另一种叫法，因此理解深度学习首先要理解人工神经网络。

### 人工神经网络

人工神经网络又叫神经网络，是借鉴了生物神经网络的工作原理形成的一种数学模型。下面是一张生物神经元的图示：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/ec026d5b1a84c5daddb54ad190304458cf120872.png)

生物神经网络就是由大量神经元构成的网络结构，如下图.

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/39d7f81c81176a142914bc9053f19be7d84ba38d.png)

生物的神经网络是通过神经元、细胞、触电等结构组成的一个大型网络结构，用来帮助生物进行思考和行动等。那么人们就想到了电脑是不是也可以像人脑一样具有这种结构，这样是不是就可以思考了？

类似于神经元的结构，人工神经网络也是基于这样的神经元组成：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/65dd819d8f9fede7d4514a29439fc0f83f7c5a37.png)

这里面的x1、x2、x3是输入值，中间的圆就像是神经元，经过它的计算得出hw,b(x)的结果作为神经元的输出值。

由这样的神经元组成的网络就是人工神经网络：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/cb1ccc8c753851f8c650c3e5049d93e82def667b.png)

其中橙色的圆都是用来计算hw,b(x)的，纵向我们叫做层（Layer），每一层都以前一层为输入，输出的结果传递给下一层。

### 这样的结构有什么特别的吗？

如果我们把神经网络看做一个黑盒，那么x1、x2、x3是这个黑盒的输入X，最右面的hw,b(x)是这个黑盒的输出Y，按照之前几篇机器学习的文章可以知道：这可以通过一个数学模型来拟合，通过大量训练数据来训练这个模型，之后就可以预估新的样本X应该得出什么样的Y。

但是使用普通的机器学习算法训练出的模型一般都比较肤浅，就像是生物的进化过程，如果告诉你很久以前地球上只有三叶虫，现在地球上有各种各样的生物，你能用简单的模型来表示由三叶虫到人类的进化过程吗？不能。但是如果模拟出中间已知的多层隐藏的阶段（低等原始生物、无脊椎动物、脊椎动物、鱼类、两栖类、爬行动物、哺乳动物、人类时代）就可以通过海量的训练数据模拟出。

也可以类比成md5算法的实现，给你无数个输入字符串和它的md5值，你能用肤浅的算法推出md5的算法吗？不能。因为md5的计算是一阶段一阶段的，后一阶段的输入依赖前一阶段的结果，无法逆推。但是如果已知中间几个阶段，只是不知道这几个阶段的参数，那么可以通过海量数据训练出来。

以上说明了神经网络结构的特别之处：通过较深的多个层次来模拟真实情况，从而构造出最能表达真实世界的模型，它的成本就是海量的训练数据和巨大的计算量。

### 神经网络模型的数学原理

每一个神经元的数学模型是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/f6a2a07c6d719bdb6ecc29ff32d932602f3625b3.png)

其中的矩阵向量乘法

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/042da8b3dc73cbe888c578366b1fc4d92ee719a4.png)

表示的就是输入多个数据的加权求和，这里的b（也就是上面图中的+1）是截距值，用来约束参数值，就像是一个向量(1,2,3)可以写成(2,4,6)也可以写成(10,20,30)，那么我们必须取定一个值，有了截距值就可以限定了

其中f叫做激活函数，激活函数的设计有如下要求：1）保证后期计算量尽量小；2）固定取值范围；3）满足某个合理的分布。常用的激活函数是sigmond函数和双曲正切函数(tanh)：

sigmond函数：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/86132e278f66841c7fb3bcffe0b5fe8065992205.png)

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/3457bfbde22dae706815bbe08a743c232f6163d9.png)

双曲正切函数(tanh)：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/b21e57c3e39a687c7d0cf3ecdb3ee35f625ea3ab.png)

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/48108636bedd7ad70c2eadf3ae06fd25e45ea5d7.png)

这两个函数显然满足2）固定取值范围；3）满足某个合理的分布，那么对于1）保证后期计算量尽量小这个要求来说，他们的好处在于：

sigmond函数的导数是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/ea56bb15e194fb1abbb02e299168034bf21c0c68.png)

tanh函数的导数是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/d680876c385acd3f9cdad9fb3857faf05be23859.png)

这会减少非常多的计算量，后面就知道了。

当计算多层的神经网络时，对于如下三层神经网络来说

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/0f3aef266b063d9c199a3f8f6fbbe6c289e03abd.png)

我们知道：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/2ac3ebb584159175856886b7edac692a097bd920.png)

其中的

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/4cb55f04a5cbfc85e4132ae86ddf5312b3716520.png)

分别表示第2层神经元的输出的第1、2、3个神经元产生的值

这三个值经过第3层最后一个神经元计算后得出最终的输出是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/ce100e312f3b4f9a9ed65824000705a61bd7c7d8.png)

以上神经网络如果有更多层，那么计算原理相同

我们发现这些神经元的激活函数f是相同的，唯一不同的就是权重W，那么我们做学习训练的目标就是求解这里的W，那么我们如何通过训练获得更精确的W呢？

### 反向传导算法

回想一下前面文章讲过的回归模型，我们也是知道大量训练样本(x,y)，未知的是参数W和b，那么我们计算W的方法是：先初始化一个不靠谱的W和b，然后用输入x和W和b预估y，然后根据预估的y和实际的y之间的差距来通过梯度下降法更新W和b，然后再继续下一轮迭代，最终逼近正确的W和b

神经网络算法也一样的道理，使用梯度下降法需要设计一个代价函数：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/55ec3240a35a666073392127eaec8e4f7bf8fb3e.png)

以上是对于一个(x,y)的代价函数，那么当我们训练很多个样本时：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/46d983583548b48eb9fbf25fa59ce8c7b95994d3.png)

其中m是样本数，左项是均方差，右项是规则化项，我们的目的就是经过多伦迭代让代价函数最小

我来单独解释一下我对这个规则化项的理解：规则化项的目的是防止过拟合，过拟合的含义就是“太适合这些样本了，导致不适合样本之外的数据，泛化能力低”，规则化项首先增大了代价函数的值，因为我们训练的目的是减小代价函数，所以我们自然就会经过多轮计算逐步减小规则化项，规则化项里面是各个W的平方和，因为∑W=1，所以要想平方和变小，只有让各个W的值尽量相同，这就需要做一个折中，也就是W既要显示出各项权重的不同，又要降低差别，因此这里的λ的值就比较关键了，λ大了权重就都一样了，小了就过拟合了，所以需要根据经验给一个合适的值。

### 具体计算过程

首先我们为W和b初始化一个很小的随机值，然后分别对每个样本经过上面说过的神经网络的计算方法，计算出y的预估值

然后按照梯度下降法对W和b进行更新：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/9688c310c4c3c0f42fb3cbb00bc910340c0faacb.png)

这里面最关键的是偏导的计算方法，对于最终的输出节点来说，代价函数J(W,b)的计算方法比较简单，就是把输出节点的激活值和实际值之间的残差代入J(W,B)公式。而隐藏层里的节点的代价函数该怎么计算呢？

我们根据前向传导（根据输入x计算hW,b(x)的方法）算法来反推：

我们把第l层的第i个节点的残差记作：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/b98904879e0ef6433d3cd95b87162d5dd52e4ac2.png)

因为hW,b(x)=f(Wx)，这里面我们把Wx记作z，那么残差表达的其实是z的残差，也就是代价函数关于z的偏导

那么输出层的残差就是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/b185972824bfe90da5964bd6eff817edeaed3e61.png)

前一层的残差的推导公式为：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/bc5660978169eb6a3793c496b8869d38ed296cfe.png)

再往前的每一层都按照这个公式计算得出残差，这个过程就叫做反向传导算法

下面在回过头来看我们的更新算法

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/a0211982e33d83001a55d94c3dbdbeab52953dff.png)

偏导数的求法如下：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/259747c458274dd26c0fd7ec2fe158044d313302.png)

说一下我对这个公式的理解：代价函数对W的偏导表达的是权重(第l层上第i个节点第j个输入)的变化，而这个变化就是一个误差值，误差体现在哪里呢？体现在它影响到的下一层节点的残差δ，那么它对这个残差的影响有多大呢，在于它的输出值a，所以得出这个偏导就是aδ。代价函数对b的偏导原理类似。

现在我们有了所有的a和δ，就可以更新所有的W了，完成了一轮迭代

大家已经注意到了，在计算δ的方法中用到了f'(z)，这也就是激活函数的导数，现在明白为什么激活函数设计成sigmond或者tanh了吧？因为他们的导数更容易计算

### 总结一下整个计算过程

1. 初始化W和b为小随机数

2. 遍历所有样本，利用前向传导算法计算出神经网络的每一层输出a和最终的输出值hW,b(x)

3. 利用hW,b(x)和真实值y计算输出层的残差δ

4. 利用反向传导算法计算出所有层所有节点的残差δ

5. 利用每一层每一个节点的a和δ计算代价函数关于W和b的偏导

6. 用得出的偏导来更新权重

7. 返回2进行下一轮迭代直到代价函数不再收敛为止

8. 得到我们的神经网络

参考文献：UFLDL教程

## 用scikit-learn做逻辑回归

一元线性、多元线性、多项式回归都属于广义的线性回归，这几类线性回归主要用于预测连续变量的值。本节介绍广义线性回归的另一种主要用于分类任务的形式：逻辑回归。

### 二类分类问题

逻辑回归最广泛的应用就是二类分类，我们以脏话判别为例来利用逻辑回归，对一句话做脏话分析判断

输入样本如下：

是脏话：fuck you

是脏话：fuck you all

不是脏话：hello everyone

我们来预测以下两句话是否是脏话：

```
fuck me
hello boy
```

代码

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression

X = []

# 前三行作为输入样本
X.append("fuck you")
X.append("fuck you all")
X.append("hello everyone")

# 后两句作为测试样本
X.append("fuck me")
X.append("hello boy")

# y为样本标注
y = [1,1,0]

vectorizer = TfidfVectorizer()

# 取X的前三句作为输入做tfidf转换
X_train = vectorizer.fit_transform(X[:-2])

# 取X的后两句用上句生成的tfidf做转换
X_test = vectorizer.transform(X[-2:])

# 用逻辑回归模型做训练
classifier = LogisticRegression()
classifier.fit(X_train, y)

# 做测试样例的预测
predictions = classifier.predict(X_test)
print predictions
```

输出结果如下：

```
[1 0]
```

判断成：

是脏话：fuck me

不是脏话：hello boy

## 利用tensorflow做手写数字识别

[原文](http://www.shareditor.com/blogshow?blogId=94)

模式识别领域应用机器学习的场景非常多，手写识别就是其中一种，最简单的数字识别是一个多类分类问题，我们借这个多类分类问题来介绍一下google最新开源的tensorflow框架，后面深度学习的内容都会基于tensorflow来介绍和演示。

### 什么是tensorflow

tensor意思是张量，flow是流。

张量原本是力学里的术语，表示弹性介质中各点应力状态。在数学中，张量表示的是一种广义的“数量”，0阶张量就是标量(比如：0、1、2……)，1阶张量就是向量(比如：(1,3,4))，2阶张量就是矩阵，本来这几种形式是不相关的，但是都归为张量，是因为他们同时满足一些特性：1）可以用坐标系表示；2）在坐标变换中遵守同样的变换法则；3）有着相同的基本运算(如：加、减、乘、除、缩放、点积、对称……)

那么tensorflow可以理解为通过“流”的形式来处理张量的一种框架，是由google开发并开源，已经应用于google大脑项目开发

### tensorflow安装

[官网安装教程](https://www.tensorflow.org/install/)

推荐使用docker安装，免去配置cuda环境。

### 手写数字数据集获取

[github download](https://github.com/liqiang311/tf/tree/master/MNIST/MNIST_data)

在http://yann.lecun.com/exdb/mnist/ 可以下载手写数据集， http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 和 http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz ，下载解压后发现不是图片格式，而是自己特定的格式，为了说明这是什么样的数据，我写了一段程序来显示这些数字：

```c
/************************
 * author: SharEDITor
 * date:   2016-08-02
 * brief:  read MNIST data
 ************************/
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

unsigned char *lables = NULL;

/**
 * All the integers in the files are stored in the MSB first (high endian) format
 */
void copy_int(uint32_t *target, unsigned char *src)
{
    *(((unsigned char*)target)+0) = src[3];
    *(((unsigned char*)target)+1) = src[2];
    *(((unsigned char*)target)+2) = src[1];
    *(((unsigned char*)target)+3) = src[0];
}

int read_lables()
{
    FILE *fp = fopen("./train-labels.idx1-ubyte", "r");
    if (NULL == fp)
    {
        return -1;
    }
    unsigned char head[8];
    fread(head, sizeof(unsigned char), 8, fp);
    uint32_t magic_number = 0;
    uint32_t item_num = 0;
    copy_int(&magic_number, &head[0]);
    // magic number check
    assert(magic_number == 2049);
    copy_int(&item_num, &head[4]);

    uint64_t values_size = sizeof(unsigned char) * item_num;
    lables = (unsigned char*)malloc(values_size);
    fread(lables, sizeof(unsigned char), values_size, fp);

    fclose(fp);
    return 0;
}

int read_images()
{
    FILE *fp = fopen("./train-images.idx3-ubyte", "r");
    if (NULL == fp)
    {
        return -1;
    }
    unsigned char head[16];
    fread(head, sizeof(unsigned char), 16, fp);
    uint32_t magic_number = 0;
    uint32_t images_num = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    copy_int(&magic_number, &head[0]);
    // magic number check
    assert(magic_number == 2051);
    copy_int(&images_num, &head[4]);
    copy_int(&rows, &head[8]);
    copy_int(&cols, &head[12]);

    uint64_t image_size = rows * cols;
    uint64_t values_size = sizeof(unsigned char) * images_num * rows * cols;
    unsigned char *values = (unsigned char*)malloc(values_size);
    fread(values, sizeof(unsigned char), values_size, fp);

    for (int image_index = 0; image_index < images_num; image_index++)
    {
        // print the label
        printf("=========================================  %d  ======================================\n", lables[image_index]);
        for (int row_index = 0; row_index < rows; row_index++)
        {
            for (int col_index = 0; col_index < cols; col_index++)
            {
                // print the pixels of image
                printf("%3d", values[image_index*image_size+row_index*cols+col_index]);
            }
            printf("\n");
        }
        printf("\n");
    }

    free(values);
    fclose(fp);
    return 0;
}

int main(int argc, char *argv[])
{
    if (-1 == read_lables())
    {
        return -1;
    }
    if (-1 == read_images())
    {
        return -1;
    }
    return 0;
}
```

下载并解压出数据集文件train-images.idx3-ubyte和train-labels.idx1-ubyte放到源代码所在目录后，编译并执行：

```bash
gcc -o read_images read_images.c
./read_images
```

一共有60000个图片，从代码可以看出数据集里存储的实际就是图片的像素

### softmax模型

逻辑回归是用于解决二类分类问题(使用sigmoid函数)，而softmax模型是逻辑回归模型的扩展，用来解决多类分类问题。

softmax意为柔和的最大值，也就是如果某个zj大于其他z，那么这个映射的分量就逼近于1，其他的分量就逼近于0，从而将其归为此分类，多个分量对应的就是多分类，数学形式和sigmoid不同，如下：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/d6aa64b86d625770a3a7492c6d285cbae85e6ad4.png)

它的特点是，所有的softmax加和为1，其实它表示的是一种概率，即x属于某个分类的概率。

在做样本训练时，这里的xi计算方法是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/d04deffd60c4527f0b643443148154cbf7819d7c.png)

其中W是样本特征的权重，xj是样本的特征值，bi是偏置量。

详细来说就是：假设某个模型训练中我们设计两个特征，他们的值分别是f1和f2，他们对于第i类的权重分别是0.2和0.8，偏置量是1，那么

xi=f1*0.2+f2*0.8+1

如果所有的类别都计算出x的值，如果是一个训练好的模型，那么应该是所属的那个类别对应的softmax值最大

softmax回归算法也正是基于这个原理，通过大量样本来训练这里的W和b，从而用于分类的

### tensorflow的优点

tensorflow会使用外部语言计算复杂运算来提高效率，但是不同语言之间的切换和不同计算资源之间的数据传输耗费很多资源，因此它使用图来描述一系列计算操作，然后一起传给外部计算，最后结果只传回一次，这样传输代价最低，计算效率最高

举个例子：

```python
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
```

这里的x不是一个实际的x，而是一个占位符，也就是一个描述，描述成了二维浮点型，后面需要用实际的值来填充，这就类似于printf("%d", 10)中的占位符%d，其中第一维是None表示可无限扩张，第二维是784个浮点型变量

如果想定义可修改的张量，可以这样定义：

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

其中W的维度是[784, 10]，b的形状是[10]

有了这三个变量，我们可以定义我们的softmax模型：

```python
y = tf.nn.softmax(tf.matmul(x,W) + b)
```

这虽然定义，但是没有真正的进行计算，因为这只是先用图来描述计算操作

其中matmul是矩阵乘法，因为x的维度是[None, 784]，W的维度是[784, 10]，所以矩阵乘法得出的是[None, 10]，这样可以和向量b相加

softmax函数会计算出10维分量的概率值，也就是y的形状是[10]

### 数字识别模型实现

基于上面定义的x、W、b，和我们定义的模型：

```python
y = tf.nn.softmax(tf.matmul(x,W) + b)
```

我们需要定义我们的目标函数，我们以交叉熵(衡量预测用于描述真相的低效性)为目标函数，让它达到最小：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/12c891ad70fde8bcc7f71266e7429b596a9d4f05.png)

其中y'是实际分布，y是预测的分布，即：

```python
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```

利用梯度下降法优化上面定义的Variable：

```python
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```

其中0.01是学习速率，也就是每次对变量做多大的修正

按照上面的思路，最终实现的代码digital_recognition.py如下：

```python
# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
```

输出0.9199

解释一下

```python
flags.DEFINE_string('data_dir', './', 'Directory for storing data')
```

表示我们用当前目录作为训练数据的存储目录，如果我们没有提前下好训练数据和测试数据，程序会自动帮我们下载到./

```python
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
```

这句直接用库里帮我们实现好的读取训练数据的方法，无需自行解析

```python
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

这几行表示我们循环1000次，每次从训练样本里选取100个样本来做训练，这样我们可以修改配置来观察运行速度

最后几行打印预测精度，当调整循环次数时可以发现总训练的样本数越多，精度就越高

## 细解卷积神经网络

[原文](http://www.shareditor.com/blogshow?blogId=95)

深度学习首先要讲的就是卷积神经网络，因为卷积神经网络沿用了之前讲过的多层神经网络的具体算法，同时在图像识别领域得到了非常好的效果。本节介绍它的数学原理和一些应用中的问题解决方案，最后通过公式讲解样本训练的方法 

### 卷积运算

卷积英文是convolution(英文含义是：盘绕、弯曲、错综复杂)，数学表达是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/deb08fb89f7e20fc96b3ef59c1e2d9ac43f9f7fd.png)

上面连续的情形如果不好理解，可以转成离散的来理解，其实就相当于两个多项式相乘，如：`(x*x+3*x+2)(2*x+5)`，计算他的方法是两个多项式的系数分别交叉相乘，最后相加。用一句话概括就是：多项式相乘，相当于系数向量的卷积。

如果再不好理解，我们可以通俗点来讲：卷积就相当于在一定范围内做平移并求平均值。比如说回声可以理解为原始声音的卷积结果，因为回声是原始声音经过很多物体反射回来声音揉在一起。再比如说回声可以理解为把信号分解成无穷多的冲击信号，然后再进行冲击响应的叠加。再比如说把一张图像做卷积运算，并把计算结果替换原来的像素点，可以实现一种特殊的模糊，这种模糊其实是一种新的特征提取，提取的特征就是图像的纹路。总之卷积就是先打乱，再叠加。

下面我们在看上面的积分公式，需要注意的是这里是对τ积分，不是对x积分。也就是说对于固定的x，找到x附近的所有变量，求两个函数的乘积，并求和。

### 卷积神经网络

英文简称CNN，大家并不陌生，因为你可能见过DNN(深度神经网络)、RNN(循环神经网络)。CNN主要应用领域是图像处理，它本质上是一个分类器。

卷积神经网络为什么这么深得人心呢？因为在卷积神经网络的第一层就是特征提取层，也就是不需要我们自己做特征提取的工作，而是直接把原始图像作为输入，这带来了很大的便利，归根结底还是归功于卷积运算的神奇。

那么第一层是怎么利用卷积运算做特征提取的呢？我们还是通过图像处理的例子来说明。参考生物学的视觉结构，当人眼观察一个事物的时候，并不是每个视神经细胞感知所有看到的“像素”，而是一个神经细胞负责一小块视野，也就是说假设看到的全部视野是1000像素，而神经细胞有10个，那么一个神经细胞就负责比1000/10得到的平均值大一圈的范围，也就是200像素，一个细胞负责200个像素，10个细胞一共是2000个像素，大于1000个像素，说明有重叠。这和上面卷积运算的原理很像。用一张图来表示如下：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/39fcb11c4a7abd0259cf1e4620e297108a1272c0.png)

### 什么是卷积核

先看下面这张图，这是计算`5*5`矩阵中间的`3*3`部分的卷积值

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/e2b273b15a19c486c9e9c6a4efc48b6bcc6a34ff.png)

绿色部分是一个5*5的矩阵，标橙的部分说明正在进行卷积计算，×1表示算上这个单元的值，×0表示不计算，这样得出的结果1×1+1×0+1×1+0×0+1×1+1×0+0×1+0×0+1×1=4，这样计算出了第一个元素的卷积

我们继续让这个橙色部分移动并计算，最终会得到如下结果：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/486e08a5091d8b01a40d4dfa00372380c0b5f560.png)

那么这里的橙色(标记×1或×0)的矩阵(一般都是奇数行奇数列)就叫做**卷积核**，即

```
1 0 1
0 1 0
1 0 1
```

卷积计算实际上是一种对图像元素的矩阵变换，是提取图像特征的方法，多种卷积核可以提取多种特征。每一种卷积核生成的图像都叫做一个通道，这回也就理解了photoshop中“通道”的概念了吧

一个卷积核覆盖的原始图像的范围(上面就是5*5矩阵范围)叫做感受野(receptive field)，这个概念来自于生物学

### 多层卷积

利用一次卷积运算(哪怕是多个卷积核)提取的特征往往是局部的，难以提取出比较全局的特征，因此需要在一层卷积基础上继续做卷积计算 ，这也就是多层卷积。例如下面这个示意图：（无效图）

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/ef6642dd8a0fa8386ea196056e9c785067ff579a.png)

这实际上有四层卷积、三层池化、加上一层全连接，经过这些计算后得出的特征再利用常规的机器学习分类算法(如soft-max)做分类训练。上面这个过程是一个真实的人脸识别的卷积神经网络。

### 池化pooling

上面讲到了池化，池化是一种降维的方法。按照卷积计算得出的特征向量维度大的惊人，不但会带来非常大的计算量，而且容易出现过拟合，解决过拟合的办法就是让模型尽量“泛化”，也就是再“模糊”一点，那么一种方法就是把图像中局部区域的特征做一个平滑压缩处理，这源于局部图像一些特征的相似性(即局部相关性原理)。

具体做法就是对卷积计算得出的特征在局部范围内算出一个平均值(或者取最大值、或者取随机采样值)作为特征值，那么这个局部范围(假如是`10*10`)，就被压缩成了`1*1`，压缩了100倍，这样虽然更“模糊”了，但是也更“泛化”了。通过取平均值来池化叫做平均池化，通过取最大值来池化叫做最大池化。

### 卷积神经网络训练过程

上面讲解了卷积神经网络的原理，那么既然是深度学习，要学习的参数在哪里呢？

上面我们讲的卷积核中的因子(×1或×0)其实就是需要学习的参数，也就是卷积核矩阵元素的值就是参数值。一个特征如果有9个值，1000个特征就有9000个值，再加上多个层，需要学习的参数还是比较多的。

和多层神经网络一样，为了方便用链式求导法则更新参数，我们设计sigmoid函数作为激活函数，我们同时也发现卷积计算实际上就是多层神经网络中的Wx矩阵乘法，同时要加上一个偏执变量b，那么前向传到的计算过程就是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/c8b01577f6fae8e9adcd978423cc9cb556e1cd4d.png)

如果有更多层，计算方法相同

因为是有监督学习，所以模型计算出的y'和观察值y之间的偏差用于更新模型参数，反向传导的计算方法参考反向传导算法：

参数更新公式是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/edfe0cdb9eb76baae37adfba1e818fa87bb534e7.png)

偏导计算公式是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/9c69f67c8b11369b46d716fce99299389292b099.png)

其中a的计算公式是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/af369d2e36bbd979e7a739cc56ad00ea234bb5fd.png)

残差δ的计算公式是：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/3a06428d22c447221f8ca889a803438521595b41.png)

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/7afb4273d5f0e1fb1545b5f3776f4ab95b6c1c1d.png)

上面是输出层残差的推导公式和计算方法，下面是隐藏层残差的推导公式和计算方法

## 深究熵的概念和公式以及最大熵原理

[原文](http://www.shareditor.com/blogshow?blogId=99)

在机器学习算法中，最常用的优化方式就是使熵最大，那么到底什么是熵呢？很多文章告诉了我们概念和公式，但是很少有人讲到这些公式都是怎么来的，那么就让我们来深究一下这里面的奥秘 

### 熵

熵的英文是entropy，本来是一个热力学术语，表示物质系统的混乱状态。

我们都知道信息熵计算公式是`H(U)=-∑(p logp)`，但是却不知道为什么，下面我们深入熵的本源来证明这个公式

假设下图是一个孤立的由3个分子构成一罐气体

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/e3ce36e6180453f7ac23a3196516bbe6b1736765.png)

那么这三个分子所处的位置有如下几种可能性：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/d417449e183cf803851eb5618e9ff93ed8efc5b6.png)

图中不同颜色表示的是宏观状态(不区分每个分子的不同)，那么宏观状态一共有4种，而微观状态(每一种组合都是一种微观状态)一共有2^3=8种

再来看4个分子的情况

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/3293ab5169c228adbd17b2a249e458840e68351b.png)

这时，宏观状态一共有5种，而微观状态一共有2^4=16种

事实上分子数目越多，微观数目会成指数型增长

这里面提到的宏观状态实际上就是熵的某一种表现，如果气体中各种宏观状态都有，那么熵就大，如果只存在一种宏观状态，那么熵就很小，如果把每个分子看做状态的形成元素，熵的计算就可以通过分子数目以某种参数求对数得到，这时我们已经了解了为什么熵公式中是对数关系

上面我们描述的一个系统(一罐气体)，假如我们有两罐气体，那么它们放在一起熵应该是可以相加的(就像上面由四种状态加了一个状态的一个分子变成5个状态)，即可加性，而微观状态是可以相乘的(每多一个分子，微观状态就会多出n-1种)，即相乘法则

综上，我们可以得出熵的计算公式是`S=k ln Ω`，其中k是一个常数，叫做玻尔兹曼常数，`Ω`是微观状态数，这个公式也满足了上面的可加性和相乘法则，即`S1+S2=k ln (Ω1Ω2)`

### 最大熵

在机器学习中我们总是运用最大熵原理来优化模型参数，那么什么样的熵是最大熵，为什么它就是最优的

这还是要从物理学的原理来说明，我们知道当没有外力的情况下气体是不断膨胀的而不会自动收缩，两个温度不同的物体接触时总是从高温物体向低温物体传导热量而不可逆。我们知道宏观状态越多熵越大，那么气体膨胀实际上是由较少的宏观状态向较多的宏观状态在变化，热传导也是一样，如此说来，一个孤立系统总是朝着熵增加的方向变化，熵越大越稳定，到最稳定时熵达到最大，这就是熵增原理

换句话说：熵是孤立系统的无序度的量度，平衡态时熵最大

将熵增原理也可以扩大到一切自发过程的普遍规律，比如如果你不收拾屋子，那么屋子一定会变得越来越乱而不会越来越干净整洁，扩大到统计学上来讲，屋子乱的概率更大，也就是说孤立系统中一切实际过程总是从概率小的状态向概率大的状态的转变过程，并且不可逆

### 信息熵

1948年，信息论之父香农发表的《通信的数学理论》中提出了“信息熵”的概念，从此信息熵对通信和计算机行业产生了巨大的影响。那么他到底说了些什么呢？

一个随机变量`ξ`有`A1、A2、A3……`共n个不同的结果，每个结果出现的概率是`p1、p2、p3……`，那么我们把`ξ`的不确定度定义为信息熵，参考上面物理学熵的定义，`A1、A2、A3……`可以理解为不同的微观状态，那么看起来信息熵应该是`log n`喽？不然，因为这个随机变量`ξ`一次只能取一个值而不是多个值，所以应该按概率把`ξ`劈开，劈成n份，每份的微观状态数分别是`1/p1`、`1/p2`、`1/p3`……，这样这n份的熵分别是`log 1/p1`、`log 1/p2`、`log 1/p3`……，再根据熵的可加性原理，得到整体随机变量`ξ`的信息熵是`∑(p log 1/p)`，即`H(ξ) = -∑(p log p)`

### 最大熵原理

继续看上面的信息熵公式，从公式可以看出，出现各种随机结果可能性越大，不确定性就越大，熵就越大。相反，如果只可能出现一种结果，那么熵就为0，因为这时p=1，-∑(p log p)=0

举个例子，投1000次硬币，最有可能的概率是正面1/2，负面1/2，因此熵是H(X) = -(0.5log0.5+0.5log0.5) = -0.5*math.log(2,1/2)*2 = -0.5*-1*2 = 1

那么假设只会出现正面，熵是H(X) = -1log1 = 0

实际上哪种是最符合实际情况的呢？显然是第一种，这就是最大熵模型的原理：在机器学习中之所以优化最大熵公式能训练出最接近正确值的参数值，是因为这是“最符合实际”的可能。换句有哲理的话说：熵越大越趋向于自然，越没有偏见

### 最大熵模型

机器学习中用到的最大熵模型是一个定义在条件概率分布P(Y|X)上的条件熵。其中X、Y分别对应着数据的输入和输出，根据最大熵原理，当熵最大时，模型最符合实际情况。那么这个条件熵应该是什么样的呢？

条件概率分布P(Y|X)上的条件熵可以理解为在X发生的前提下，Y发生所“新”带来的熵，表示为H(Y|X)，那么有

```
H(Y|X) = H(X,Y) - H(X)
```

其中H(X,Y)表示X、Y的联合熵，表示X、Y都发生时的熵，H(Y|X)的计算公式推导如下：

![image](http://shareditor-shareditor.oss-cn-beijing.aliyuncs.com/dynamic/a026ef2db35f1a14489e49e1f241fe7123229eaf.png)

因此我们在机器学习中想方设法优化的就是这个东东，由于这里的p(x,y)无法统计，因此我们转成p(x)p(y|x)，这样得到公式如下：

```
H(Y|X) = -∑p(x)p(y|x)log p(y|x)
```

那么机器学习训练的过程实际就是求解p(y|x)的过程，其中p(x)可以通过x的最大似然估计直接求得

### 总结

至此，我们介绍完了熵的概念和公式以及最大熵原理和最大熵模型公式的由来，总之，熵来源于热力学，扩展于信息论，应用在机器学习领域，它表达的是一种无序状态，也是最趋向于自然、最符合实际的情况。为了更深入感受最大熵模型的魅力，后续我会把最大熵模型的应用不断渗透到机器学习教程的具体算法中

## 逻辑回归公式的数学推导

[原文](http://www.shareditor.com/blogshow?blogId=102)

机器学习中一些重要的公式，比如逻辑回归概率公式，多数情况下我们知道何时拿来用，但是它们都是怎么得来的呢，本节让我们详细探讨下 

### 逻辑回归中的数学推导

逻辑回归模型是基于这样的逻辑分布得出的模型

F(x) = 1/(1+e^x)

由此也得出了二项逻辑回归分布是：

P(Y=1|x) = e^(wx+b)/(1+e^(wx+b))

P(Y=0|x) = 1/(1+e^(wx+b))

也得出了多项逻辑回归分布是：

P(Y=k|x) =  e^(wx)/(1+∑e^(wx))

那么这个 1/(1+e^x)到底是怎么来的呢？我们来证明这一公式

首先假设0、1分布当Y=1的概率为

P(Y=1) = φ

那么

P(Y=0) = 1-φ

把他们变成统一的形式可以是：

P(y; φ) = φ^y (1-φ)^(1-y)

解释一下，这里如果y=0，那么前一项是1，就是p(y;φ) = 1-φ，而如果y=1，那么后一项就为1，就是p(y;φ) = φ

下面继续推导，我们知道有一个等式：a = e^(ln a)

那么把右面改成指数形式如下：

P(y; φ) = φ^y (1-φ)^(1-y) = e^(log(φ^y (1-φ)^(1-y))) = e ^ (y logφ + (1-y) log(1-φ)) = e^(log(φ/(1-φ))y+log(1-φ))

因为任何分布都能写成一种指数形式的分布：

p(y; η) = b(y) e^(ηT(η) - a(η))

这里面我们按照对应关系得出η=log(φ/(1-φ))，那么φ/(1-φ) = e^η，那么解出φ = 1/(1+e^(-η))

所以得出P(Y=1) = φ =  1/(1+e^(-η))

大功告成，终于知道逻辑回归公式是怎么来的了

## R语言特征工程实战

特征工程是机器学习过程中和模型训练同样重要的部分，特征如何提取、如何处理、如何选择、如何使用都是特征工程的范畴，特征工程需要具备数据分析的能力，那些称为数据科学家的人一定是有很强的特征工程能力的人。R语言是大数据领域的主流语言之一，本文主要介绍用R语言的图形工具做特征工程的实战方法

### R语言介绍

熟悉R语言的朋友请直接略过。R语言是贝尔实验室开发的S语言(数据统计分析和作图的解释型语言)的一个分支，主要用于统计分析和绘图，R可以理解为是一种数学计算软件，可编程，有很多有用的函数库和数据集。

### R的安装和使用

在 https://mirrors.tuna.tsinghua.edu.cn/CRAN/ 下载对应操作系统的安装包安装。安装好后单独创建一个目录作为工作目录(因为R会自动在目录里创建一些有用的隐藏文件，用来存储必要的数据)

执行

```
R
```

即可进入R的交互运行环境

简单看一个实例看一下R是如何工作的：

```
[root@centos:~/Developer/r_work $] R

R version 3.3.1 (2016-06-21) -- "Bug in Your Hair"
Copyright (C) 2016 The R Foundation for Statistical Computing
Platform: x86_64-apple-darwin13.4.0 (64-bit)

> x <- c(1,2,3,4,5,6,7,8,9,10)
> y <- x*x
> plot(x,y,type="l")
>
```

以上看得出我们画了y = x^2的曲线

R语言的语法和C类似，但是稍有不同，R语言里向量和矩阵的操作和python的sci-learn类似，但是稍有不同：

1. R的赋值语句的符号是"<-"而不是"="

2. R里的向量用c()函数定义，R里没有真正的矩阵类型，矩阵就是一系列向量组成的list结构

有时候如果我们想要加载一个库发现没有安装，就像这样：

```
> library(xgboost)
Error in library(xgboost) : 不存在叫‘xgboost’这个名字的程辑包
```

那么就这样来安装：

```
> install.packages("xgboost")
```

输入后会提示选择下载镜像，选择好后点ok就能自动安装完成，这时就可以正常加载了：

```
> library(xgboost)
>
```

想了解R语言的全部用法，推荐《权威的R语言入门教程《R导论》-丁国徽译.pdf》，请自行下载阅读，也可以继续看我下面的内容边用边学

### 特征工程

按我的经验，特征工程就是选择和使用特征的过程和方法，这个说起来容易，做起来真的不易，想要对实际问题设计一套机器学习方法，几乎大部分时间都花在了特征工程上，相反最后的模型开发花不了多长时间(因为都是拿来就用了)，再有需要花一点时间的就是最后的模型参数调优了。花费时间排序一般是：特征工程>模型调参>模型开发

### Titanic数据集特征工程实战

Titanic数据集是这样的数据：Titanic(泰坦尼克号)沉船灾难死亡了很多人也有部分人成功得救，数据集里包括了这些字段：乘客级别、姓名、性别、年龄、船上的兄弟姐妹数、船上的父母子女数、船票编号、票价、客舱编号、登船港口、是否得救。

我们要做的事情就是把Titanic数据集中部分数据作为训练数据，然后用来根据测试数据中的字段值来预测这位乘客是否得救

### 数据加载

训练数据可以在 https://www.kaggle.com/c/titanic/download/train.csv 下载，测试数据可以在 https://www.kaggle.com/c/titanic/download/test.csv 下载

下面开始我们的R语言特征工程，创建一个工作目录r_work，下载train.csv和test.csv到这个目录，看下里面的内容：

```
[root@centos:~/Developer/r_work $] head train.csv
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
```

我们看到文件内容是用逗号分隔的多个字段，第一行是schema，第二行开始是数据部分，其中还有很多空值，事实上csv就是Comma-Separated Values，也就是用“逗号分隔的数值”，它也可以用excel直接打开成表格形式

R语言为我们提供了加载csv文件的函数，如下：

```
> train <- read.csv('train.csv', stringsAsFactors = F)
> test <- read.csv('test.csv', stringsAsFactors = F)
```

如果想看train和test变量的类型，可以执行：

```
> mode(train)
[1] "list"
```

我们看到类型是列表类型

如果想预览数据内容，可以执行：

```
> str(train)
'data.frame':   891 obs. of  12 variables:
 $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
 $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
 $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
 $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
 $ Sex        : chr  "male" "female" "female" "female" ...
 $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
 $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
 $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
 $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
 $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
 $ Cabin      : chr  "" "C85" "" "C123" ...
 $ Embarked   : chr  "S" "C" "S" "S" ...
```

可以看到其实train和test变量把原始的csv文件解析成了特定的数据结构，train里有891行、12列，每一列的字段名、类型以及可能的值都能预览到

因为test数据集也是真实数据的一部分，所以在做特征工程的时候可以把test和train合并到一起，生成full这个变量，后面我们都分析full：

```
> library('dplyr')
> full  <- bind_rows(train, test)
```

### 头衔特征的提取

因为并不是所有的字段都应该用来作为训练的特征，也不是只有给定的字段才能作为特征，下面我们开始我们的特征选择工作，首先我们从乘客的姓名入手，我们看到每一个姓名都是这样的结构："名字, Mr/Mrs/Capt等. 姓"，这里面的"Mr/Mrs/Capt等"其实是一种称谓(Title)，虽然人物的姓名想必和是否得救无关，但是称谓也许和是否得救有关，我们把所有的Title都筛出来：

```
> table(gsub('(.*, )|(\\..*)', '', full$Name))

        Capt          Col          Don         Dona           Dr     Jonkheer
           1            4            1            1            8            1
        Lady        Major       Master         Miss         Mlle          Mme
           1            2           61          260            2            1
          Mr          Mrs           Ms          Rev          Sir the Countess
         757          197            2            8            1            1
```

解释一下，这里面的full$Name表示取full里的Name字段的内容，gsub是做字符串替换，table是把结果做一个分类统计(相当于group by title)，得出数目

通过结果我们看到不同Title的人数目差别比较大

我们把这个Title加到full的属性里：

```
> full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
```

这时我们可以按性别和title分两级统计(相当于group by sex, title):

```
> table(full$Sex, full$Title)

         Capt Col Don Dona  Dr Jonkheer Lady Major Master Miss Mlle Mme  Mr Mrs
  female    0   0   0    1   1        0    1     0      0  260    2   1   0 197
  male      1   4   1    0   7        1    0     2     61    0    0   0 757   0

          Ms Rev Sir the Countess
  female   2   0   0            1
  male     0   8   1            0
```

为了让这个特征更具有辨别性，我们想办法去掉那些稀有的值，比如总次数小于10的，我们都把title改成“Rare Title”

```
> rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
> full$Title[full$Title %in% rare_title]  <- 'Rare Title'
```

同时把具有相近含义的title做个归一化

```
> full$Title[full$Title == 'Mlle']        <- 'Miss'
> full$Title[full$Title == 'Ms']          <- 'Miss'
> full$Title[full$Title == 'Mme']         <- 'Mrs'
```

这回我们看下title和是否得救的关系情况

```
> table(full$Title, full$Survived)

               0   1
  Master      17  23
  Miss        55 130
  Mr         436  81
  Mrs         26 100
  Rare Title  15   8
```

还不够直观，我们可以通过马赛克图来形象的看：

```
> mosaicplot(table(full$Sex, full$Title), shade=TRUE)
```

这回看出比例情况的差异了，比如title为Mr的死亡和得救的比例比较明显，说明这和是否得救关系密切，title作为一个特征是非常有意义的

这样第一个具有代表意义的特征就提取完了

### 家庭成员数特征的提取

看过电影的应该了解当时的场景，大家是按照一定秩序逃生的，所以很有可能上有老下有小的家庭会被优先救援，所以我们统计一下一个家庭成员的数目和是否得救有没有关系。

为了计算家庭成员数目，我们只要计算父母子女兄弟姐妹的数目加上自己就可以，所以：

```
> full$Fsize <- full$SibSp + full$Parch + 1
```

下面我们做一个Fsize和是否得救的图像

```
> library("ggplot2")
> library('ggthemes')
> ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) + geom_bar(stat='count', position='dodge') + scale_x_continuous(breaks=c(1:11)) + labs(x = 'Family Size') + theme_few()
```

我们先解释一下上面的ggplot语句

第一个参数full[1:891,]表示我们取全部数据的前891行的所有列，取891是因为train数据一共有891行

aes(x = Fsize, fill = factor(Survived))表示坐标轴的x轴我们取Fsize的值，这里的fill是指用什么变量填充统计值，factor(Survived)表示把Survived当做一种因子，也就是只有0或1两种“情况”而不是数值0和1，这样才能分成红绿两部分统计，不然如果去掉factor()函数包裹就会像这个样子(相当于把0和1加了起来)：

这里的“+”表示多个图层，是ggplot的用法

geom_bar就是画柱状图，其中stat='count'表示统计总数目，也就是相当于count(*) group by factor(Survived)，position表示重叠的点放到什么位置，这里设置的是“dodge”表示规避开的展示方式，如果设置为"fill"就会是这样的效果：

scale_x_continuous(breaks=c(1:11))就是说x轴取值范围是1到11，labs(x = 'Family Size')是说x轴的label是'Family Size'，theme_few()就是简要主题

下面我们详细分析一下这个图说明了什么事情。我们来比较不同家庭成员数目里面成功逃生的和死亡的总数的比例情况可以看出来：家庭人数是1或者大于4的情况下红色比例较大，也就是死亡的多，而人数为2、3、4的情况下逃生的多，因此家庭成员数是一个有意义的特征，那么把这个特征总结成singleton、small、large三种情况，即：

```
> full$FsizeD[full$Fsize == 1] <- 'singleton'
> full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
> full$FsizeD[full$Fsize > 4] <- 'large'
```

再看下马赛克图：

```
> mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)
```

从图中可以看出差异明显，特征有意义

### 模型训练

处理好特征我们就可以开始建立模型和训练模型了，我们选择随机森林作为模型训练。首先我们要把要作为factor的变量转成factor：

```
> factor_vars <- c('PassengerId','Pclass','Sex','Embarked','Title','FsizeD')
> full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
```

然后我们重新提取出train数据和test数据

```
> train <- full[1:891,]
> test <- full[892:1309,]
```

接下来开始训练我们的模型

```
> library('randomForest')
> set.seed(754)
> rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Embarked + Title + FsizeD, data = train)
```

下面画出我们的模型误差变化：

```
> plot(rf_model, ylim=c(0,0.36))
> legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
```

图像表达的是不同树个数情况下的误差率，黑色是整体情况，绿色是成功获救的情况，红色是死亡的情况，可以看出通过我们给定的几个特征，对死亡的预测误差更小更准确

我们还可以利用importance函数计算特征重要度：

```
> importance(rf_model)
         MeanDecreaseGini
Pclass          40.273719
Sex             53.240211
Embarked         8.566492
Title           85.214085
FsizeD          23.543209
```

可以看出特征按重要程度从高到底排序是：Title > Sex > Pclass > FsizeD > Embarked

### 数据预测

有了训练好的模型，我们可以进行数据预测了

```
> prediction <- predict(rf_model, test)
```

这样prediction中就存储了预测好的结果，以0、1表示

为了能输出我们的结果，我们把test数据中的PassengerId和prediction组合成csv数据输出

```
> solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
> write.csv(solution, file = 'solution.csv', row.names = F)
```

最终的solution.csv的内容如下：

```
[root@centos:~/Developer/r_work $] head solution.csv
"PassengerID","Survived"
"892","0"
"893","1"
"894","0"
"895","0"
"896","1"
"897","0"
"898","1"
"899","0"
"900","1"
```

本文部分内容参考：https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic/comments

## 看数据科学家是如何找回丢失的数据的

[原文1](http://www.shareditor.com/blogshow?blogId=107)

[原文2](http://www.shareditor.com/blogshow?blogId=108)

在做特征工程过程中，经常遇到某些样本缺失了某个特征的值，影响我们的机器学习过程，如果是较小的样本集数据科学家可不会直接舍弃这些样本，而是利用有效的手段把丢失的数据找回来，他们是怎么找回的呢？我接下来的几篇文章会通过实例讲几种缺失值补全的方法 

### 补全数据的纯手工方案

我们以泰坦尼克号数据集为例

先重温一下这个数据集里面都有哪些字段：

PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

分别表示：

样本编号、是否得救、乘客级别、姓名、性别、年龄、船上的兄弟姐妹数、船上的父母子女数、船票编号、票价、客舱编号、登船港口。

我们检查一下Embarked这个字段哪些乘客是缺失的：

```
> full$PassengerId[full$Embarked == '']
[1] 62  830
```

看来在这1309位乘客中PassengerId 为62和830的乘客缺失了Embarked字段值，那么我们如何来补全这个数据呢？我们分析一下哪个字段可能和Embarked(登船港口)的值有关，我们猜测票价有可能和Embarked有关，但是不同级别的票价一定又是不一样的，那么我们可以看一下不同级别票价的统计规律，庆幸的是Embarked只有三个取值：C Q S分别表示C = Cherbourg; Q = Queenstown; S = Southampton

我们先来看一下62  830的票价和乘客级别是多少：

```
> full[c(62, 830), 'Fare']
[1] 80 80
> full[c(62, 830), 'Pclass']
[1] 1 1
```

等级都是1级，票价都是80

现在我们再看下这三个港口对应不同级别的乘客平均票价是多少，在此之前我们先排除掉62  830这两位乘客的数据：

```
> library("dplyr")
> embark_fare <- full %>% filter(PassengerId != 62 & PassengerId != 830)
```

下面我们利用强大的ggplot2画出盒图(boxplot)，首先说一下什么是盒图，盒图由五个数值点组成：最小值(min)，下四分位数(Q1)，中位数(median)，上四分位数(Q3)，最大值(max)。也可以往盒图里面加入平均值(mean)。下四分位数、中位数、上四分位数组成一个“带有隔间的盒子”。上四分位数到最大值之间建立一条延伸线，这个延伸线成为“胡须(whisker)”。盒图用来反映离散数据的分布情况。

下面我们画出不同Embarked、不同等级乘客对应的Fare的盒图

```
> library("ggplot2")
> library('ggthemes')
> ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +geom_boxplot()+geom_hline(aes(yintercept=80),colour='red', linetype='dashed', lwd=2)+theme_few()
```

讲解一下这个命令，`geom_boxplot`表示画盒图，`geom_hline`表示沿着横轴方向画线，如果想沿着纵轴那么就用`geom_vline`，lwd表示线宽

为了能找到和62, 830两位乘客相似的情况，单独把Fare为80的位置画出了一条横线，用来参照。我们发现Pclass=1的乘客Fare均值最接近80的是C港口，因此我们把这两位乘客的Embarked就赋值为C：

```
> full$Embarked[c(62, 830)] <- 'C'
```

当然我们还可以画这样一张图来看待这个事情：

```
> ggplot(full[full$Pclass == '1' & full$Embarked == 'C', ],
+ aes(x = Fare)) +
+ geom_density(fill = '#99d6ff', alpha=0.4) +
+ geom_vline(aes(xintercept=median(Fare, na.rm=T)),
+ colour='red', linetype='dashed', lwd=1) +
+ geom_vline(aes(xintercept=80),colour='green',linetype='dashed', lwd=1) +
+ theme_few()
```

讲解一下：这里选择Pclass==1，Embarked == 'C'的数据，画出了概率密度曲线，同时把Fare的均值画了一条红色的竖线，也在Fare=80的位置画了一条绿色的竖线作为参照，可以直观看出均值和80很接近

本文部分内容参考：https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic/comments