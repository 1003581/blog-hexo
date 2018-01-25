---
title: 生成对抗网络GAN
date: 2018-01-19 23:30:00
tags: 深度学习
categories: 深度学习
---

生成对抗网络GAN
<!-- more -->

# 理论学习

## 第一篇论文 Generative Adversarial Networks

GAN提出者的第一篇论文下载地址 [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)，[PDF](https://arxiv.org/pdf/1406.2661.pdf)，该论文的讲解[http://blog.csdn.net/sallyxyl1993/article/details/64123922](http://blog.csdn.net/sallyxyl1993/article/details/64123922)。

![](http://upload-images.jianshu.io/upload_images/5952841-3602de80374dfd54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- G 生成器 生成器通过噪音的输入得到了伪造的样本。
- D 判别器 判别器需要判别输入的样本是真实的还是伪造的。
- D的训练是有监督的。D判别一个样本是否真实的概率最优时为0.5。此时判别器无法进行判别，G生成的模型已经以假乱真。
- 需要同时训练G和D。

翻译原文如下：

**3 Adversarial nets**

- [png1](http://upload-images.jianshu.io/upload_images/5952841-dfcf7935cdbc8517.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [png2](http://upload-images.jianshu.io/upload_images/5952841-0ca98b88e41f7383.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [png3](http://upload-images.jianshu.io/upload_images/5952841-76bdb19dba9a3e33.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**3 对抗网络**

当模型都是多层感知器时，对抗网络框架非常容易直接应用。为了在数据$x$上学习生成器的分布$p_g$，我们预先定义了输入噪音变量$p_z(z)$，然后实现了一个数据空间的映射$G(z; \theta _g)$，G是一个由参数为$\theta _g$的多层感知器代表的可微函数。我们同时定义了第二个多层感知器$D(x, \theta d)$，该感知器输入一个单一的标量。$D(x)$代表了输入x来自真实数据而不是$p_g$的概率。我们训练D，使得分配正确训练样本和G生成样本的标签的概率最大化。我们同时训练G，使得$\log(1-D(G(z)))$最小化。

换句说，D和G在如下的价值函数$V(G, D)$中扮演最大最小化游戏。

![](http://upload-images.jianshu.io/upload_images/5952841-2303df9ec01616d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在下一节中，我们对对抗网络进行了理论分析，实际上表明了训练准则允许我们在给定的能力范围内通过G和D来恢复数据的分布。图1为该方法不太正式但十分数学化的解释。在实践中，我们必须使用可迭代的数学化的方法去实现这个游戏。在训练的内部循环中去完成D的优化在数学上是被禁止的，而且在有限的数据集上会表现过拟合。这将导致判别器D一直保持在其最优附近，而生成器G变化十分缓慢。这种策略类似于SML/PCD，从一个马尔可夫链到下一个的训练过程中，样本保持不变，以避免在马尔可夫链的学习内部循环中燃烧浪费。正式算法在算法1中进行展现。

在实践中，公式1或许没有为G提供足够的梯度去更好地学习。学习的早期，生成器G是较差的，判别器D可以以较高的置信度去拒绝生成式样本，因为他们之间区别明显。这种情况下，$\log(1-D(G(z)))$比较饱和（靠近0）。相比于训练G时的最小化$\log(1-D(G(z)))$，我们选择最大化$\logD(G(z))$去训练G。这个目标函数导致了G和D动力学的相同的固定点，但在学习的早期提供了更强的梯度。

图1：通过同时更新判别分布（D，蓝色，虚线）来训练生成对抗网络，以便从数据生成分布（黑色，虚线）$p_x$ 与生成分布$p_g$ (G)（绿色，实线）之间进行判别。下面的横线是z被采样的区域，这种情况下是均匀的。上面的水平线是x中的一部分。向上箭头显示了映射$x=G(z)$如何在变换后的样本上施加非线性变换$p_g$。G在$p_g$高密度区域收缩，在$p_g$低密度区域扩大。（a）图显示了在接受收敛的一个对抗对：$p_g$类似于$p_{data}$，并且D在部分区域准确分类。（b）图在算法D的内部循环中被训练来区别样本和数据，收敛到$D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}$。（c）图显示了G更新后，D的梯度已经引导$G(z)$流向更可能被分类为实际样本的区域。（d）显示训练一定步数后，如果G和D拥有足够的能力，他们会达到一个都无法提升的点，因为$p_{g}=p_{data}$，判别器无法区分两种分布，此时$D(x)=1/2$。

训练步骤

![](http://upload-images.jianshu.io/upload_images/5952841-ebfe75a4ee845b86.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Ian Goodfellow NIPS 2016 Tutorial

原文如下：[http://www.sohu.com/a/121189842_465975](http://www.sohu.com/a/121189842_465975)

[Generative Modeling](http://upload-images.jianshu.io/upload_images/5952841-21584f4f74e8fde7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 密度分布：在不了解事件概率分布的情况下，先假设随机分布，然后通过数据观测来确定真正的概率密度是怎么样的。
- 样本生成：通过训练样本数据，训练模型得到类似的样本。

[RoadMap](http://upload-images.jianshu.io/upload_images/5952841-01dd5f072589d89a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

路线图

- 为什么学习生成模型？
- 生成模型如何工作？GANs相比其他如何？
- GANs如何工作？
- 技巧和窍门
- 研究前沿
- 将GANs与其他方法合并

**为什么要学习生成模型？**

[Why study generative models](http://upload-images.jianshu.io/upload_images/5952841-4a1a795c24e5d084.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 这是一种对我们处理高维数据和复杂概率分布的能力很好的检测；
- 为了未来的规划或模拟型强化学习做好理论准备（所谓的 model-free RL）
- 缺乏数据
    - 半监督学习中使用
- 多模型输出
- 实际生成模型的任务

实际使用如下

![](http://upload-images.jianshu.io/upload_images/5952841-eb1de3031b47d44e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**生成模型是如何工作的？GAN 跟其它模型比较有哪些优势？**

该怎么创造生成模型呢？这涉及到概率领域一个方法：最大似然估计。

[Maximum Likelihood](http://upload-images.jianshu.io/upload_images/5952841-cb6ce828915242c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**生成模型大家族**

[Taxonomy of Generative Models](http://upload-images.jianshu.io/upload_images/5952841-5efd5787b507f39f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

几个模型示意图

- [Fully Visible Belief Nets](http://upload-images.jianshu.io/upload_images/5952841-6374c62a4d645e71.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [WaveNet](http://upload-images.jianshu.io/upload_images/5952841-6bb2c882b94cdd10.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [Change of Variables](http://upload-images.jianshu.io/upload_images/5952841-0fc3750e7368171f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [Variational Autoencoder](http://upload-images.jianshu.io/upload_images/5952841-b816a85241b48aa9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [Boltzmann Machines](http://upload-images.jianshu.io/upload_images/5952841-1319fe4f8a7f5f0a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

相比这些模型，GAN的特点如下：

[GANs](http://upload-images.jianshu.io/upload_images/5952841-384e5440d7b8f931.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 使用了 latent code（用以表达 latent dimension、控制数据隐含关系等等）；
- 数据会逐渐统一；
- 不需要马尔可夫链；
- 被认为可以生成最好的样本（当然，ian 本人也说，这事儿没法衡量什么是「好」或「不好」）。

**GAN是如何工作的？**

生成对抗模型GAN由判别模型和生成模型组成，如图

[Adversarial Nets Framework](http://upload-images.jianshu.io/upload_images/5952841-64b7be059f96c943.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

给定一批真实数据，A需要造假，B需要鉴伪。

一个造假一流的A则是我们想要的生成模型。

[Generator Network](http://upload-images.jianshu.io/upload_images/5952841-36b6ee0e445f3413.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里就是我们的生成模型了。如图所示，它把噪声数据 z（也就是我们说的假数据）通过生成模型 G，伪装成了真实数据 x。（当然，因为 GAN 依旧是一个神经网络，你的生成模型需要是可微的（differentiable））

[Training Procedure](http://upload-images.jianshu.io/upload_images/5952841-93ccb2765fd747a7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

训练的过程也非常直观，你可以选择任何类 SGD 的方法（因为 A 和 B 两个竞争者都是可微的网络）。并且你要同时训练两组数据：一组真实的训练数据和一组由骗子 A 生成的数据。当然，你也可以一组训练每跑一次时，另一组则跑 K 次，这样可以防止其中一个跟不上节奏。

[Minimax Game](http://upload-images.jianshu.io/upload_images/5952841-7535cb863093f5ff.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同样，既然要用类 SGD 优化，我们就需要一个目标函数（objective function）来判断和监视学习的成果。在这里，J(D) 代表判别网络（也就是警察 B）的目标函数——一个交叉熵（cross entropy）函数。其中左边部分表示 D 判断出 x 是真 x 的情况，右边部分则表示 D 判别出的由生成网络 G（也就是骗子）把噪音数据 z 给伪造出来的情况。

这样，同理，J(G) 就是代表生成网络的目标函数，它的目的是跟 D 反着干，所以前面加了个负号（类似于一个 Jensen-Shannon（JS）距离的表达式）。

这其实就是我们熟悉的最小最大博弈（minimax game）：两个人的零和博弈，一个想最大，另一个想最小。那么，我们要找的均衡点（也就是纳什均衡）就是 J(D) 的鞍点（saddle point）。

![判别策略 Discriminator Strategy](http://upload-images.jianshu.io/upload_images/5952841-89b8b9a481bc8e74.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如图所示，我们手上有真实数据（黑色点，data）和模型生成的伪数据（绿色线，model distribution，是由我们的 z 映射过去的）（画成波峰的形式是因为它们都代表着各自的分布，其中纵轴是分布，横轴是我们的 x）。而我们要学习的 D 就是那条蓝色的点线，这条线的目的是把融在一起的 data 和 model 分布给区分开。写成公式就是 data 和 model 分布相加做分母，分子则是真实的 data 分布。

我们最终要达到的效果是：D 无限接近于常数 1/2。换句话说就是要 Pmodel 和 Pdata 无限相似。这个时候，我们的 D 分布再也没法分辨出真伪数据的区别了。这时候，我们就可以说我们训练出了一个炉火纯青的造假者（生成模型）。

于是，最终我们得到的应该是如下图的结果：

[Final Model](http://upload-images.jianshu.io/upload_images/5952841-cd8a93ae4ac7c391.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

蓝色断点线是一条常数线（1/2），黑色与绿色完美重合了。

但是，这也是有问题的：我们的生成模型跟源数据拟合之后就没法再继续学习了（因为常数线 y = 1/2 求导永远为 0）。

[Non-Saturating Game](http://upload-images.jianshu.io/upload_images/5952841-9a2f7c5984d78fb6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了解决这个问题，除了把两者对抗做成最小最大博弈，还可以把它写成非饱和（Non-Saturating）博弈：

也就是说用 G 自己的伪装成功率来表示自己的目标函数（不再是直接拿 J(D) 的负数）。这样的话，我们的均衡就不再是由损失（loss）决定的了。J(D) 跟 J(G) 没有简单粗暴的相互绑定，就算在 D 完美了以后，G 还可以继续被优化。

在应用上，这套 GAN 理论最火的构架是 DCGAN（深度卷积生成对抗网络/Deep Convolutional Generative Adversarial Network）。熟悉卷积神经网络（CNN）的同学对此应该不会陌生，这其实就是一个反向的 CNN。

![DCGAN Architecture](http://upload-images.jianshu.io/upload_images/5952841-0b287fba070da2b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

DCGAN目的是创造图片，其实就类似于把一组特征值慢慢恢复成一张图片。所以相比于CNN，两者的比较就是：在每一个卷积层，CNN是把大图片的重要特征提取出来，一步一步地减小图片尺寸。而DCGAN是把小图片（小数组）的特征放大，并排列成新图片。这里，作为DCGAN的输入的最初的那组小数据就是我们刚刚讲的噪声数据。

[Transposed-convolution](http://upload-images.jianshu.io/upload_images/5952841-e9d9fe9a6c60c354.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面几页幻灯片给出了一些案例效果。

- [DCGANs for LSUN Bedrooms](http://upload-images.jianshu.io/upload_images/5952841-556f970428f6da90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [Vector Space Arithmetic](http://upload-images.jianshu.io/upload_images/5952841-5a1c5ad4c3afdc39.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 戴墨镜的男人 - 不戴墨镜的男人 + 不戴墨镜的女人 = 戴墨镜的女人

刚刚我们讲了 J(G) 的两种方法，它们给 GAN 带来了两种不同的求解方式。

除了以上提到的两种书写 J(G) 的方法，最近的几篇论文又提出了一些新观点。熟悉统计学科的同学应该都知道，说到 JS 距离（也就是刚刚在 minimax 处提到的），就会想到 KL 距离；它们都是统计学科中常用的散度（divergence）方程。散度方程是我们用来创造目标函数的基础。

![D KL](http://upload-images.jianshu.io/upload_images/5952841-adc57487c8fd5916.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240)

GAN 作为一个全由神经网络构造并通过类 SGD 方法优化的模型，目标函数的选择是至关重要的。Ian 给我们展示了一下选择好的散度方程（并转化成 GAN 所需的目标函数）的重要性：

[Is the divergence important?](http://upload-images.jianshu.io/upload_images/5952841-8b2e3ec4883c8180.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里，Q 指的是我们手头噪音数据的分布，P 指的是需要找寻的目标分布。

但是，在之后的几篇论文中，很多学者（包括 Ian 本人）都提到，使用 KL(Q || P) 更能模拟出我们所面临的情景。而本质上讲，标准的 Maximum Likelihood 思维写出来就应该是 KL(P || Q) 的形式。

首先，不管是 Q 在前还是 P 在前，面对无限多的数据的时候，我们都可以最终学到完美符合我们心声的真实数据分布 P。但是问题就在于，现实生活中，我们拿到的数据是有限的。我们的学习过程可能不能持续到最后模型完美的时刻。所以这里，区别就出现了：根据 KL 公式的理论意义，KL(P || Q) 里的 Q 是用来拟合真实数据 P 的，它会极大地想要解释全部 P 的内涵（overgeneralization）。这时候，遇到多模态（multimodal）的情况（比如一张图中的双峰，羞羞），KL(P || Q) 会想要最大地覆盖两座峰。如果此时数据并不够多，它会在覆盖到一半的路上就停了下来。

相反，KL(Q || P) 是一种 undergeneralization 的情况。一个被优化的 Q 一般会先想着去覆盖一个比较大的峰，有空了再去看另一个峰。

再换句话说，它们俩一个是激进派一个是保守派。而因为我们数据是有限的，在复杂的社会环境下，保守派能确保至少算出来的那一部分是靠谱的，而激进派却容易犯错。

先不管 Q 和 P 谁前谁后，我们都把我们 G 的目标函数改造成解最大似然的形式：

[Modifying GANs to do Maximum Likelihood](http://upload-images.jianshu.io/upload_images/5952841-0ac70be88f90085d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

以上我们讲到的三种生成模型目标函数的方法，效果比较如下：

[Comparison of Generator Losses](http://upload-images.jianshu.io/upload_images/5952841-d9ee58073626db84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中，还是 Maximal Likelihood 最像香港记者，跑得最快。

GAN 不光可以用来生成（复刻）样本，还可以被转型成强化学习模型（Reinforcement Learning）

[Reducing GANs to RL](http://upload-images.jianshu.io/upload_images/5952841-16ca3103aa0fd513.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**Tips and Tricks**

**第一个技巧是把数据标签给 GAN**

[Labels improve subjective sample quality](http://upload-images.jianshu.io/upload_images/5952841-c281472a0c0dcccf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

虽然 GAN 是一种无监督算法。但是，如果要想提高训练学习的效果，有点标签还是会有很大的帮助的。也就是说，学习一个条件概率 p(y|x) 远比一个单独的 p(x) 容易得多。

实际运用中，我们可以只需要一部分有标签数据（如果有的话）就能大幅提升 GAN 的训练效果；我们称之为半监督（semi-supervising）。当然，要注意的是，如果我们用了半监督学习，我们的数据就有三类了：真实无标签数据、有标签数据和由噪音数据生成的数据。它们互相之间是不能混的。同时，我们的目标函数也成了监督方法和无监督方法的结合。

跟其它机器学习算法一样，如果我们给了 GAN 一些有标签的数据，这些标签最好是平滑（smooth）过的，也就是说把要么是 0 要么是 1 的离散标签变成更加平滑的 0.1 和 0.9 等等。

[One-sided label smoothing](http://upload-images.jianshu.io/upload_images/5952841-ac0199e236399e8e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然而，这样又会造成一些数学问题。因为，如果我们的真实数据被标注为 0.9（alpha），假数据被标注为 0.1（beta），那么我们最优的判别函数就会被写成如下图所示的样式。大家发现问题了吗？这个式子的分子不纯洁了，混进了以 beta 为系数的假数据分布。所以，对于假数据，我们还是建议保留标签为 0。一个平滑，另一个不平滑，也就称为 one-sided label smoothing（单边标签平滑）。

[Do not smooth negative labels](http://upload-images.jianshu.io/upload_images/5952841-ea5d986a9e6db84e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

标签平滑化有很多好处，特别对于 GAN 而言，这能让判别函数不会给出太大的梯度信号（gradient signal），也能防止算法走向极端样本的陷阱。

[Benefits of label smoothing](http://upload-images.jianshu.io/upload_images/5952841-bd4b8db18568eb6f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**第二个技巧是 Batch Norm**

[Batch Norm](http://upload-images.jianshu.io/upload_images/5952841-3d5ac302afb5ef57.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Batch Norm 的意思就是说，取「一批」数据，把它们给规范化（normalise）一下（减平均值，除以标准差）。它的作用就不用说了：让数据更集中，不用担心太大或者太小的数据，也让学习效率更高。

不过直接用 batch norm 也有问题的。同一批（batch）里面的数据太过相似，对一个无监督的 GAN 而言，很容易被带偏而误认为它们这些数据都是一样的。也就是说，最终的生成模型的结果会混着同一个 batch 里好多其它特征。这不是我们想要的形式。

[Batch norm in G can cause strong intra-batch correlation](http://upload-images.jianshu.io/upload_images/5952841-ccac967d721f92f0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

所以，我们可以使用 Reference Batch Norm：

[Reference Batch Norm](http://upload-images.jianshu.io/upload_images/5952841-51a4bd358ae1658f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

取出一批数据（固定的）当作我们的参照数据集 R。然后把新的数据 batch 都依据 R 的平均值和标准差来做规范化。

这个方法也有一些问题：如果 R 取得不好，效果也不会好。或者，数据可能被 R 搞得过拟合。换句话说：我们最后生成的数据可能又都变得跟 R 很像。

所以，再进阶一点，我们可以使用 Virtual Batch Norm

这里，我们依旧是取出 R，但是所有的新数据 x 做规范化的时候，我们把 x 也加入到 R 中形成一个新的 virtual batch V。并用这个 V 的平均值和标准差来标准化 x。这样就能极大减少 R 的风险。

**第三个技巧：平衡好 G 和 D**

[Balancing G and D](http://upload-images.jianshu.io/upload_images/5952841-4d4848f9a80aa989.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通常，在对抗网络中，判别模型 D 会赢。并且在实际中，D 也比 G 会深很多。Ian 建议大家不要太担心 D 会变得太聪明，我们可以用下面这些方法来达到效果最大化：

- 就像之前说的，使用非饱和（non-saturating）博弈来写目标函数，保证 D 学完之后，G 还可以继续学习；
- 使用标签平滑化。

**研究前沿**

GAN 依旧面临着几大问题。

问题 1：不收敛

[Non-convergence](http://upload-images.jianshu.io/upload_images/5952841-6b9446d351188807.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

现在 GAN 面临的最大问题就是不稳定，很多情况下都无法收敛（non-convergence）。原因是我们使用的优化方法很容易只找到一个局部最优点，而不是全局最优点。或者，有些算法根本就没法收敛。

模式崩溃（mode collapse）就是一种无法收敛的情况，这在 Ian 2014 年的首篇论文中就被提及了。比如，对于一个最小最大博弈的问题，我们把最小（min）还是最大（max）放在内循环？这个有点像刚刚说的 reverse KL 和 maximum likelihood 的区别（激进派和保守派）。minmax V(G,D) 不等于 maxmin V(G,D)。如果 maxD 放在内圈，算法可以收敛到应该有的位置，如果 minG 放在内圈，算法就会一股脑地扑向其中一个聚集区，而不会看到全局分布。

[Mode Collapse](http://upload-images.jianshu.io/upload_images/5952841-82d6f237a8ba7cb9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

大家觉得这个特性是不是跟我们上面提到的什么东西有点像？对咯，就是 reverse KL！

[Reverse KL loss does not explain mode collapse](http://upload-images.jianshu.io/upload_images/5952841-1ce9eeb046552184.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为 reverse KL 是一种保守派的损失（loss），有人觉得可能使用 reverse KL 会导致模式崩溃（mode collapse），但 Ian 表示，实际上其他的任何目标函数都可能会造成模式崩溃，这并不能解释问题的原因。

我们可以看到，遇到了模式崩溃，我们的数据生成结果就会少很多多样性，基本上仅会带有部分的几个特征（因为学习出来的特征都只聚集在全部特征中几个地方）。

[Mode collapse causes low output diversity](http://upload-images.jianshu.io/upload_images/5952841-89c2742732461211.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了解决这个问题，目前表现最好的 GAN 变种是 minibatch GAN。

[Minibatch Feature](http://upload-images.jianshu.io/upload_images/5952841-ea34c3223a43f014.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

把原数据分成小 batch，并保证太相似的数据样本不被放到一个小 batch 里面。这样，至少我们每一次跑的数据都足够多样。原理上讲，这种方法可以避免模式崩溃。（更多的优化方法还有待学界的研究。）

接下来这几页幻灯展示了使用 minibatch GAN 的效果：

[minibatch GAN Result](http://upload-images.jianshu.io/upload_images/5952841-d88f781b05853d02.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

当然，还有一些尴尬的问题依旧是「未解之谜」。

这里你们看到的，已经是精挑细选出来的好图片了，然而…

[Cherry-Picked Results](http://upload-images.jianshu.io/upload_images/5952841-2d5a005c237b2e99.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

生成的图片的细节数量经常会被搞错：

[Problems with Counting](http://upload-images.jianshu.io/upload_images/5952841-522d76ed17a85b45.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对于空间的理解会搞错。因为图片本身是用 2D 表示的 3D 世界，所以由图片样本生成的图片，在空间的表达上不够好。比如，生成的小狗似乎是贴在墙上的 2D 狗:

[Problems with Perspective](http://upload-images.jianshu.io/upload_images/5952841-38826d88e98715b0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

奇怪的东西会出现：

[Problems with Global Structure](http://upload-images.jianshu.io/upload_images/5952841-d5d505a0b134a240.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

说到这里，Ian 开了一个玩笑，放出了一张真实的猫的姿势奇异的照片（虽然很奇怪，但喵星人还是办到了！）：

[This one is real](http://upload-images.jianshu.io/upload_images/5952841-2faec0dc3eca4af7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

除了 minibatch 之外，另一种 GAN 的模型变种是 unrolling GAN（不滚的 GAN）。也就是说，每一步并不把判别模型 D 这个学习的雪球给滚起来，而是把 K 次的 D 都存起来，然后根据损失（loss）来选择最好的那一个。它的效果似乎也不错。

[Unrolled GANs](http://upload-images.jianshu.io/upload_images/5952841-3c09d65ad20e41c0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

问题 2：评估

[Evaluation](http://upload-images.jianshu.io/upload_images/5952841-1f9d20d2ff677ed0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对于整个生成模型领域来说，另一个问题是没法科学地进行评估。比如你拿一堆图片生成另一堆图片，可是这两批图片其实可能看起来天差地别。人可以判断出生成的小狗照片对不对，机器却没法量化这个标准。

问题 3：离散输出

[Discrete outputs](http://upload-images.jianshu.io/upload_images/5952841-4df669985d43581c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

到目前为止我们提到的输出或者案例都是连续的情况。如果我们的 G 想要生成离散值，就会遇到一个数学问题：无法微分（differentiate）。当然，这个问题其实在 ANN 时代就被讨论过，并有很多解决方案，比如，Williams(1992) 经典的 REINFORCE、Jang et al.(2016) 的 Gumbel-softmax、以及最简单粗暴地用连续数值做训练，最后框个范围，再输出离散值。

问题 4：强化学习的连接。

GAN 在给强化学习做加持的时候，也有点问题。首先，GAN 与强化学习的配合使用，目前大概比较火的有如下几篇论文（这里不做深入了，有兴趣的同学请自行阅读）：

- [RL connections](http://upload-images.jianshu.io/upload_images/5952841-f043d3a20ba2a40f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [Finding equilibria in games](http://upload-images.jianshu.io/upload_images/5952841-55b60517fe50cdb4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果把 GAN 加持过的 RL 用在各种游戏的学习中，GAN 该有的问题其实还是会有：

- 无法收敛；
- 面对有限步数的游戏，穷举更加简单粗暴，并且效果好；
- 如果游戏的出招是个连续值怎么办？（比如《英雄联盟》中，Q 下去还得选方向，R 起来还得点位置，等等）；
- 当然，可以用刚刚说的 unrolling 来「探寻」最优解，但是每一步都要记 K 个判别模型，代价太大。

**将GAN与其它方法结合**

GAN 不光自身有变种和优化，也能被其它算法融合吸收，进而发挥出强大的效果：

[Plug and Play Generative Models](http://upload-images.jianshu.io/upload_images/5952841-bcf054323fe4a4ca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这就是在 NIPS 几天前刚刚发布的 PPGN（Plug and Play Generative Models/即插即用生成模型）(Nguyen et al, 2016) 了。这是生成模型领域新的 State-of-the-art（当前最佳）级别的论文。它从 ImageNet 中生成了 227*227 的真实图片，是目前在这个数据集上跑得最惊人的一套算法。

Ian 本人也表示惊呆了，目测明年肯定是要被狂推上各种会议。（机器之心也会在第一时间给大家解读）

效果图（跟刚刚 Ian 自己的例子相比，差距确实还是肉眼看得出来的）：

- [PPGN Samples](http://upload-images.jianshu.io/upload_images/5952841-0cda8993486d80ae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- [PPGN for caption to image](http://upload-images.jianshu.io/upload_images/5952841-593a0583fce5813b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这时候有人要问了：刚刚不是说 minibatch GAN 表现最好吗？

哈哈，这就是一个学术界的分类梗了。不管是加上 minibatch 还是 unrolling，它们本质上还是 GAN。而 PPGN 不是 GAN 构架，它可以说是融合了包括 GAN 在内的很多算法和技巧的有新算法。

Ian 顺便夹了点私活：他说 PPGN 如果不带上 GAN，立马崩。

[GAN loss is a key ingredient](http://upload-images.jianshu.io/upload_images/5952841-b601431ec3cfb2d1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**总结**

- GAN 是一种用可以利用监督学习来估测复杂目标函数的生成模型（注：这里的监督学习是指 GAN 内部自己拿真假样本对照，并不是说 GAN 是监督学习）。
- GAN 可以估测很多目标函数，包括最大似然（Maximum likelihood）（因为这是生成模型大家族的标配）。
- 在高维度+连续+非凸的情况下找到纳什均衡依旧是一个有待研究的问题（近两年想上会议的博士生们可以重点关注）。
- GAN 自古以来便是 PPGN 不可分割的重要部分。（看到自己的理论基础被运用到一个新高度，GAN 之父会心一笑~）
