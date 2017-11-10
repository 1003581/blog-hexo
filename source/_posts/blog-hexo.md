---
title: 使用Hexo搭建个人博客(Github+Coding)
date: 2017-09-16 22:38:13
tags: 
- hexo
- traivs
categories: tool
---

使用Hexo而放弃Jekyll的原因：

<!-- more -->

# Hexo和Jekyll的比较

参见[博客](http://blog.csdn.net/aluomaidi/article/details/52620729)

我一开始使用Jekyll进行建站，最后放弃的原因有以下几点：

1. 难看，界面没有Hexo的好看。
2. 文章目录无法动态生成，或者是我不会。
3. 调试不方便，需要自己改较多的html代码（本人非前端）。
4. Hexo搭建较Jekyll学习成本低。

附原本的Jekyll博客代码。[https://github.com/liqiang311/liqiang311.github.io.jekyll](https://github.com/liqiang311/liqiang311.github.io.jekyll)，博客引自[https://github.com/leopardpan/leopardpan.github.io](https://github.com/leopardpan/leopardpan.github.io)。

# 准备工作

## 创建GitHub、Coding仓库

首页，你得拥有一个Github的帐号和Coding的帐号。

接下来，你需要在Github上创建一个名字为`your-username.github.io`的仓库，这里的`your-username`为你的github用户名，比如我的为`liqiang311`。接下来，在Coding创建一个仓库，选择`公有`，名字随意。

## Hexo环境

本人在Windows进行管理，所以需要在Windows下进行安装Hexo环境。

参考自[博客](http://blog.csdn.net/xiaoliuge01/article/details/50997754)

## 购买域名

阿里的万网、腾讯的等等都可以选择。

注：作者购买的万网域名，但是使用了腾讯的DNSpod解析。原因是万网的解析不支持顶级域名解析（liqiang311.com），DNSPod目前支持，操作步骤见[baidu](https://jingyan.baidu.com/article/2c8c281daa4faa0008252ac7.html)

# 博客代码

本博客代码，代码仅作参考，下面介绍如何详细部署。

[https://github.com/liqiang311/blog-hexo](https://github.com/liqiang311/blog-hexo)


# 利用Traivs CI实现自动部署

1. 参考1[http://www.w3cboy.com/post/2016/03/travisci-hexo-deploy/](http://www.w3cboy.com/post/2016/03/travisci-hexo-deploy/)

2. 参考2
    1. [1](https://huangyijie.com/2016/09/20/blog-with-github-travis-ci-and-coding-net-1/)
    2. [2](https://huangyijie.com/2016/10/05/blog-with-github-travis-ci-and-coding-net-2/)
    3. [3](https://huangyijie.com/2017/06/22/blog-with-github-travis-ci-and-coding-net-3/)

# SEO相关

[Hexo-优化：提交sitemap及解决百度爬虫抓取-GitHub-Pages-问题](http://www.yuan-ji.me/Hexo-%E4%BC%98%E5%8C%96%EF%BC%9A%E6%8F%90%E4%BA%A4sitemap%E5%8F%8A%E8%A7%A3%E5%86%B3%E7%99%BE%E5%BA%A6%E7%88%AC%E8%99%AB%E6%8A%93%E5%8F%96-GitHub-Pages-%E9%97%AE%E9%A2%98/)

# MathJax

替换markdown渲染器

[参考1](https://peterxugo.github.io/2017/05/27/hexo%E5%86%99%E5%8D%9A%E5%AE%A2/)

[参考2](http://blog.junyu.io/posts/0011-hexo-math-plugin-test-report.html)

# 字体修改


