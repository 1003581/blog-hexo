---
title: Python抓包
date: 2017-09-14 16:03:54
tags: python
categories: python
---

如今大数据是互联网技术的热门，应用也很广泛，所以无论是做互联网产品还是学术研究，抓取他人的资源是快速有效的方法，只要不盗取版权就不为过。开源的爬虫软件很多，本节来介绍最流行也是使用最多的python爬虫开源项目scrapy 

<!-- more -->

## 安装

```
apt-get install -y python-dev zlib1g-dev libxml2-dev libxslt1-dev libssl-dev libffi-dev
pip install scrapy
```

安装目录`/usr/local/lib/python2.7/dist-packages/scrapy`

```
root@ubuntu:~# scrapy
Scrapy 1.4.0 - no active project

Usage:
  scrapy <command> [options] [args]

Available commands:
  bench         Run quick benchmark test
  fetch         Fetch a URL using the Scrapy downloader
  genspider     Generate new spider using pre-defined templates
  runspider     Run a self-contained spider (without creating a project)
  settings      Get settings values
  shell         Interactive scraping console
  startproject  Create new project
  version       Print Scrapy version
  view          Open URL in browser, as seen by Scrapy

  [ more ]      More commands available when run from project directory

Use "scrapy <command> -h" to see more info about a command
```

若出现`'module' object has no attribute 'OP_NO_TLSv1_1'`错误，则安装twist指定版本。

在[文档](https://doc.scrapy.org/en/latest/intro/install.html)中查看Twisted的最低版本，安装此版本`pip install twisted==14.0.0`。

[中文文档](http://scrapy-chs.readthedocs.io/zh_CN/latest/)

[官方文档](https://doc.scrapy.org/en/latest/intro/install.html)

## 使用样例

创建文件`test.py`如下：

```python
# -*- coding: utf-8 -*-
import scrapy


class ShareditorSpider(scrapy.Spider):
    name = 'shareditor'
    start_urls = ['http://www.shareditor.com/']

    def parse(self, response):
        for href in response.css('a::attr(href)'):
            full_url = response.urljoin(href.extract())
            yield scrapy.Request(full_url, callback=self.parse_question)

    def parse_question(self, response):
        yield {
            'title': response.css('h1 a::text').extract()[0],
            'link': response.url,
        }
```

执行

```
scrapy runspider test.py
```

会看到抓取打印的debug信息，它爬取了www.shareditor.com网站的全部网页

是不是很容易掌握？

### 创建网络爬虫常规方法

上面是一个最简单的样例，真正网络爬虫需要有精细的配置和复杂的逻辑，所以介绍一下scrapy的常规创建网络爬虫的方法

执行

```
scrapy startproject myfirstpro
```

自动创建了myfirstpro目录，进去看下内容：

```
[root@centos7vm tmp]# cd myfirstpro/myfirstpro/
[root@centos7vm myfirstpro]# ls
__init__.py  items.py  pipelines.py  settings.py  spiders
```

### 讲解一下几个文件

settings.py是爬虫的配置文件，讲解其中几个配置项：

 

USER_AGENT是ua，也就是发http请求时指明我是谁，因为我们的目的不纯，所以我们伪造成浏览器，改成
