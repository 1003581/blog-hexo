---
title: Python笔记
date: 2017-09-14 16:03:44
tags: python
categories: python
---

# 源代码

<!-- more -->

[https://github.com/liqiang311/python.git](https://github.com/liqiang311/python.git)

# 各类资料

## 文档

- [Python语言规范](http://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules/)

## 发布命令行工具

- [如何发布一个Python命令行工具](http://blog.csdn.net/starsliu/article/details/50999603)
- [15.4. argparse](https://docs.python.org/2/library/argparse.html?highlight=argparser)
- [Argparse Tutorial](https://docs.python.org/2/howto/argparse.html)

## pyenv

- [github](https://github.com/pyenv/pyenv)

## ElementTree

- [使用Python库ElementTree解析Hadoop的xml配置文件](http://www.fx114.net/qa-10-166440.aspx)
- [Python xml属性/节点/文本的增删改[xml.etree.ElementTree]](http://blog.csdn.net/wklken/article/details/7603071)

## Docker

- [Docker SDK for Python](https://docker-py.readthedocs.io/en/stable/index.html)

## PyNLPIR

- [docs.io](http://pynlpir.readthedocs.io/en/latest/tutorial.html)

## ChatterBot

- [docs.io](https://chatterbot.readthedocs.io/en/stable/)

## whoosh

- [docs.io](https://whoosh.readthedocs.io/en/latest/)

## hdfs3

- [docs.io](http://hdfs3.readthedocs.io/en/latest/install.html#pypi-and-apt-get)

## requests

- [Requests: 让 HTTP 服务人类](http://cn.python-requests.org/zh_CN/latest/)

## Theano

- [docs.io](http://deeplearning.net/software/theano/index.html)

## PyTorch

- [docs.io](http://pytorch.org/tutorials/)

## six

- [pypi](https://pypi.python.org/pypi/six#downloads)

# 常用代码

## 文件头

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
```

## utf8-reload(python2)

```python
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
```

注意： 

1. Python 3 与 python 2 有很大的区别，其中Python 3 系统默认使用的就是utf-8编码。 
2. 所以，对于使用的是Python 3 的情况，就不需要sys.setdefaultencoding("utf-8")这段代码。 
3. 最重要的是，Python 3 的 sys 库里面已经没有 setdefaultencoding() 函数了。

python3

```
import importlib
importlib.reload(sys)
```

## 类型转换

- 字符转数字`ord('A')`
- 数字转字符`chr(65)`
- Unicode编码转换为utf-8`u'ABC'.encode('utf-8')`
- utf-8编码转换wieldUnicode`'abc'.decode('utf-8')`

## 发布为exe文件

py代码，data_files中存放附带打包文件

```python
from distutils.core import setup
#import glob
import py2exe

setup(console=["main.py"],
      data_files=[(".",["cfg.ini",]),])
```

命令行中输入

```shell
python mypy2exe.py py2exe
```

会生成build和dist文件夹

## 发布为Linux可执行文件

使用[PyInstaller](http://www.cnblogs.com/mywolrd/p/4756005.html)

```
pip install pyinstaller
pyinstaller -F xxx.py
./dist/xxx
```

## scrapy使用

见`Python抓包`。


