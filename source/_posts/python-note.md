---
title: Python笔记
date: 2017-09-14 16:03:44
tags: python
categories: python
---

# 源代码

<!-- more -->

[https://github.com/liqiang311/python.git](https://github.com/liqiang311/python.git)

## 文件首

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
```

让源代码按照UTF-8编码读取

第一行注释是为了告诉Linux/OS X系统，这是一个Python可执行程序，Windows系统会忽略这个注释；

第二行注释是为了告诉Python解释器，按照UTF-8编码读取源代码，否则，你在源代码中写的中文输出可能会有乱码。

## utf8-reload

python2

```
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

字符转数字`ord('A')`

数字转字符`chr(65)`

Unicode编码转换为utf-8`u'ABC'.encode('utf-8')`

utf-8编码转换wieldUnicode`'abc'.decode('utf-8')`

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

## scrapy使用




