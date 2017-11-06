---
title: 各类镜像加速汇总
date: 2017-09-14 15:59:21
tags: mirrors
categories: tool
---

汇总用到的各类加速镜像如下：

<!-- more -->

# Ubuntu

- [163镜像源](http://mirrors.163.com/.help/ubuntu.html)
- [Aliyun镜像源](http://mirrors.aliyun.com/help/ubuntu)
- [tuna镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)

使用方法

```
vim /etc/apt/sources.list
```

附常用16.04和14.04源

```
deb http://mirrors.163.com/ubuntu/ trusty main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ trusty-security main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ trusty-updates main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ trusty-proposed main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ trusty-backports main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ trusty main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ trusty-security main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ trusty-updates main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ trusty-proposed main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ trusty-backports main restricted universe multiverse
```

```
deb http://mirrors.163.com/ubuntu/ xential main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xential-security main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xential-updates main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xential-proposed main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xential-backports main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xential main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xential-security main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xential-updates main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xential-proposed main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xential-backports main restricted universe multiverse
```

# Pypi

[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

使用方法

```
Linux:
vim /root/.pip/pip.conf

Win10:
在user目录新建pip目录，新建文件pip.ini， 如c:\users\xx\pip\pip.ini
```

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host=https://pypi.tuna.tsinghua.edu.cn/simple
```

# Conda

[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

使用方法

命令行输入

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

# DockerHub

[使用DaoCloud进行加速](http://www.daocloud.io/mirror#accelerator-doc)

使用方法：

先注册，登录后点击上面的链接。

```
curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://a6841873.m.daocloud.io
```

# npm

[taobao.org](https://my.oschina.net/anylain/blog/293936)

[Windows环境下npm install 报错: operation not permitted, rename的解决方法](http://www.jb51.net/article/93520.htm)

