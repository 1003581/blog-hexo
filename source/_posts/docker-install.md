---
title: Docker和Docker-Compose安装
date: 2017-09-14 16:00:56
tags: 
- docker
- docker-compose
categories: docker
---

# Docker安装

<!-- more -->

[官网文档链接](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository)

安装命令如下:

```bash
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get install linux-image-extra-$(uname -r) linux-image-extra-virtual
apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce
```

# Docker-Compose安装

[官方安装文档](https://docs.docker.com/compose/install/)

[安装命令](https://github.com/docker/compose/releases)

```
curl -L https://github.com/docker/compose/releases/download/1.16.1/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```



# 加速

见[http://www.liqiang311.com/2017/09/sources/](http://www.liqiang311.com/2017/09/sources/)