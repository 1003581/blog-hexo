---
title: Docker命令
date: 2017-09-14 15:11:50
tags: docker
categories: docker
---

## 镜像

<!-- more -->

### Pull

```
docker pull ubuntu
docker pull ubuntu:14.04
docker pull dl.dockerpool.com:5000/ubuntu
```

### 查看已有镜像

```
docker images
```

### Tag

```
docker tag ubuntu:14.04 ubuntu:14.04.1
```

### 详情

```
docker inspect 镜像ID
docker inspect -f {“.Architecture”}{} 镜像ID
```

### 删除

```
docker rmi 镜像名称
docker rmi -f IMAGE
docker rmi $(docker images -q)       # 删除所有镜像
docker images|grep none|awk '{print $3 }'|xargs docker rmi  # 删除所有Tag为<none>的镜像
```

### 搜索

```
docker search TERM  # 搜索某个镜像  
--automated=false   # 仅显示自动创建的镜像  
--no-trunc=false    # 输出信息不截断显示  
-s, --starts=0      # 指定仅显示评价为指定星级以上的镜像
```

### 镜像导出

```
docker save –o ubuntu_14.04.tar ubuntu:14.04
```

### 载入镜像

```
docker load < ubuntu_14.04.tar
docker load –input ubuntu_14.04.tar
```

### 定制

在run ubuntu镜像，并修改配置后执行以下指令

```
docker commit -m "my ubuntu" -a "Docker Newbee" 容器ID MyUbuntu:tag
```

其中  

```
-a  # 作者消息  
-m  # 提交消息  
-p  # 提交时暂停容器运行
```

### 上传

默认传到DockerHub

```
docker push user/test.latest
```

### 导入本地镜像

```
cat xxx.tar.gz |docker import – ubuntu:14.04
```

## 容器

### 创建

```
docker create –it ubuntu:latest
```

### Run

```
docker run -it docker.example.com.cn:5000/official/ubuntu:14.04.5 /bin/bash
```

`i`表示让容器的标准输入保持打开,`t`表示让docker分配一个伪终端并绑定到容器的标准输入上

### 退出

容器变为死亡态

```
exit
```

或者

容器仍在运行

```
Ctrl+D
```

### 守护态

运行容器(d-daemonized)

```
docker run -d ubuntu bash"
```

### 查看

```
docker ps           # 正在运行的容器
docker ps -a        # 全部容器
```

### LOG

```
docker logs 容器ID
```

### 停止

```
docker stop 容器ID
```

```
docker kill $(docker ps -q)
```

停止所有正在运行的容器

### 开始

```
docker start 容器ID
```

### 重启

```
docker restart 容器ID
```

### 进入正在运行的容器

```
docker exec -ti 容器ID [命令]
```

### 删除容器

```
docker rm 容器ID
docker rm $(docker ps -a -q)    # 删除所有容器
docker rm $(docker ps -aq -f status=exited)     # 删除所有退出状态的容器
docker ps -a|grep k8s|awk '{print $1}'|xargs docker rm -f   # 删除所有名字为***的容器
```

### 导入导出

```
docker export CONTAINER
docker import
```

### 数据卷

在容器内创建数据卷

```
docker run -d -P --name hehe -v /hehe python:2.7.13 python /test/py/py1.py
```

将本机目录挂载到容器目录中

```
docker run -d -P --name py -v /home/ubuntu/docker/py:/test/py python:2.7.13 python /test/py/py1.py
```

其中`-P` 允许外部访问容器需要暴露的端口,`-v` 为挂载/home/ubuntu/docker/py:/test/py 本机目录:容器目录  

数据卷容器(任意一方修改目录，其他人均能看到)

```
docker run -idt -v /dbdata --name dbdata ubuntu:14.04.5
docker run -idt --volumes-from dbdata --name db1 ubuntu:14.04.5
docker run -idt --volumes-from dbdata --name db2 ubuntu:14.04.5
```

### 网络基础配置

端口映射

```
docker run -d -P python:2.7.13 python while.py
```

容器互联

```
docker run -d -P --name web --link db:db python:2.7.13 python while.py
```

其中`--link` 格式为 `name:alisa` `name`为要链接的容器的名称 `alias`为链接的别名

## Docker Proxy

### Ubuntu16.04

[link](http://blog.csdn.net/logsharp/article/details/53126826)

```
mkdir /etc/systemd/system/docker.service.d
touch proxy.conf
vim proxy.conf
```
```
[Service] 
Environment="HTTP_PROXY=http://proxy.example.com:80/" 
Environment="HTTPS_PROXY=http://proxy.example.com:80/" 
Environment="NO_PROXY=10.0.0.0/8,192.0.0.0/8,registry,*.example.com.cn,localhost,127.0.0.0/8,::1" 
```
```
systemctl daemon-reload
systemctl show --property=Environment docker
systemctl restart docker
```

### Ubuntu14.04

[link](http://blog.csdn.net/u011563903/article/details/52161648?locationNum=6)

```
vim /etc/default/docker
```

添加如下内容

```
export http_proxy="http://proxynj.example.com.cn:80/"
export https_proxy="http://proxynj.example.com.cn:80/"
export no_proxy="10.0.0.0/8,192.0.0.0/8,registry,*.example.com.cn,localhost,127.0.0.0/8,::1"
```

```
service docker restart 
```

### Red Hat
```
vim /etc/sysconfig/docker
```
```
http_proxy=http://proxynj.example.com.cn:80/
https_proxy=http://proxynj.example.com.cn:80/
no_proxy=10.0.0.0/8,192.0.0.0/8,registry,*.example.com.cn,localhost,127.0.0.0/8,::1
```
```
service docker restart
```

## Docker Hub

[Hub](https://hub.docker.com/r/liqiang311)

[加速器](http://www.daocloud.io/mirror#accelerator-doc)

```
curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://a6841873.m.daocloud.io
```

## Docker运行GUI软件

[【微信分享】林帆：Docker运行GUI软件的方法](http://www.csdn.net/article/2015-07-30/2825340)

[jessfraz's Blog: Docker Containers on the Desktop](https://blog.jessfraz.com/post/docker-containers-on-the-desktop/)

[jessfraz Dockerfile](https://github.com/jessfraz/dockerfiles)

## [docker 容器的网络模式](http://cizixs.com/2016/06/12/docker-network-modes-explained)

## Docker配置替代官方源

参照加速器命令

```
curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://mirrors.***.com
```
