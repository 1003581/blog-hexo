---
title: Dockerfile详解
date: 2017-09-14 15:11:50
tags: docker
categories: docker
---

- [官方文档](https://docs.docker.com/engine/reference/builder/)
- [个人Dockerfile](https://github.com/liqiang311/docker-hub)
- [编写最佳的Dockerfile的方法](http://www.jb51.net/article/115327.htm)

<!-- more -->

```
# 第一行必须指定基于的基础镜像
FROM ubuntu

# 维护者信息
MAINTAINER docker_user docker_user@email.com

# 镜像的操作指令
# RUN <command> 默认是在shell终端中执行
# RUN ["/bin/bash","-c","echo hello"]将在指定的终端中运行，第一个为指定终端，后续为参数
# 每一次RUN都将在当前镜像的基础上执行命令，并提交为新的镜像
RUN echo "deb http://archive.ubuntu.com/ubuntu/ raring main universe" >> /etc/apt/sources.list
RUN apt-get update && apt-get install -y ningx
RUN echo "\ndaemon off;" >> /etc/nginx/nginx.conf

# 容器启动时执行命令
# CMD ["executable","param1","param2"] 指定终端中运行
# CMD comand param1 param2 默认在/bin/sh中运行
# CMD ["param1", "param2"] 提供给ENTRYPOINT的默认参数
# 每个Dockerfile只有一条CMD命令，若存在多条，则以最后一条为准，若用户启动容器时指定了CMD，则覆盖Dockerfile中的CMD
CMD /usr/sbin/nginx

# EXPOSE <port> [<port>...]
# ex.EXPOSE 22 80 8443
# 指定容器暴露的端口号，若启动时用-P，则主机会自动分配一个端口转发到容器端口，使用-p则可以指定主机端口到容器端口的映射

# 设置环境变量
ENV PG_MAJOR 9.3
ENV PG_VERSION 9.3.4
ENV PATH /sur/local/postgres-$PG_MAJOR/bin:$PATH


# 将指定src的目录/文件/URL/tar文件 复制 到容器中的dest
ADD <src> <dest>
# 将主机中src的文件/目录 复制 到容器的dest中
COPY <src> <dest>

# 配置容器启动后执行的命令，一个文件中只有一个，多个时，最后一个生效
ENTRYPOINT ["executable","param1","param2"]
ENTRYPOINT command param1 param2 (shell中执行)

# 创建一个可以从本地主机或者其他容器挂载的挂载点，一般用来存放数据库和需要保持的数据。
VOLUME ["/data"]

# 指定运行容器时用到的用户名或者UID
USER daemon

WORKDIR /path/tp/workdir

ONBUILD [INSTRUCTION]
```


```
docker build -t tag /tmp/dockerfilepath
```