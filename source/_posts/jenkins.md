---
title: Jenkins使用
date: 2017-09-14 15:58:52
tags: jenkins
categories: linux
---

# 部署

<!-- more -->

## 准备工作

下载安装Docker，如何安装Docker和Docker-Compose见[http://www.liqiang311.com/2017/09/docker-install/](http://www.liqiang311.com/2017/09/docker-install/)

下载jenkins镜像

```
docker pull jenkins:latest
```

下载部署代码

```
git clone https://github.com/liqiang311/jenkins.git
```

## 构建

命令如下：

```
cd jenkins
docker-compose up -d
```

打开浏览器`localhost:8080`

第一次打开会提示输入密码，密码在`/var/jenkins_home/secrets/initialAdminPassword`，输入如下指令查看密码

```
cat jenkins_home/secrets/initialAdminPassword
```

将输出的字符串复制到浏览器中，点击继续。

提示输入新的用户名密码，选择不输出，右下角选择以admin继续运行。

构建完毕。

点击右上角admin下拉的设置，在密码栏更改密码。

## 安装插件

点击左侧`系统设置`->`管理插件`，然后在可选插件里点击立即获取，若需要配置代理，则在高级中设置代理。

若无法使用网络，请前往[http://updates.jenkins-ci.org/download/plugins/](http://updates.jenkins-ci.org/download/plugins/)进行手动下载，然后在高级选项中上传插件文件。

常用插件如下(不定时更新)：

- `Build Monitor View`(提供Monitor视图)
- `Node and Label parameter plugin`(提供下面多节点任务使用)
- `Multi slave config plugin`(提供一个任务跑在多台节点上)
- `Display Console Output Plugin`(显示日志)
- `GitHub plugin`

# 备份Jenkins数据

```
0 0 * * * rsync -avzWu --delete --progress --password-file=/etc/rsyncd.password  root@10.42.10.39::jenkins /root/jenkins/ >> /etc/rsync.log 2>&1 && date >> /etc/rsync.log
```

[rsync命令](http://man.linuxde.net/rsync)

> -a, --archive 归档模式，表示以递归方式传输文件，并保持所有文件属性，等于-rlptgoD。
> -v, --verbose 详细模式输出。
> -z, --compress 对备份的文件在传输时进行压缩处理。
> -w, --whole-file 拷贝文件，不进行增量检测
> -u, --update 仅仅进行更新，也就是跳过所有已经存在于DST，并且文件时间晚于要备份的文件，不覆盖更新的文件。
> --delete 删除那些DST中SRC没有的文件。
> --progress 显示备份过程。
> --password-file=FILE 从FILE中得到密码。
