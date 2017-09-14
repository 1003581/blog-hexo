---
title: Linux常用命令
date: 2017-09-14 16:02:27
tags: linux
categories: linux
---

# 下载

<!-- more -->

[Ubuntu ISO 清华下载](https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/)

# Ubuntu

## 版本号

```
lsb_release -a
cat /etc/issue
uname -a
```

## 设置root密码 

```
sudo passwd root
```

## 为Root用户开启SSH登录

```
sed -i 's/^PermitRootLogin.*$/PermitRootLogin yes/g' /etc/ssh/sshd_config
service ssh restart
```

## 添加管理员账户

```
su
useradd -s /bin/bash -mr <username>
passwd <username>
visudo
```

在`root    ALL=(ALL:ALL) ALL`下按格式对齐添加一行`<username>   ALL=(ALL:ALL) ALL`, `Ctrl+O`回车保存,`Ctrl+X`退出.

## 启动禁用账户

[link](http://blog.csdn.net/rainylin/article/details/6132916)

## 开机自启动

```
update-rc.d xxxxx defaults
```

## 时区设置

更改时区

```
dpkg-reconfigure tzdata
```

同步时间

```
ntpdate cn.pool.ntp.org
ntpdate 10.30.1.105(自定义时间服务器)
date -s "29 Jun 2017 09:42:00"(手动指定时间)
```

## 免密钥登录

1. `ssh-keygen -t rsa`
2. `ssh-copy-id -i ~/.ssh/id_rsa.pub root@10.42.10.xx`

## 进程相关

```
# 查看所有进程
ps -ef
# 查看指定进程状态
ps -ef | grep 进程名

ps -aux
```

查看进行详细信息

```
# 查看指定进程详细信息
pmap -d PID
# 查看指定进程详细信息的最后一行
pmap -d PID | tail -1

# mapped 表示该进程映射的虚拟地址空间大小，即该进程预先分配的虚拟内存大小
# writeable/private 表示进程所占用的私有地址空间大小，即该进程实际使用的内存大小
# shared 表示进程和其他进程共享的内存大小
```

top

```
# 所有进程状态
top
# 查看指定进程的状态
top -p PID
# 每隔x秒刷新一次指定进程状态
top -p PID -d x

# PID           进程ID
# USER          进程所有者
# PR            进程优先级，越小越优先
# NI
# VIRT          进程占用的虚拟内存
# RES           进程占用的物理内存
# SHR           进程使用的共享内存
# S             进程状态 S-休眠 R-运行 Z-僵死 N-优先级为负数
# %CPU          进程占用CPU使用率
# %MEM          进程使用的物理内存和总内存的百分比
# TIME+         进程启动后占用的总CPU时间
# COMMAND       进程启动名称
```

## 用管道命令移动指定文件

```
find . -name "*.cpp" | xargs -i mv {} ./
```

## \`dirname $0\`

[link](http://www.cnblogs.com/xupeizhi/archive/2013/02/19/2917644.html)

在命令行状态下单纯执行$ cd \`dirname $0 \` 是毫无意义的。因为他返回当前路径的"."。

这个命令写在脚本文件里才有作用，他返回这个脚本文件放置的目录，并可以根据这个目录来定位所要运行程序的相对位置（绝对位置除外）。

在/home/admin/test/下新建test.sh内容如下:

```
cd `dirname $0`
echo `pwd`
```

然后返回到/home/admin/执行

```
sh test/test.sh
```

运行结果:

```
/home/admin/test
```

这样就可以知道一些和脚本一起部署的文件的位置了，只要知道相对位置就可以根据这个目录来定位，而可以不用关心绝对位置。这样脚本的可移植性就提高了，扔到任何一台服务器，（如果是部署脚本）都可以执行。

## 命令行设置静态IP(未成功)

首先查看自己的ip和gateway

```
ifconfig
netstat -rn
```

第一行0.0.0.0对应的就是，然后编辑文件

```
vim /etc/network/interfaces
```

初始内容为:

```
# interfaces(5) file used by ifup(8) and ifdown(8)
auto lo
iface lo inet loopback
```

后续追加内容如下:

```
auto eth0
iface eth0 inet static　    #静态ip
address 192.168.216.128　　 #要设置的ip
gateway 192.168.216.2       #这个地址需要自己确认
netmask 255.255.255.0       #子网掩码

dns-nameservers 192.168.216.2
```

重启ubuntu的网卡

```
sudo /etc/init.d/networking restart
```

重启电脑

```
reboot
```

附:若出现无法出现eth0输入

```
ifconfig -a
ifconfig eth0 up
dhclient eth0
ifconfig eth0
reboot
```


