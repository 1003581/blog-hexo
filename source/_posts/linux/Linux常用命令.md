---
title: Linux常用命令
date: 2017-09-14 16:02:27
tags: linux
categories: linux
---

## 下载

[Ubuntu ISO 清华下载](https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/)
<!-- more -->

## 系统相关

### 版本号
- `cat /etc/issue`
- `lsb_release -a`
- `uname -a`
### 设置root密码
`sudo passwd root`
### 时区设置
- 更改时区
  ```
  apt-get update
  apt-get install tzdata
  dpkg-reconfigure tzdata
  ```
- 或者无交互设置
  ```
  apt-get update
  apt-get install -y tzdata
  cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
  echo 'Asia/Shanghai' > /etc/timezone
  ```
### 同步时间
  ```
  ntpdate cn.pool.ntp.org
  ntpdate 10.30.1.105(自定义时间服务器)
  date -s "29 Jun 2017 09:42:00"(手动指定时间)
  ```
### 重启网卡
`sudo /etc/init.d/networking restart`
### 重启电脑
`reboot`
### 修改crontab默认编辑器
- 方法1：修改环境变量
  ```shell
  echo "export EDITOR=/usr/bin/vim" >> /etc/profile
  source /etc/profile
  ```
- 方法2：输入以下命令，选择3即可。
  ```
  update-alternatives --config editor
  ```

  ```
  There are 4 choices for the alternative editor (providing /usr/bin/editor).

    Selection    Path                Priority   Status
  ------------------------------------------------------------
  * 0            /bin/nano            40        auto mode
    1            /bin/ed             -100       manual mode
    2            /bin/nano            40        manual mode
    3            /usr/bin/vim.basic   30        manual mode
    4            /usr/bin/vim.tiny    10        manual mode
  ```
### [ubuntu网络重启后或主机重启后，/etc/resolv.conf恢复原样的解决](http://blog.csdn.net/bytxl/article/details/44201347)

## 系统监控

### [查看CPU型號、核心數量、頻率和溫度](https://magiclen.org/linux-view-cpu/)
### [查看程序端口占用情况](http://www.cnblogs.com/benio/archive/2010/09/15/1826728.html)
`netstat –apn | grep 8080`

## 用户

### 显示普通账户和系统用户

`awk -F: '{if($3>=1000){printf "Common User: %s \n",$1}}' /etc/passwd`  
`awk -F: '{if($3>=1000){printf "Common User: %s \n",$1}else{printf "root or sysuser:%s\n",$1}}' /etc/passwd `

### [启动禁用账户](http://blog.csdn.net/rainylin/article/details/6132916)

### [userdel](http://www.cnblogs.com/DaDaOnline/p/5527833.html)

### 添加管理员账户

```
su
useradd -s /bin/bash -mr <username>
passwd <username>
visudo
```
- 在`root    ALL=(ALL:ALL) ALL`下按格式对齐添加一行`<username>   ALL=(ALL:ALL) ALL`, `Ctrl+O`回车保存,`Ctrl+X`退出.

无交互方式

```
useradd -s /bin/bash -m username
echo username:password​ | chpasswd
adduser username sudo
```

> 注意：chpasswd若username采用数字id开头，则会出现修改不掉密码的情形，未找到缘由。

## SSH

### 免密钥登录
1. `ssh-keygen -t rsa`
2. `ssh-copy-id -i ~/.ssh/id_rsa.pub root@10.42.10.xx`
### 为Root用户开启SSH登录
```
sed -i 's/^PermitRootLogin.*$/PermitRootLogin yes/g' /etc/ssh/sshd_config
service ssh restart
```
### [ssh-keygen命令](http://man.linuxde.net/ssh-keygen)
### [ssh连接远程主机执行脚本的环境变量问题](http://blog.csdn.net/liuyuzhu111/article/details/51334635)
### [SSH连接反应慢的分析解决](http://xjsunjie.blog.51cto.com/999372/658354/)

## 进程相关

### 查看指定进程详细信息
`pmap -d PID`
### 查看指定进程详细信息的最后一行
- mapped 表示该进程映射的虚拟地址空间大小，即该进程预先分配的虚拟内存大小
- writeable/private 表示进程所占用的私有地址空间大小，即该进程实际使用的内存大小
- shared 表示进程和其他进程共享的内存大小
### top
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
### 开机自启动进程
`update-rc.d xxxxx defaults`

## Vim
### [Vim命令合集]http://www.cnblogs.com/softwaretesting/archive/2011/07/12/2104435.html)
### [Tab键不能自动补全问题](http://logicluo.iteye.com/blog/2145084)

## sed
### http://www.cnblogs.com/dong008259/archive/2011/12/07/2279897.html 
### http://www.frostsky.com/2014/01/linux-sed-command/ 
### 在匹配行插入某个文件的内容
`sed -i '/FROM/r common.ubuntu.head' $tmp`

## stdin/stdout/stderr
### [Shell标准输出、标准错误 >/dev/null 2>&1](http://blog.sina.com.cn/s/blog_4aae007d010192qc.html)

## 积累命令

### 用管道命令移动指定文件 `find . -name "*.cpp" | xargs -i mv {} ./`
### [\`dirname $0\`](http://www.cnblogs.com/xupeizhi/archive/2013/02/19/2917644.html)
### [批量重命名文件]((http://www.lx138.com/page.php?ID=Vkd0U1ZtVkJQVDA9))
### [pushd和popd](http://www.jianshu.com/p/53cccae3c443)
