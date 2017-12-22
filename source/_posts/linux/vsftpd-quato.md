---
title: vsftpd和quota配额
date: 2017-09-14 16:00:33
tags: 
- vsftpd
- quota
categories: linux
---

# vsftpd服务器

<!-- more -->

[vsftpd入门](http://os.51cto.com/art/201008/222036.htm)

[vsftpd官网](https://security.appspot.com/vsftpd.html)

[vsftpd.conf](http://vsftpd.beasts.org/vsftpd_conf.html)

`cat /etc/passwd`命令解析如下

`ftp:x:14:50:FTP User:/var/ftp:/sbin/nologin`

以冒号分割，空格没有作用

1. `ftp` 是用户名
1. `x` 是密码字段，是隐藏的；
1. `14` 是用户的UID字段，可以自己来设定，不要和其它用户的UID相同，否则会造成系统安全问题；
1. `50` 用用户组的GID，可以自己设定，不要和其它用户组共用FTP的GID，否则会造成系统全全问题；
1. `FTP User` 是用户说明字段；
1. `/var/ftp` 是ftp用户的家目录，即对应该用户的`/`，可以自己来定义；
1. `/sbin/nologin` 这是用户登录SHELL ，这个也是可以定义的，/sbin/nologin 表示不能登录系统；系统虚拟帐号（也被称为伪用户）一般都是这么设置。比如我们把ftp用户的/sbin/nologin 改为 /bin/bash ，这样ftp用户通过本地或者远程工具ssh或telnet以真实用户身份登录到系统。这样做对系统来说是不安全的；如果您认为一个用户没有太大的必要登录到系统，就可以只给他FTP帐号的权限，也就是说只给他FTP的权限，而不要把他的SHELL设置成 /bin/bash 等；

在 /etc/shells增加一行

```
/sbin/nologin
```

创建ftp 用户yy

```
useradd -s /sbin/nologin -d /data -m yy -p 123
echo "yy" >> /etc/vsftpd/user_list
```

## 安装

```shell
apt-get install vsftpd
```

## 配置

安装软件后，`/etc`目录下无`vsftpd`文件夹，只有`/etc/vsftpd.conf`。

默认配置如下，全部配置见：[man page](https://linux.die.net/man/5/vsftpd.conf)、[中文](http://blog.csdn.net/istruth/article/details/41776767?locationNum=4)

## 问题

[有些目录无法登录 vsftpd: "500 OOPS: priv_sock_get_cmd"](http://worldend.logdown.com/posts/247495-solve-vsftpd-500-oopspriv-sock-get-cmd)

# Quota:


man 手册 (uid代替纯数字用户名)

[inux quota配置](http://zhoualine.iteye.com/blog/1613788)

[Quota  installation and configuration on Ubuntu ](https://www.howtoforge.com/tutorial/linux-quota-ubuntu-debian/#assigning-quotas-for-particular-user-or-group)

- 创建用户data
```
./create_ftp_user.sh data
```
- 修改用户data的密码
```
./gpu_change_passwd.sh data 8d777f38 data123
```
- 查看用户ID
```
id username
```
- 查看用户配额
```
sudo repquota -auvs
```

- 设置用户配额（200G）：
```
setquota  -u 1255  0 209715200 0 0  -a  /mnt/bigdata
```

setquota  -u 1104  0 188743680 0 0  -a  /mnt/bigdata

