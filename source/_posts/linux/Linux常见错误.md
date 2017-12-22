---
title: Linux常见错误
date: 2017-09-14 16:02:36
tags: linux
categories: linux
---

## `no talloc stackframe at ../source3/param/loadparm.c:4864, leaking momroy`

<!-- more -->

方法

```
sudo pam-auth-update 
```

## apt-get update出错 

错误信息

```
Ign http://mirrors.zte.com.cn trusty InRelease                                 
Ign http://mirrors.zte.com.cn trusty Release.gpg
Ign http://mirrors.zte.com.cn trusty Release
Err http://mirrors.zte.com.cn trusty/main Sources                              
  
Err http://mirrors.zte.com.cn trusty/main Sources                        
  
Err http://mirrors.zte.com.cn trusty/main Sources  
  
Err http://mirrors.zte.com.cn trusty/multiverse i386 Packages
  
Err http://mirrors.zte.com.cn trusty/main Sources  
  Undetermined Error
Err http://mirrors.zte.com.cn trusty/multiverse i386 Packages
  
Err http://mirrors.zte.com.cn trusty/multiverse i386 Packages
  
Err http://mirrors.zte.com.cn trusty/multiverse i386 Packages
  Undetermined Error
W: Failed to fetch http://mirrors.zte.com.cn/ubuntu/dists/trusty/main/source/Sources  Undetermined Error

W: Failed to fetch http://mirrors.zte.com.cn/ubuntu/dists/trusty/multiverse/binary-i386/Packages  Undetermined Error

E: Some index files failed to download. They have been ignored, or old ones used instead.
```

source

解决办法：apt.conf中未设置代理

```
vim /etc/apt/apt.conf
Acquire::http::Proxy::mirrors.zte.com.cn DIRECT;
```

## elliptic curve routines:EC_GROUP_new_by_curve_name:unknown group

调用`yum update`即可。

## yum与python不匹配

参考[blog](http://webcache.googleusercontent.com/search?q=cache:tS7lB4Sz9U0J:smilepad.blog.51cto.com/6094369/1333478+&cd=9&hl=zh-CN&ct=clnk&gl=hk&client=aff-cs-360chromium)

```
wget http://mirrors.zte.com.cn/centos/6.9/os/x86_64/Packages/python-2.6.6-66.el6_8.x86_64.rpm
wget http://mirrors.zte.com.cn/centos/6.9/os/x86_64/Packages/python-devel-2.6.6-66.el6_8.x86_64.rpm
wget http://mirrors.zte.com.cn/centos/6.9/os/x86_64/Packages/python-libs-2.6.6-66.el6_8.x86_64.rpm
rpm -Uvh --replacepkgs python*.rpm
vim /usr/bin/yum
```

行首改为

```
/usr/bin/python2.6
```

添加源

```
bash -c "echo -e '[base]\n\
name=CentOS-$releasever - Base\n\
baseurl=http://mirrors.zte.com.cn/centos/6.9/os/x86_64/\n\
gpgcheck=0'"\
> /etc/yum.repos.d/CentOS-Base.repo

yum update
```
