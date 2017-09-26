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