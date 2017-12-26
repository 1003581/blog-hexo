---
title: k8s入门操作
date: 2017-12-01 15:11:50
tags: k8s
categories: docker
---

k8s
<!-- more -->

# 安装

## 基于Docker本地运行Kubernetes

参考：

- [基于Docker本地运行Kubernetes](https://www.kubernetes.org.cn/doc-5)
- [kubernetes 条件需求](http://www.cnblogs.com/zhangeamon/p/5197655.html)

###  安装Docker

安装步骤 [Docker和Docker-Compose安装](http://liqiang311.com/docker/Docker%E5%92%8CDocker-Compose%E5%AE%89%E8%A3%85/)

### 你的内核必须支持 memory and swap accounting

确认你的linux内核开启了如下配置

```
cat /boot/config-***-generic | grep CONFIG_RESOURCE_COUNTERS
cat /boot/config-***-generic | grep CONFIG_MEMCG
```

若部分为开启，则显示如下：

```
root@Slave2:~# cat /boot/config-***-generic | grep CONFIG_RESOURCE_COUNTERS
CONFIG_RESOURCE_COUNTERS=y
root@Slave2:~# cat /boot/config-***-generic | grep CONFIG_MEMCG
CONFIG_MEMCG=y
CONFIG_MEMCG_SWAP=y
# CONFIG_MEMCG_SWAP_ENABLED is not set
# CONFIG_MEMCG_KMEM is not set
```

### 在内核启动时开启 memory and swap accounting 选项

```
sed -i 's/^GRUB_CMDLINE_LINUX=""/GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"/g' /etc/default/grub
update-grub
```

输入命令 

```shell
cat /proc/cmdline
```

显示如下：

```
BOOT_IMAGE=/vmlinuz-3.13.0-24-generic root=/dev/mapper/ubuntu--vg-root ro cgroup_enable=memory swapaccount=1
```
