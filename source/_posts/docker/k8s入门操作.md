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

### 启动etcd

```
docker run --net=host -d googlegcr/etcd:2.0.12 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data
```

### 启动master

```
docker run \
    --volume=/:/rootfs:ro \
    --volume=/sys:/sys:ro \
    --volume=/dev:/dev \
    --volume=/var/lib/docker/:/var/lib/docker:ro \
    --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
    --volume=/var/run:/var/run:rw \
    --net=host \
    --pid=host \
    --privileged=true \
    -d \
    googlegcr/hyperkube:v1.0.1 \
    /hyperkube kubelet --containerized --hostname-override=127.0.0.1 --address=0.0.0.0 --api-servers=http://localhost:8080 --config=/etc/kubernetes/manifests
```

### 运行service proxy

```
docker run -d --net=host --privileged googlegcr/hyperkube:v1.0.1 /hyperkube proxy --master=http://127.0.0.1:8080 --v=2
```

搁浅

## 第二次安装

参考

- [Ubuntu 14.04主机上部署k8s集群](http://www.cnblogs.com/ilinuxer/p/6368466.html)

记录命令如下：

```
# 下载kubernetes仓库
# 若为内网环境，则可以使用本地git仓库先行导入后后续clone
# 若由Windows下载后续FTP传入，可能会丢失可执行文件的权限，导致编译出错
git clone https://github.com/kubernetes/kubernetes.git

# 下载必要镜像
docker pull 
```

放弃

## 直接参照官网安装

[https://kubernetes.io/docs/home/](https://kubernetes.io/docs/home/)

