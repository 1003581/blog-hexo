---
title: nvidia-docker命令详解
date: 2017-09-14 16:03:26
tags: nvidia-docker
categories: gpu docker
---

## 准备

<!-- more -->

### 相关链接

- [NVIDIA/nvidia-docker Wiki](https://github.com/NVIDIA/nvidia-docker/wiki)
- [下载地址](https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb)

### 安装

[Installation](https://github.com/NVIDIA/nvidia-docker/wiki/Installation)

```
dpkg -i nvidia-docker_1.0.1-1_amd64.deb
```

### 优势

集成nvidia-docker有以下好处:

- 可复制的版本
- 易于部署
- 单独设备的隔离
- 跨异构驱动程序运行
- 只需要安装NVIDIA驱动程序
- 启用“开启和忘记”GPU应用程序
- 促进合作

## nvidia-docker

[官方Wiki](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker)

### 描述

nvidia-docker是docker顶部的薄包装, 作为docker命令行界面的替代品.  提供这个二进制文件, 方便用户自动检测和设置利用NVIDIA硬件的GPU容器.  如果您不打算使用它, 请参阅内部部分. 

在内部, nvidia-docker调用docker, 依靠NVIDIA Docker插件来发现驱动程序文件和GPU设备.  nvidia-docker使用的命令可以使用环境变量NV_DOCKER来覆盖：

```
# Running nvidia-docker with a custom docker command
NV_DOCKER='sudo docker -D' nvidia-docker <docker-options> <docker-command> <docker-args>
```

请注意, nvidia-docker仅修改运行的行为并创建Docker命令.  所有其他命令只是传递到docker命令行界面.  因此, 在构建Docker映像时, 无法执行GPU代码. 

### GPU隔离

使用环境变量`NV_GPU`通过逗号分隔的ID列表导出GPU.  ID是指定设备的索引或UUID. 
设备索引类似于nvidia-docker-plugin REST接口, nvidia-smi报告的索引, 或者运行`CUDA_DEVICE_ORDER = PCI_BUS_ID`的CUDA代码时, 它与默认的CUDA排序不同.  默认情况下, 导出所有GPU. 

```
# Running nvidia-docker isolating specific GPUs by index
NV_GPU='0,1' nvidia-docker <docker-options> <docker-command> <docker-args>
# Running nvidia-docker isolating specific GPUs by UUID
NV_GPU='GPU-836c0c09,GPU-b78a60a' nvidia-docker <docker-options> <docker-command> <docker-args>
```

### 本地运行

如果nvidia-docker-plugin安装在主机上并在本地运行, 则不需要额外的步骤.  需要启动使用NVIDIA GPU的容器时, nvidia-docker将通过查询插件执行必要的操作. 

### 远程运行

远程使用nvidia-docker需要在远程主机上运行nvidia-docker-plugin. 

可以使用环境变量`DOCKER_HOST`或`NV_HOST`设置远程主机目标. 

规定如下：

- 如果`NV_HOST`被设置, 那么它用于联系插件. 
- 如果`NV_HOST`未设置, 但`DOCKER_HOST`已设置, 则`NV_HOST`默认为`DOCKER_HOST`位置, 使用端口3476上的http协议(更多)

`NV_HOST`的规范定义为：`[(http|ssh)://][<ssh-user>@][<host>][:<ssh-port>]:[<http-port>]`

http协议要求nvidia-docker-plugin在可达接口上进行监听(默认情况下, nvidia-docker-plugin只侦听本地主机).  选择ssh的时候, 只需要有效的SSH凭据(您的ssh代理中的密码或私钥). 

```
# Run CUDA on the remote host 10.0.0.1 using HTTP
DOCKER_HOST='10.0.0.1:' nvidia-docker run cuda

# Run CUDA on the remote host 10.0.0.1 using SSH
NV_HOST='ssh://10.0.0.1:' nvidia-docker -H 10.0.0.1: run cuda

# Run CUDA on the remote host 10.0.0.1 using SSH with custom user and ports
DOCKER_HOST='10.0.0.1:' NV_HOST='ssh://foo@10.0.0.1:22:80' nvidia-docker run cuda
```

## nvidia-docker-plugin

[官方Wiki](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker-plugin)

### 描述

nvidia-docker-plugin是一个Docker Engine插件, 旨在简化在异构环境中部署GPU感知容器的过程.  它作为守护进程, 发现主机驱动程序文件和GPU设备以及源自Docker守护程序的卷安装请求的答案. 

该插件还提供了一个REST API, 可以查询以获取GPU信息, 也可以根据给定的卷名称和设备编号生成Docker参数. 

### 用法

可以使用以下参数调整插件守护程序：

```
Usage of nvidia-docker-plugin:
  -d string
        Path where to store the volumes (default "/var/lib/nvidia-docker/volumes")
  -l string
        Server listen address (default "localhost:3476")
  -s string
        Path to the plugin socket (default "/run/docker/plugins")
  -v    Show the plugin version information
```

如果您正在使用二进制包, 那么可以在init配置文件中更改：`/etc/default/nvidia-docker`或`/etc/systemd/system/nvidia-docker.service.d/override.conf`, 具体取决于您的分发. 

一旦插件运行, nvidia-docker将能够连接到它并请求信息进行容器化.  如果升级NVIDIA驱动程序, 您将需要重新启动该插件. 

### REST API

默认情况下, nvidia-docker-plugin在端口3476上本地提供其REST接口. 这将有效防止nvidia-docker通过http协议远程访问计算机(请参阅远程部署).  可以使用-l使用选项(例如通配符地址端口1234的-l：1234)来更改此行为. 

REST端点如下所述.  请注意, 如果给定的客户端仅关心最新的API版本, 则可以省略版本前缀(即/ gpu / info当前等效于/v1.0/gpu/info)

#### Version 1.0

- GET `/v1.0/gpu/info`, `/v1.0/gpu/info/json`
> 查询GPU设备信息 (类似于 `nvidia-smi -q`).  
> 应答信息格式分别为 `text/plain` and `application/json`.
- GET `/v1.0/gpu/status`, `/v1.0/gpu/status/json`
> 查看GPU设备状态 (类似于 `nvidia-smi`). 
> 应答信息格式分别为 `text/plain` and `application/json`.
- GET `/v1.0/docker/cli`, `/v1.0/docker/cli/json`
> 查询与docker运行或docker创建一起使用的命令行参数.   
> 它接受两个查询字符串参数：`dev`用于设备(类似于`NV_GPU`)和`vol`的卷.   
> 应答信息格式分别为 `text/plain` and `application/json`.  
>   
> 如果您不想依赖nvidia-docker替代CLI(参见Internals), 这很有用.  例如：  
> ```docker run -ti `curl -s http//localhost:3476/v1.0/docker/cli?dev=0+1\&vol=nvidia_driver` cuda```
- GET `/v1.0/mesos/cli`
> 查询启动Mesos代理时使用的命令行参数.   
> 与`/v1.0/gpu/info/json`类似, 但在zlib / base64(RFC 6920)中压缩和编码设备信息. 

### 已知限制

1. NVIDIA驱动程序安装需要与Docker插件的卷目录保持在同一分区上.   
`nvidia-docker-plugin`内部需要创建一些驱动程序文件的硬链接.  因此, 您需要NVIDIA驱动程序(通常位于`/usr`下)与Docker插件卷目录位于同一分区(默认情况下为`/var`).  可能的解决方法包括在不同的位置安装NVIDIA驱动程序(请参阅 - 安装程序的高级选项)或更改插件卷目录(请参阅`nvidia-docker-plugin -d`)

不建议将`nvidia-docker-plugin`卷目录放在与NVIDIA驱动程序(即`/usr/lib`, `/usr/lib64` ...)相同的目录结构下.  这是由于NVIDIA安装程序在执行升级之前执行冲突文件的检查. 

## 内部架构

### NVIDIA驱动

#### 挑战

为了在您的机器上执行GPU应用程序, 您需要安装NVIDIA驱动程序.  NVIDIA驱动程序由多个内核模块组成：

```
$ lsmod | grep nvidia
nvidia_uvm            711531  2 
nvidia_modeset        742329  0 
nvidia              10058469  80 nvidia_modeset,nvidia_uvm
```

它还提供了一组用户级驱动程序库, 使您的应用程序能够与内核模块以及GPU设备进行通信：

```
$ ldconfig -p | grep -E 'nvidia|cuda'
libnvidia-ml.so (libc6,x86-64) => /usr/lib/nvidia-375/libnvidia-ml.so
libnvidia-glcore.so.375.66 (libc6,x86-64) => /usr/lib/nvidia-375/libnvidia-glcore.so.375.66
libnvidia-compiler.so.375.66 (libc6,x86-64) => /usr/lib/nvidia-361/libnvidia-compiler.so.375.66
libcuda.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so
...
```

请注意库如何与驱动程序版本相关联.   
驱动程序安装程序还提供了诸如`nvidia-smi`和`nvidia-modprobe`之类的实用程序二进制文件. 

集成GPU应用程序的早期想法之一是将用户级驱动程序库安装在容器内(例如使用驱动程序安装程序中的选项--no-kernel-module).  然而, 用户级驱动程序库与内核模块的版本相关, 所有Docker容器共享主机操作系统内核.  内核模块的版本必须与用户级库的版本完全匹配(主要版本和次要版本).  尝试运行具有不匹配环境的容器会立即在容器内产生错误：

```
$ nvidia-smi 
Failed to initialize NVML: Driver/library version mismatch
```

这种方法使镜像不可移植, 使镜像分享变得不可能, 从而打败了Docker的主要优势.  解决方案是使镜像与驱动程序版本无关.   DockerHub上提供的Docker镜像是通用的, 但是当创建容器时, 环境必须通过`docker run`的`--volume`参数来挂载主机上的用户级库, 从而指定了主机内核模块. 

NVIDIA驱动程序支持多个主机操作系统, 有多种安装驱动程序的方法(例如runfile或deb/rpm软件包), 安装程序也可以自定义.  在整个发行版中, 驱动程序文件没有便携式位置.  要导入的文件列表也可能取决于您的驱动程序版本. 

#### nvidia-docker

我们的方法等同于如上所示运行`ldconfig -p`：我们以编程方式解析ldcache文件(`/etc/ld.so.cache`), 以发现库的预定义列表的位置.  有些库可以在系统上找到多次, 其中一些库不能被选择.  例如, OpenGL库可以由多个供应商提供.  我们有一个函数, 能够拉黑这些已知由多重来源提供的库. 

由于这些库可以分散在主机文件系统中, 因此我们创建了一个由已发现库的硬链接组成的Docker [命名卷](https://docs.docker.com/engine/userguide/containers/dockervolumes/) , 如果硬连接无效, 则有一个复制回退路径.  这个卷可以通过`nvidia-docker-plugin`守护进程来进行管理, 该守护进程实现了[`volume plugins`](https://docs.docker.com/engine/extend/plugins_volume/)的Docker API.

```
$ docker volume inspect nvidia_driver_375.66
[
    {
        "Driver": "nvidia-docker",
        "Labels": null,
        "Mountpoint": "/var/lib/nvidia-docker/volumes/nvidia_driver/375.66",
        "Name": "nvidia_driver_375.66",
        "Options": {},
        "Scope": "local"
    }
]

```

nvidia-docker包装器将自动将卷参数添加到命令行, 然后将控制权传递给docker, 您只需要运行nvidia-docker-plugin守护程序即可. 

#### 替代选择

如果不想使用nvidia-docker包装器, 可以手动添加命令行参数：

```
$ docker run --volume-driver=nvidia-docker --volume=nvidia_driver_375.66:/usr/local/nvidia:ro
```

请参阅该wiki的下一部分, 了解如何发现和导入设备文件. 

为避免使用--volume-driver(因为每个命令行只能使用一次), 您可以手动创建命名卷：

```
$ docker volume create --name=nvidia_driver_375.66 -d nvidia-docker
```

上述两个解决方案仍然需要使用nvidia-docker-plugin, 但是由于Docker正式支持`volume plugins`, 因此这个问题应该更少. 

如果您不想使用`volume plugins`, 则必须使用`ldconfig -p`或解析`ldcache`手动查找驱动程序文件.  如果您的部署环境完全相同, 您可以简单地硬编码用例的文件路径.  为了验证, 您可以查看由nvidia-docker-plugin创建的卷：

```
$ ls -R `docker volume inspect -f "{{ .Mountpoint }}" nvidia_driver_375.66`
/var/lib/nvidia-docker/volumes/nvidia_driver/375.66:
bin  lib  lib64

/var/lib/nvidia-docker/volumes/nvidia_driver/375.66/bin:
nvidia-cuda-mps-control  nvidia-cuda-mps-server  nvidia-debugdump  nvidia-persistenced  nvidia-smi

/var/lib/nvidia-docker/volumes/nvidia_driver/375.66/lib:
libEGL_nvidia.so.0             libGLESv2_nvidia.so.375.66  libnvcuvid.so.1                      libnvidia-fbc.so.1          libnvidia-ptxjitcompiler.so.375.66
libEGL_nvidia.so.375.66        libGLESv2.so.2              libnvcuvid.so.375.66                 libnvidia-fbc.so.375.66     libnvidia-tls.so.375.66
libEGL.so.1                    libGL.so.1                  libnvidia-compiler.so.375.66         libnvidia-glcore.so.375.66  libvdpau_nvidia.so.1
libGLdispatch.so.0             libGL.so.1.0.0              libnvidia-eglcore.so.375.66          libnvidia-glsi.so.375.66    libvdpau_nvidia.so.375.66
libGLESv1_CM_nvidia.so.1       libGLX_indirect.so.0        libnvidia-egl-wayland.so.375.66      libnvidia-ifr.so.1
libGLESv1_CM_nvidia.so.375.66  libGLX_nvidia.so.0          libnvidia-encode.so.1                libnvidia-ifr.so.375.66
libGLESv1_CM.so.1              libGLX_nvidia.so.375.66     libnvidia-encode.so.375.66           libnvidia-ml.so.1
libGLESv2_nvidia.so.2          libGLX.so.0                 libnvidia-fatbinaryloader.so.375.66  libnvidia-ml.so.375.66

/var/lib/nvidia-docker/volumes/nvidia_driver/375.66/lib64:
libcuda.so               libGLESv1_CM_nvidia.so.1       libGL.so.1.0.0           libnvidia-compiler.so.375.66         libnvidia-fbc.so.375.66     libnvidia-opencl.so.1
libcuda.so.1             libGLESv1_CM_nvidia.so.375.66  libGLX_indirect.so.0     libnvidia-eglcore.so.375.66          libnvidia-glcore.so.375.66  libnvidia-opencl.so.375.66
libcuda.so.375.66        libGLESv1_CM.so.1              libGLX_nvidia.so.0       libnvidia-egl-wayland.so.375.66      libnvidia-glsi.so.375.66    libnvidia-ptxjitcompiler.so.375.66
libEGL_nvidia.so.0       libGLESv2_nvidia.so.2          libGLX_nvidia.so.375.66  libnvidia-encode.so.1                libnvidia-ifr.so.1          libnvidia-tls.so.375.66
libEGL_nvidia.so.375.66  libGLESv2_nvidia.so.375.66     libGLX.so.0              libnvidia-encode.so.375.66           libnvidia-ifr.so.375.66     libOpenGL.so.0
libEGL.so.1              libGLESv2.so.2                 libnvcuvid.so.1          libnvidia-fatbinaryloader.so.375.66  libnvidia-ml.so.1           libvdpau_nvidia.so.1
libGLdispatch.so.0       libGL.so.1                     libnvcuvid.so.375.66     libnvidia-fbc.so.1                   libnvidia-ml.so.375.66      libvdpau_nvidia.so.375.66
```

我们不建议基于通过查找找到库的解决方案, 因为您可能从较旧的驱动程序安装中选择流浪库. 
我们建议使用驱动程序版本对该卷的名称进行后缀, 这样可以防止错误驱动程序/库版本不匹配, 如果您更新驱动程序但忘记重新创建一个新的卷. 

### GPU隔离

#### 挑战

GPU在`/dev`中显示为独立的设备文件, 以及其他设备：

```
$ ls -l /dev/nvidia*
crw-rw-rw- 1 root root 195,   0 Jul 10 10:03 /dev/nvidia0
crw-rw-rw- 1 root root 195,   1 Jul 10 10:03 /dev/nvidia1
crw-rw-rw- 1 root root 195,   2 Jul 10 10:03 /dev/nvidia2
crw-rw-rw- 1 root root 195,   3 Jul 10 10:03 /dev/nvidia3
crw-rw-rw- 1 root root 195, 255 Jul 10 10:03 /dev/nvidiactl
crw-rw-rw- 1 root root 246,   0 Jul 10 10:03 /dev/nvidia-uvm
crw-rw-rw- 1 root root 246,   1 Jul 10 10:03 /dev/nvidia-uvm-tools
```

Docker允许用户在启动容器(`docker run`)时通过参数`--device`来放行对特定设备的访问(使用`cgroup`). 在多GPU机器上, 常见的用例是并行启动多个作业, 每个作业都使用可用GPU的子集. 最基本的解决方案是使用环境变量`CUDA_VISIBLE_DEVICES`, 但这并不能保证正确的隔离. 通过在Docker中使用设备白名单, 您可以限制容器可以访问的GPU. 例如, 容器A被授予访问`/dev/nvidia0`, 而容器B被授予对`/dev/nvidia1`的访问权限. 设备`/dev/nvidia-uvm`和`/dev/nvidiactl`不对应于GPU, 并且对于所有容器都可以访问它们. 

第一个挑战是将设备文件(或换句话说, 次要设备数量)映射到PCI总线排序(由nvidia-smi报告). 当您的机器上有不同型号的GPU并且您想要将容器特别分配给一个GPU时, 这一点非常重要.  nvidia-smi报告的GPU编号并不总是匹配设备文件的次要号码：

```
$ nvidia-smi -q
GPU 0000:05:00.0
 Minor Number: 3

GPU 0000:06:00.0
 Minor Number: 2
```

第二个挑战与`nvidia_uvm`内核模块相关，它在引导时不会自动加载，因此`/dev/nvidia-uvm`不会被创建，并且容器可能没有足够的权限来加载内核模块本身。 在启动任何CUDA容器之前，必须手动加载内核模块。

#### nvidia-docker

使用`NVML`库中的函数`nvmlDeviceGetCount`枚举GPU，并使用函数`nvmlDeviceGetMinorNumber`获取相应的设备。 如果设备次要号码为N，则设备文件只是`/dev/nvidiaN`。 使用环境变量`NV_GPU`控制隔离，通过传递GPU的索引来隔离，例如：

```
$ NV_GPU=0,1 nvidia-docker run -ti nvidia/cuda nvidia-smi
```

nvidia-docker包装器将找到相应的设备文件，并将`--device`参数添加到命令行，然后将控制权传递给docker。

要手动加载`nvidia_uvm`并创建`/dev/nvidia-uvm`，我们在启动`nvidia-docker-plugin`守护程序时，在主机上使用命令`nvidia-modprobe -u -c = 0`。

#### 替代选择

如果你不想使用`nvidia-docker`

```
$ docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0
```

这需要与用于安装包含用户级驱动程序库的卷的命令行参数结合使用。 可以使用来自NVML的nvmlDeviceGetCount或CUDA API的cudaGetDeviceCount列出可用的GPU。 我们建议使用NVML，因为它还提供nvmlDeviceGetMinorNumber来查找要装载的设备文件。 如果要使用NVML或nvidia-smi收集使用率指标，则必须在设备文件和孤立的GPU之间进行正确的映射。 如果您仍然想使用CUDA API，请确保取消设置环境变量`CUDA_VISIBLE_DEVICES`，否则系统上的某些GPU可能不会被列出。

要加载nvidia_uvm，您还应该使用`nvidia-modprobe -u -c = 0`（如果可用）。 如果不是，您需要手动执行mknod。

### 镜像检查

#### 挑战

安装用户级驱动程序库和设备文件会破坏容器的环境，只有当容器运行GPU应用程序时才应该执行此操作。 这里的挑战是确定给定的图像是否将使用GPU。 我们还应该防止基于与主机NVIDIA驱动程序版本不兼容的Docker映像启动容器，您可以在此wiki页面上找到更多详细信息。

#### nvidia-docker

没有通用的解决方案来检测是否有任何镜像将使用GPU代码。在nvidia-docker中，我们假设基于我们的nvidia / cuda映像（DockerHub上可用）的任何映像都将是GPU应用程序，因此它们需要驱动程序卷和设备文件。
更具体地说，当使用nvidia-docker运行时，我们检查在命令行中指定的镜像。在这个镜像中，我们查找标签`com.nvidia.volumes.needed`的存在和值。 我们提供的nvidia / cuda图像开始时都包含这个标签。执行`FROM nvidia / cuda`的所有Docker文件将自动继承此元数据，从而可以与nvidia-docker无缝工作。

为了检测镜像与主机驱动程序不兼容，我们依赖于第二个元数据，即`com.nvidia.cuda.version`标签。此标签存在于每个CUDA基本镜像中，并具有相应的版本号。该版本与驱动程序支持的最大CUDA版本进行比较，nvidia-docker为此使用CUDA API函数cudaDriverGetVersion。如果驱动程序对于运行此版本的CUDA来说太旧了，则在启动容器之前会出现错误：

```
$ nvidia-docker run --rm nvidia/cuda
nvidia-docker | 2016/04/21 21:41:35 Error: unsupported CUDA version: driver 7.0 < image 7.5
```

#### 替代选择

在这种情况下，nvidia-docker不会简单地将参数注入docker命令行。 因此，重现这种行为更为复杂。 您将需要在工作流程或容器编排解决方案的上游检查图像。 查看图像内的标签很简单：

```
$ docker inspect -f '{{index .Config.Labels "com.nvidia.volumes.needed"}}' nvidia/cuda
nvidia_driver
$ docker inspect -f '{{index .Config.Labels "com.nvidia.cuda.version"}}' nvidia/cuda
8.0.61
```

If you build your own custom CUDA images, we suggest you to reuse the same labels for compatibility reasons.
