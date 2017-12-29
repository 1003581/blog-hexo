---
title: DGX固件升级
date: 2017-12-27 16:03:03
tags: gpu
categories: gpu
---

由于DGX服务器出现掉卡问题，联系NVIDIA后需要进行固件升级，记录步骤如下
<!-- more -->
# 升级列表

- BMC 3.20.30
- SBIOS 3A04
- VBIOS 86.00.41.00.05
- PSU FirmwareUpdate
- DGX BaseOS v3.1.2（可选，3.1.2为最新的16.04系统）

百度云下载[地址](https://pan.baidu.com/s/1jIxOScQ) 密码ctlz

# BMC

[官网英文教程](http://docs.nvidia.com/dgx/dgx1-user-guide/maintenance.html#task_updating-the-bmc)

注意：下载的为zip包，需要自己解压后，上传`S2W_NVD_v32030.ima_enc`文件。

操作完成后，发生错误，BMC 无法连接，ip无法ping通。

解决方法：

1. 将S2W_NVD_v32030.zip传到dgx服务器，然后执行如下命令
    ```shell
    unzip S2W_NVD_v32030.zip
    cd S2W_NVD_v32030
    chmod 755 -R .
    ./linux.sh
    ```
1. 等待执行结束后，关机，拔电源线，过五分钟后重启。

此时bmc ip会被重置，需要开机时按F11，进入BIOS->server mgr->bmc network configuration，修改为原本的静态地址。同时登录BMC系统后，原本的帐号密码被重置了，重置为qct.admin/qct.admin

> 猜测：ip和帐号密码被重置，是因为执行了第一步命令（这些命令已经之前在BMC网页上操作过了，且BMC网站上指定了保存IPMI），怀疑出现问题时断电重启是否就可以了。（等待确认）。

# SBIOS

[官网教程](http://docs.nvidia.com/dgx/dgx1-user-guide/maintenance.html#task_updating-the-sbios)

此步骤需要先关机（从BMC中操作），然后再从BMC中开机。

使用压缩包中的`S2W_3A04.BIN`文件。

# VBIOS

按照`DGX-1-VBIOS-update-860041_v03.pdf`文件进行，使用`vbios41.tar.gz`文件，该压缩文件中只有一个可执行文件。

注意：在执行这个可执行文件前，需要把nvidia相关的进程全部停止，包括：

- nvidia-persistenced
- nvidia-docker
- nvidia_uvm
- nvidia_drm
- nvidia_modeset
- nv_peer_mem
- nvidia

可以通过以下命令来查看是否有nvidia相关的进程运行

```
ps aux|grep nvidia
```

若有，则强行杀掉，之后输入

```
lsmod |grep nvidia
```

若输出为空，表示正常，否则需重新停止相关进程和模块。

注意：若机器中运行有定时程序、监控程序等去定期执行nvidia-smi等命令，nvidia相关模块会自动被加载，所以在升级VBIOS前必须停止相关调用。

一切就绪后执行脚本即可。

# PSU

教程`DGX-1-PSU-recalibration.pdf`文件，拷贝`ib_LiteonPSU_Calitool.sh`脚本，执行如下命令

···
chmod +x ib_LiteonPSU_Calitool.sh
./ib_LiteonPSU_Calitool.sh 1
···
