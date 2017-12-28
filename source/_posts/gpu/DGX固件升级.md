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
- BIOS 3A04
- VBIOS 86.00.41.00.05
- PSU FirmwareUpdate
- DGX BaseOS v3.1.2

百度云下载[地址](https://pan.baidu.com/s/1jIxOScQ) 密码ctlz

# VBIOS

[介绍页面](https://nvidia-esp.custhelp.com/app/answers/announcement_detail/a_id/4577)

采用容器安装

下载gz包，从[这里](https://dgxdownloads.nvidia.com/custhelp/dgx1/BIOS/nvidia-dgx-fw-0101-20171023.tar.gz)下载， 或者从百度网盘中寻找`nvidia-dgx-fw-0101-20171023.tar.gz`。
`
