---
title: win7通过注册表关闭自动睡眠和锁屏
date: 2017-09-14 15:11:28
tags: tool
categories: tool
---

## Win7屏保通过注册表来修改

<!-- more -->

`cmd`打开`regedit`

依次索引如下

```
计算机\HKEY_CURRENT_USER\Software\Policies\Microsoft\Windows\Control Panel\Desktop
```

- 将`ScreenSaveActive`置为`0`
- 将`ScreenSaveIsSecure`置为`1`
- 将`ScreenSaveTimeOut`置为`0`

注销计算机即可。

## XShell自动断开

打开设备管理器, 网卡右键属性, 电源管理, 关闭`允许计算机关闭此设备以节约电源`.