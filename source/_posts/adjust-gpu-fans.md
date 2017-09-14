---
title: 根据温度自动调节显卡风扇速度
date: 2017-09-14 16:03:03
tags: gpu
categories: gpu
---

# 根据温度自动调节显卡风扇

<!-- more -->

操作步骤

```
git clone https://github.com/liqiang311/set-gpu-fans.git
mv set-gpu-fans /opt
apt-get update
apt-get install -y xinit
cd /opt/set-gpu-fans
chmod +x cool_gpu
chmod +x nvscmd
nohup ./cool_gpu &
```

输入`nvidia-smi`,当显示有一个7MB的进程运行时，表示成功。

若失败，检查/opt下的nohup.out

原文参考[http://www.jianshu.com/p/ab956df5e40c](http://www.jianshu.com/p/ab956df5e40c)
