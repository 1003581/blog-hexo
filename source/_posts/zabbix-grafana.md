---
title: 使用Zabbix+Grafana监控
date: 2017-09-14 16:00:50
tags: 
- zabbix
- grafana
categories: linux
---

# [部署代码](https://github.com/liqiang311/zabbix-grafana)

<!-- more -->

# 部署步骤

## 准备工作

### 下载部署代码，并且下载granafa插件

```
git clone https://github.com/liqiang311/zabbix-grafana.git
git clone https://github.com/alexanderzobnin/grafana-zabbix.git zabbix-grafana/grafana/plugins/grafana-zabbix
```

### 下载docker镜像

如何安装Docker和Docker-Compose见[http://liqiang311.com/docker-install/](http://liqiang311.com/docker-install/)

```
docker pull mysql:5.7
docker pull zabbix/zabbix-server-mysql:latest
docker pull zabbix/zabbix-web-nginx-mysql:latest
docker pull grafana/grafana:latest
```

## 启动命令

```
cd zabbix-grafana
docker-compose up -d
```

# 配置

## Zabbix

登录`ip:10052`，帐号为`Admin`，密码为`zabbix`

进入后右上角可以更改语言为中文

## Grafana

web`ip:3000`

默认帐号`admin`/`admin`

`Plugins`->`app`->`Zabbix`->点击Enable

### 添加Data Source

点击左上角Grafana图标，选择`Data Sources`->`Add data Source`

填写以下内容

```
Name: zabbix
Type: Zabbix
url: http://localhost:10052/api_jsonrpc.php
Access: proxy
Basic Auth: √
Basic Auth Details User: admin
Basic Auth Details Password: zabbix
Zabbix API details Username: admin
Zabbix API details Password: zabbix
```

然后点击下方 `Save & Test`。若成功连接，则按钮上方会显示绿色信息：

```
Success
Zabbix API version: 3.2.5
```

# 客户端安装

若要监控磁盘使用率、CPU等，需在主机上安装如下软件

```bash
apt-get install zabbix-agent
```

然后编辑如下配置文件

```bash
vim /etc/zabbix/zabbix_agentd.conf
```

将其中的85行的Server改为上文Zabbix配置的IP，如`127.0.0.1`

将其中的126行的ServerActive配置为上文Zabbix的server IP，如`127.0.0.1:10051`

保存文件退出。

重启zabbix-agent

```
service zabbix-agent restart
```

# 自定义监控选项

参考[link](http://www.cnblogs.com/jjzd/p/6474193.html)

在`/etc/zabbix/zabbix_agentd.conf`中底部添加如下代码：

```
UserParameter=gpu0.temp, nvidia-smi -q -g 0 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu1.temp, nvidia-smi -q -g 1 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g

UserParameter=cpu0.temp, sensors coretemp-isa-0000|grep Physical|awk '{print $4}'|cut -d "." -f1
UserParameter=cpu1.temp, sensors coretemp-isa-0001|grep Physical|awk '{print $4}'|cut -d "." -f1
```

重启zabbix-agent `service zabbix-agent restart`

在server中添加监控, 键值名为 gpu0.temp


# 相关资料

- [grafana zabbix 插件 Github](https://github.com/alexanderzobnin/grafana-zabbix)
- [grafana zabbix 插件 Docs](http://docs.grafana-zabbix.org/installation/)
