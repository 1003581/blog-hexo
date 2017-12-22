---
title: 基于Docker的Zabbix+Grafana监控
date: 2017-09-14 16:00:50
tags: 
- zabbix
- grafana
categories: linux
---

基于Docker进行部署。
<!-- more -->

# 相关资料

- [grafana zabbix 插件 Github](https://github.com/alexanderzobnin/grafana-zabbix)
- [grafana zabbix 插件 Docs](http://docs.grafana-zabbix.org/installation/)

# 部署代码

[部署代码Github](https://github.com/liqiang311/zabbix-grafana)

# 部署步骤

## 准备工作

### 下载部署代码

```
git clone --recursive https://github.com/liqiang311/zabbix-grafana.git
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

填写以下内容，此数据源为Zabbix的数据库，在第二个数据源中会用到。

> 注：为提及的选项均表示不选择。

```
Name: Zabbix
Type: MySQL
MySQL Connection Host: localhost:10053
Database: zabbix
User: root
Password: mysql57
```

然后点击下方 `Save & Test`。若成功连接，则按钮上方会显示绿色信息：

```
Success
Database Connection OK
```

继续添加数据源。内容如下：

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
Direct DB Connection Enable: √
Direct DB Connection SQL Data Source: Zabbix
Alerting Enable alerting: √
Alerting Add thresholds: √
```

然后点击下方 `Save & Test`。若成功连接，则按钮上方会显示绿色信息：

```
Success
Zabbix API version: 3.2.5
```

# 客户端安装

若要监控磁盘使用率、CPU等，需在主机上安装如下软件

```shell
apt-get install zabbix-agent
```

然后编辑如下配置文件

```shell
vim /etc/zabbix/zabbix_agentd.conf
```

将其中的85行的Server改为上文Zabbix配置的IP，如`127.0.0.1`

将其中的126行的ServerActive配置为上文Zabbix的server IP，如`127.0.0.1:10051`

保存文件退出。

重启zabbix-agent

```shell
service zabbix-agent restart
```

# 自定义监控选项

参考[link](http://www.cnblogs.com/jjzd/p/6474193.html)

在`/etc/zabbix/zabbix_agentd.conf`中底部添加如下代码：

```
UserParameter=cpu0.temp, sensors coretemp-isa-0000|grep Physical|awk '{print $4}'|cut -d "." -f1
UserParameter=cpu1.temp, sensors coretemp-isa-0001|grep Physical|awk '{print $4}'|cut -d "." -f1
UserParameter=gpu0.temp, nvidia-smi -q -g 0 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu1.temp, nvidia-smi -q -g 1 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu2.temp, nvidia-smi -q -g 2 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu3.temp, nvidia-smi -q -g 3 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu4.temp, nvidia-smi -q -g 4 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu5.temp, nvidia-smi -q -g 5 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu6.temp, nvidia-smi -q -g 6 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g
UserParameter=gpu7.temp, nvidia-smi -q -g 7 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g

```

重启zabbix-agent `service zabbix-agent restart`

在zabbix web中添加监控项, 其中`键值`填写`gpu0.temp`。

# Zabbix中添加模版

使用场景：公司来了一大批GPU服务器，需要对这么服务器进行监控每个GPU卡的温度和CPU核的温度，以及这些服务器的磁盘使用率。

痛点：监控项太多，手动添加不易维护。

添加模版步骤如下：

在Zabbix Web界面，点击`配置`->`模版`->`创建模版`，填写信息如下：

```
模版名称：gpus
可见的名称：gpus
新的群组：gpus
```

点击`添加`。

然后点击该模版的`监控项`,点击右上角的`创建监控项`。

内容如下：

```
名称：cpu0.temp
类型：Zabbix 客户端
键值：cpu0.temp
信息类型：数字（无正负）
数据类型：十进制数字
```

其他默认，点击`添加`。

然后你需要利用这个监控项，来克隆出`cpu0.temp`,`gpu0.temp`,`gpu1.temp`,`gpu2.temp`,`gpu3.temp`,`gpu4.temp`,`gpu5.temp`,`gpu6.temp`,`gpu7.temp`，路漫漫，加油！

还需要监控磁盘使用情况。

继续添加

```
名称：fs.size
类型：Zabbix 客户端
键值：vfs.fs.size[/,pused]
信息类型：浮点数
```

其他默认，点击`添加`。

最后效果如图：

![img](http://outz1n6zr.bkt.clouddn.com/2017-12-14_161919.png)

继续添加主机。点击 Zabbix Web中的`配置`->`主机`->右上角`创建主机`。

填写如下：

```
主机名称：10.42.10.1
可见的名称：gpu1
agent代理程序的接口 IP地址 10.42.10.1
```

点击上方主机旁边的`模版`,然后点击`选择`，选择刚刚创建的模板，然后选择后，点击`添加`按钮(带下划线的)。见下图

![img](http://outz1n6zr.bkt.clouddn.com/2017-12-14_163541.png)

最后点击`添加`按钮，创建主机成功，可以发现这些主机中已经包含了很多监控项。

依次类推，通过刚刚这个主机，克隆出所有的主机。

> 注意：我们创建的模版中，选择监控了8个gpu卡，但是有的服务器中只有4块或者1块，需要在主机中将这些监控项进行禁用。如下：

![img](http://outz1n6zr.bkt.clouddn.com/2017-12-14_172159.png)

# Gafana中使用Templating

上面我们在Zabbix中添加了对许多卡的监控，现在利用Grafana的模版进行监控。

在Grafana中打开`Dashboards`，然后选择你创建的Dashboards，比如`GPUs`,然后进入该Dashboards首页，点击`设置`中的`Templating`

![img](http://outz1n6zr.bkt.clouddn.com/2017-12-15_091138.png)

选择`Variables`界面的`New`，填写内容如下：

```
Name: 自定义变量名，比如Host
Type: Query
Data source: zabbix
Refresh: On Dashboard Load
Query: Groups.*
Multi-value: ✔
Include All option: ✔
```

> Query说明：格式为Groups.Host.Application.Items，若要创建Group级变量，则填*，若要全部的Host，则填写`*.*`，全部Items则写`*.*.*.*`，上文填写为获取指定Group下的Host

若填写正确，则`Add`按钮上方会出现所有的匹配项（Preview of values(shows max 20)）（前20个）。

点击`Add`。

以此方式再添加Items变量。Query中填写`Group.*.*.*`。

添加后如图。

![img](http://outz1n6zr.bkt.clouddn.com/2017-12-15_092802.png)

接下来是如何使用。

选择新建一个`Graph`Panel，编辑这个Panel，填写方式如下:

```
Query Mode: Metrics
Group: Group
Host: $Host
Item: $Items
Data Source: zabbix
```

![img](http://outz1n6zr.bkt.clouddn.com/2017-12-15_093212.png)

若不想全部Items显示到一个Graph中，则需要修改Graph Panel的`General`选项中的`Templating`，选择`Repeat Panel`，选择要Repeat的方式，`Min span`指的是分开多个后每个panel的最小宽度。最终效果图如下：

![img](http://outz1n6zr.bkt.clouddn.com/2017-12-15_093452.png)

# 定期备份

## Zabbix

使用如下命令将Zabbix的数据库进行备份，备份为sql文件，自行保存到其他服务器。然后添加到Crontab中。

```
docker exec zabbix-mysql mysqldump -uroot -pmysql57 zabbix > init.sql
scp init.sql 10.40.64.206:/root
```

恢复

```
docker cp init.sql zabbix-mysql:/tmp/
docker exec zabbix-mysql mysql -uroot -pmysql57 zabbix -e "source /tmp/init.sql"
```

## Grafana

[参考](http://www.cnblogs.com/xkus/p/7462953.html)

备份

```
docker cp grafana:/var/lib/grafana/grafana.db ./
scp grafana.db 10.40.64.206:/root
```

恢复

```
docker cp grafana.db grafana:/var/lib/grafana/grafana.db
docker restart grafana
```