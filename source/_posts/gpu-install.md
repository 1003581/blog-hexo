---
title: GPU集群环境一键式部署
date: 2017-11-27 23:30:00
tags: gpu
categories: linux
---

下载各个安装包

<!-- more -->

- [下载特定NVIDIA驱动](http://www.nvidia.com/Download/Find.aspx?lang=cn)
- [下载Nvidia-docker](https://github.com/NVIDIA/nvidia-docker/releases)
- [下载Docker-Compose](https://github.com/docker/compose/releases)
- [下载MLNX_OFED](http://ch.mellanox.com/page/products_dyn?product_family=26&mtag=linux_sw_drivers)

## 安装GPU驱动

```shell
svn export svn://10.42.10.xx/gpu/install/nvidia-diag-driver-local-repo-ubuntu1404-384.81_1.0-1_amd64.deb --username username --password passwd --no-auth-cache
dpkg -i nvidia-diag-driver-local-repo-ubuntu1404-384.81_1.0-1_amd64.deb
apt-key add /var/nvidia-diag-driver-local-repo-384.81/7fa2af80.pub
apt-get update -qq
apt-get install -y -qq cuda-drivers
nvidia-smi
rm nvidia-diag-driver-local-repo-ubuntu1404-384.81_1.0-1_amd64.deb
```

需重启

## 安装Docker、Docker-Compose、Nvidia-Docker

公网安装见[安装Docker、Docker-Compose](http://liqiang311.com/docker-install/)

```shell
wget -qO- http://gitlab.zte.com.cn/mirrors/tunasync/raw/zte/import-docker-pgp-key.sh | sudo sh
echo "deb http://mirrors.zte.com.cn/docker/apt/repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list
apt-get remove lxc-docker*
apt-get update -qq
apt-get install -y -qq docker-engine
docker --version

svn export svn://10.42.10.xx/gpu/install/docker-compose-Linux-x86_64 --username username --password passwd --no-auth-cache
mv docker-compose-Linux-x86_64 /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
docker-compose --version

svn export svn://10.42.10.xx/gpu/install/nvidia-docker_1.0.1-1_amd64.deb --username username --password passwd --no-auth-cache
dpkg -i nvidia-docker_1.0.1-1_amd64.deb
rm nvidia-docker_1.0.1-1_amd64.deb
nvidia-docker-plugin -v
```

## 安装ib驱动

```shell
svn export svn://10.42.10.xx/gpu/install/MLNX_OFED_LINUX-4.2-1.0.0.0-ubuntu14.04-x86_64.tgz --username username --password passwd --no-auth-cache
tar zxvf MLNX_OFED_LINUX-4.2-1.0.0.0-ubuntu14.04-x86_64.tgz
rm MLNX_OFED_LINUX-4.2-1.0.0.0-ubuntu14.04-x86_64.tgz
apt-get update -qq && apt-get install -y -qq perl dpkg autotools-dev autoconf libtool automake1.10 automake m4 dkms debhelper tcl tcl8.4 chrpath swig graphviz tcl-dev tcl8.4-dev tk-dev tk8.4-dev bison flex dpatch zlib1g-dev curl libcurl4-gnutls-dev python-libxml2 libvirt-bin libvirt0 libnl-dev libglib2.0-dev libgfortran3 automake m4 pkg-config libnuma-dev libnuma1 logrotate ethtool lsof gfortran quilt
./MLNX_OFED_LINUX-4.2-1.0.0.0-ubuntu14.04-x86_64/mlnxofedinstall --force
rm -rf MLNX_OFED_LINUX-4.2-1.0.0.0-ubuntu14.04-x86_64
/etc/init.d/openibd restart
/etc/init.d/opensmd start
```

## 配置Zabbix-Agent

```shell
apt-get update -qq
apt-get install -y -qq zabbix-agent lm-sensors
sed -i 's/^Server=127.0.0.1/Server=10.42.10.xx/g' /etc/zabbix/zabbix_agentd.conf
sed -i 's/^ServerActive=127.0.0.1/ServerActive=10.42.10.xx:10051/g' /etc/zabbix/zabbix_agentd.conf
echo "UserParameter=gpu0.temp, nvidia-smi -q -g 0 2>&1|grep -i "gpu current temp"|awk '{print $5}'| sed s/\%//g" >> /etc/zabbix/zabbix_agentd.conf
echo "UserParameter=cpu0.temp, sensors coretemp-isa-0000|grep Physical|awk '{print $4}'|cut -d "." -f1" >> /etc/zabbix/zabbix_agentd.conf
echo "UserParameter=cpu1.temp, sensors coretemp-isa-0001|grep Physical|awk '{print $4}'|cut -d "." -f1" >> /etc/zabbix/zabbix_agentd.conf
service zabbix-agent restart
```