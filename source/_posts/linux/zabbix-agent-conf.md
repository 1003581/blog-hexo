---
title: Zabbix Agent配置文件详解
date: 2017-12-21 16:00:50
tags: 
- zabbix
categories: linux
---

参考自[http://blog.51cto.com/lookingdream/1839558](http://blog.51cto.com/lookingdream/1839558)
<!-- more -->

```
# This is a config file for the Zabbix agent daemon (Unix)
# To get more information about Zabbix, visit http://www.zabbix.com

############ GENERAL PARAMETERS #################

### Option: PidFile
#	Name of PID file.
#
# Mandatory: no
# Default:
# PidFile=/tmp/zabbix_agentd.pid
# PidFile=PID路径
# 说明：指定程程序PIDFILE路径，可修改到其它路径，但SNC不建议修改

PidFile=/var/run/zabbix/zabbix_agentd.pid

### Option: LogFile
#	Name of log file.
#	If not set, syslog is used.
#
# Mandatory: no
# Default:
# LogFile=
# LogFile=路径
# 说明：客户端AGENT运行产生的日志文件路径，可修改到其它路径，如/var/log/zabbix_agnetd.log，视具体情况修改，也可保持默认

LogFile=/var/log/zabbix-agent/zabbix_agentd.log

### Option: LogFileSize
#	Maximum size of log file in MB.
#	0 - disable automatic log rotation.
#
# Mandatory: no
# Range: 0-1024
# Default:
# LogFileSize=1
# LogFileSize=数字
# 说明：AGENT产生日志大小控制，默认1M，若为0，则表示不产生任何日志，数字范围（1-1024M）不建议关闭日志功能，建议保持默认

LogFileSize=0

### Option: DebugLevel
#	Specifies debug level
#	0 - no debug 无日志级别
#	1 - critical information 灾难信息级别
#	2 - error information 一般错误信息级别
#	3 - warnings 警告级别
#	4 - for debugging (produces lots of information) 调试级别
# 说明：0~4级别，日志产生量在相同单位时间，生成的日志量为递增，即0级别日志量最少，4级别最多，默认3级别，建议视具体情况，自行把握
#
# Mandatory: no
# Range: 0-4
# Default:
# DebugLevel=3

### Option: SourceIP
#	Source IP address for outgoing connections.
#
# Mandatory: no
# Default:
# SourceIP=
# SourceIP=IP地址
# 说明：当系统设置有多个IP时，需要指定一个IP与二级代理或服务端通信，若系统只有一个IP，也建议指定一个IP

### Option: EnableRemoteCommands
#	Whether remote commands from Zabbix server are allowed.
#	0 - not allowed
#	1 - allowed
#
# Mandatory: no
# Default:
# EnableRemoteCommands=0
# EnableRemoteCommands=0或1
# 说明：是否允许在本地执行远程命令，建议设置为“允许”，因为SNC对命令下发功能进行了二次开发，功能强大，极大的方便日志运维工作

### Option: LogRemoteCommands
#	Enable logging of executed shell commands as warnings.
#	0 - disabled 不产生日志
#	1 - enabled 产生日志
#
# Mandatory: no
# Default:
# LogRemoteCommands=0
# LogRemoteCommands=1或0
# 说明:在参数EnableRemoteCommands=1的情况下，执行远程命令是否保存操作日志，若已设置EnableRemoteCommands=1
# 建议LogRemoteCommands=1，以便日后查证。若EnableRemoteCommands=0，此参数不生效

##### Passive checks related
# 与被动模式有关的参数设置
# 什么是被动模式？
# 被动模式下，由二级代理或服务端主动请求AGENT，去获取所采集到的监控数据

### Option: Server
#	List of comma delimited IP addresses (or hostnames) of Zabbix servers.
#	Incoming connections will be accepted only from the hosts listed here.
#	If IPv6 support is enabled then '127.0.0.1', '::127.0.0.1', '::ffff:127.0.0.1' are treated equally.
#
# Mandatory: no
# Default:
# Server=
# Server=IP地址或主机名，建议IP地址
# 说明：在有二级代理情况下，此IP地址应该填写二级代理服务器的IP，反之，若无二级代理服务器，则此IP应设置为服务端IP

Server=127.0.0.1

### Option: ListenPort
#	Agent will listen on this port for connections from the server.
#
# Mandatory: no
# Range: 1024-32767 （监控端口范围）
# Default:
# ListenPort=10050
# ListenPort=数字
# 说明：此AGENT端以本地服务的形式运行，需要监听端口，强烈建议设置为10050，以便移动整个系统统一规划管理，当然，特殊情况下可修改为1024-32767 未使用的端口

### Option: ListenIP
#	List of comma delimited IP addresses that the agent should listen on.
#	First IP address is sent to Zabbix server if connecting to it to retrieve list of active checks.
#
# Mandatory: no
# Default:
# ListenIP=0.0.0.0
# ListenIP=IP地址
# 说明：对应的ListenPort监听到哪个IP上面，建议指定IP时，不用0.0.0.0

### Option: StartAgents
#	Number of pre-forked instances of zabbix_agentd that process passive checks.
#	If set to 0, disables passive checks and the agent will not listen on any TCP port.
#
# Mandatory: no
# Range: 0-100 数字范围（0-100）
# Default:
# StartAgents=3
# StartAgents=数字
# 说明：在被动模式下，此参数用于设置控制监听进程可启用的子进程的数量，若监控项较多且采集很频繁，建议加大此数值;
# 若此数值为0，则禁止使用被动模式。另外，一般情况，不建议修改此数值，当且仅当某些监控项无法采集到数据，或数据采集数据有延迟现象时，可调整。第四，启用线程越多，则相对越耗系统资源

##### Active checks related
# 主动模式相关参数设置
# 什么时主动模式？
# 在主动模式下，AGENT端（即采集客户端）将所采集的结果，主动提交给二级代理服务器或服务器，而此种情况下，二级代理服务器或服务器将被动接收采集信息

### Option: ServerActive
#	List of comma delimited IP:port (or hostname:port) pairs of Zabbix servers for active checks.
#	If port is not specified, default port is used.
#	IPv6 addresses must be enclosed in square brackets if port for that host is specified.
#	If port is not specified, square brackets for IPv6 addresses are optional.
#	If this parameter is not specified, active checks are disabled.
#	Example: ServerActive=127.0.0.1:20051,zabbix.domain,[::1]:30051,::1,[12fc::1]
#
# Mandatory: no
# Default:
# ServerActive=
# ServerActive=IP地址或IP地址：端口号
# 说明：在主动模式下，ServerActive为二级代理服务器或服务器，默认端口为10051，若需更改端口，则为#ServerActive=IP:port,非特殊情况下，不建议修改。

ServerActive=127.0.0.1

### Option: Hostname
#	Unique, case sensitive hostname.
#	Required for active checks and must match hostname as configured on the server.
#	Value is acquired from HostnameItem if undefined.
#
# Mandatory: no
# Default:
# Hostname=
# Hostname=主机名
# 说明：手工自定义一个主机名，可以和系统的主机名一样，也可以不一样，此参数可根据实际情况启用或关闭，建议关闭此参数，并启用HostnameItem参数

Hostname=Zabbix server

### Option: HostnameItem
#	Item used for generating Hostname if it is undefined. Ignored if Hostname is defined.
#	Does not support UserParameters or aliases.
#
# Mandatory: no
# Default:
# HostnameItem=system.hostname
# HostnameItem：自动获取主机名
# 说明：system.hostname是ZABBIX内置的一个自动获取主机名的方法，为了方便配置，建议打开此参数而关闭Hostname参数#。注意：HostnameItem的优化级低于Hostname，当两个参数都启用且配置的情况下，ZABBIX获取的主机名，将以Hostname为#准

### Option: HostMetadata
#	Optional parameter that defines host metadata.
#	Host metadata is used at host auto-registration process.
#	An agent will issue an error and not start if the value is over limit of 255 characters.
#	If not defined, value will be acquired from HostMetadataItem.
#
# Mandatory: no
# Range: 0-255 characters 0-255个字符
# Default:
# HostMetadata=
# HostMetadata=字符串
# 说明：用于定义当前主机唯一标识符，最大长度255个，仅适用于自动发现情况下，默认不定义，建议不定义

### Option: HostMetadataItem
#	Optional parameter that defines an item used for getting host metadata.
#	Host metadata is used at host auto-registration process.
#	During an auto-registration request an agent will log a warning message if
#	the value returned by specified item is over limit of 255 characters.
#	This option is only used when HostMetadata is not defined.
#
# Mandatory: no
# Default:
# HostMetadataItem=
# 说明；用于获取主机的HostMetadata，建议保持默认

### Option: RefreshActiveChecks
#	How often list of active checks is refreshed, in seconds.
#
# Mandatory: no
# Range: 60-3600
# Default:
# RefreshActiveChecks=120
# RefreshActiveChecks=数字
# 说明：被监控的主机多久（秒）重新请求二级代理或服务端刷新一监控列表。范围为60-3600秒。ZABBIX运行原理为：，zabbix客户端启动后，在等待RefreshActiveChecks秒后，开始从二级代理或服务端请求并下载监控项信息，保存在本地专门的buffersend中，再过RefreshActiveChecks秒后，重新获取监控项信息。这就是为什么当配置监控项，要过一会才能生效的原因。这个数值，就是等待时间。建议，不要将此数值设置过小，以免加大AGENT端和服务端及数据库的压力，建议为120秒。

### Option: BufferSend
#	Do not keep data longer than N seconds in buffer.
#
# Mandatory: no
# Range: 1-3600
# Default:
# BufferSend=5
# BufferSend=数字
# 说明：多少秒后，将BUFFER中的数据提交到二级代理或服务端。范围（1-36600）此数值的大小决定了采集后，提交数据的及时性，数值越小，则提交得越频繁，对服务器压力越大，同时对AGENT端系统资源消耗越大，则表现出来的现象是报警非常及时，建议根据实际情况自行考虑，也可保持默认，若发现ZABBIX消耗资源较多，建议加大此数值。

### Option: BufferSize
#	Maximum number of values in a memory buffer. The agent will send
#	all collected data to Zabbix Server or Proxy if the buffer is full.
#
# Mandatory: no
# Range: 2-65535
# Default:
# BufferSize=100
# BufferSize=数值
# 说明：此参数作用设置保存采集数据在内存中的容量大小。若此agent端监控项较多，建议加大此数值。BufferSize与BufferSend之间有联系的。当达到bUFFERSEND或Buffersize已满时，都会触发数据提交动作。

### Option: MaxLinesPerSecond
#	Maximum number of new lines the agent will send per second to Zabbix Server
#	or Proxy processing 'log' and 'logrt' active checks.
#	The provided value will be overridden by the parameter 'maxlines',
#	provided in 'log' or 'logrt' item keys.
#
# Mandatory: no
# Range: 1-1000
# Default:
# MaxLinesPerSecond=100
# MaxLinesPerSecond=数值
# 说明：定义了AGENT在1秒内发送的日志行数，用于避免网络或cpu过载，建议保持默认

############ ADVANCED PARAMETERS #################
# 高级参数设置

### Option: Alias
#	Sets an alias for an item key. It can be used to substitute long and complex item key with a smaller and simpler one.
#	Multiple Alias parameters may be present. Multiple parameters with the same Alias key are not allowed.
#	Different Alias keys may reference the same item key.
#	For example, to retrieve the ID of user 'zabbix':
#	Alias=zabbix.userid:vfs.file.regexp[/etc/passwd,^zabbix:.:([0-9]+),,,,\1]
#	Now shorthand key zabbix.userid may be used to retrieve data.
#	Aliases can be used in HostMetadataItem but not in HostnameItem parameters.
#
# Mandatory: no
# Range:
# Default:
# 设置参数的别名。它可以替代长和复杂的一个小而简单的一个有用的参数名称

### Option: Timeout
#	Spend no more than Timeout seconds on processing
#
# Mandatory: no
# Range: 1-30
# Default:
# Timeout=3
# Timeout=数值
# 说明：当agent采集一个数据时，多长少算超时。建议保持默认

### Option: AllowRoot
#	Allow the agent to run as 'root'. If disabled and the agent is started by 'root', the agent
#	will try to switch to user 'zabbix' instead. Has no effect if started under a regular user.
#	0 - do not allow
#	1 - allow
#
# Mandatory: no
# Default:
# AllowRoot=0
# AllowRoot=0或1
# 说明：是否允许ROOT帐号运行此客户端。0：不允许，1:允许，当一个脚本执行需要以ROOT身份执行的，则此开关必须打开，建议根据实际情况开启或关闭，

### Option: Include
#	You may include individual files or all files in a directory in the configuration file.
#	Installing Zabbix will create include directory in /etc/zabbix, unless modified during the compile time.
#
# Mandatory: no
# Default:
# Include=
# Include=目录路径或扩展配置文件路径
# 说明：从配置文件可管理性或扩展性考虑，若需配置大量参数的且为了方便后续管理可以启用此参数，建议根据实际情况考虑，不过，一般情况下无须启用
# Include=/etc/zabbix/zabbix_agentd.userparams.conf
# Include=/etc/zabbix/zabbix_agentd.conf.d/
Include=/etc/zabbix/zabbix_agentd.conf.d/

####### USER-DEFINED MONITORED PARAMETERS #######
# 自定义监控脚本

### Option: UnsafeUserParameters
#	Allow all characters to be passed in arguments to user-defined parameters.
#	0 - do not allow
#	1 - allow
#
# Mandatory: no
# Range: 0-1
# Default:
# UnsafeUserParameters=0
# UnsafeUserParameters=0或1
# 说明：是否启用用户自定义监控脚本，1启用，0不启用。由于ZABBIX实现监控方法的多样性，一般都采用脚本来实现监控数据的采集，所以，建议开启，否则功能将受限。

### Option: UserParameter
#	User-defined parameter to monitor. There can be several user-defined parameters.
#	Format: UserParameter=<key>,<shell command>
#	See 'zabbix_agentd' directory for examples.
#
# Mandatory: no
# Default:
# UserParameter=

####### LOADABLE MODULES #######

### Option: LoadModulePath
#	Full path to location of agent modules.
#	Default depends on compilation options.
#
# Mandatory: no
# Default:
# LoadModulePath=${libdir}/modules
# 说明：扩展模块路径，强烈建议不要改动，除非你具有AGENT 开发能力

### Option: LoadModule
#	Module to load at agent startup. Modules are used to extend functionality of the agent.
#	Format: LoadModule=<module.so>
#	The modules must be located in directory specified by LoadModulePath.
#	It is allowed to include multiple LoadModule parameters.
#
# Mandatory: no
# Default:
# LoadModule=
# 说明：扩展模块路径，强烈建议不要改动，除非你具有AGENT 开发能力



# UserParameter=
# 说明：用户自定义监控脚本，当且仅当UnsafeUserParameters=1时UserParameter生效。以下为SNC初始自定监控脚本，不建议修改，已有选项，但可自义添加。
# 自定义监控项配置语法
# UserParameter=key,command
# 如何使用：以获取mysql监控为例
# 步骤1，设置自定义脚本
# UserParameter=mysql.questions,mysqladmin -uroot --password='XXXXX' status|cut -f4 -d":"|cut -f1 -d"S"
# 保存退出，并重启AGENT
# 步骤2，手工验证
# 在二级代理端或服务器端用命令 zabbix_get -s IP -k mysql.questions 将返回采集信息
# 步骤3，在管理页面添加监控项
# 注意:成功关键，脚本本身具有可执行权限，且脚本运行正常

UserParameter=cmd[*],$1 $2 $3 $4 $5 $6 $7 $8 $9
UserParameter=setenv[*],java -jar /smp/sncmon/java/setenv/setenv.jar $1 $2
UserParameter=oracle[*],java -jar /smp/sncmon/java/oracle/oracle.jar $1 /smp/sncmon/java/oracle/
UserParameter=db2[*],/smp/sncmon/java/db2/linux.sh $1
UserParameter=mindwaresctipt,setsid /smp/sncmon/shell/middleware/middle_zabbix.sh
UserParameter=agent.restart,/smp/sncmon/shell/agent/agentrestart.sh
UserParameter=diskmon,/smp/sncmon/shell/diskmon/disk_mon.sh $1
```