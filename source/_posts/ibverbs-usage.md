---
title: ibverbs文档翻译
date: 2017-09-14 15:58:43
tags: 
- rdma
- ibverbs
categories: c++
---

[百度云下载](https://pan.baidu.com/s/1o78iHmi)

<!-- more -->

# 产品简介

## 官方驱动及文档地址： 

[http://www.mellanox.com/page/products_dyn?product_family=26&mtag=linux_sw_drivers](http://www.mellanox.com/page/products_dyn?product_family=26&mtag=linux_sw_drivers)

![image](http://www.mellanox.com/uploads/product_families/cat_72/gfx_00512.jpg)

## 文档列表
- [Product Brief](http://www.mellanox.com/related-docs/prod_software/PB_OFED.pdf)
- [RDMA Aware Networks Programming User Manual](http://www.mellanox.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf) 
- [Performance Tuning Guide for Mellanox Network Adapters](http://www.mellanox.com/related-docs/prod_software/Performance_Tuning_Guide_for_Mellanox_Network_Adapters.pdf)

# Product Brief (PB_OFED.pdf)

为Field-Proven RDMA and Transport Offload Hardware Solutions的高性能服务器和存储连接软件

## Benifits
- 支持同一适配卡上的InfiniBand和以太网连接
- 在所有Mellanox InfiniBand和以太网设备上运行的单个软件堆栈
- 支持HPC应用，如科学研究，油气勘探，汽车碰撞测试
- 用户级verbs允许诸如MPI和UDAPL之类的协议接口到Mellanox InfiniBand和10 / 40GbE RoCE硬件。内核级别verbs允许诸如SDP，iSER，SRP之类的协议接口到Mellanox InfiniBand和10 / 40GbE RoCE硬件。
- 通过Mellanox消息加速（MXM）提高性能和可扩展性
- RoCEv2使L3路由能够提供更好的隔离，并实现超大规模Web2.0和云部署，具有出色的性能和效率
- 支持数据中心应用程序，如Oracle 11g RAC，IBM DB2，Purescale金融服务低延迟消息传递应用程序，如IBM WebSphere LLM，Red Hat MRG Tibco
- 支持大数据分析，例如RDMA NYSE Data Fabric上的Hadoop
- 支持使用RDMA优势的高性能存储应用程序
- 通过KVM和XenServer支持I / O虚拟化技术，如SR-IOV和半虚拟化
- SDP和IPoIB组件使得TCP / IP和基于套接字的应用程序能够受益于InfiniBand传输
- 在用户和内核级别支持OpenFabrics定义的Verbs API。
- SCSI中间层接口，支持基于SCSI的块存储和管理

## Mellanox OpenFabrics 企业发行版(Enterprise Distribution) for Linux（MLNX_OFED）
使用商品servers和存储系统的集群在不断增长的大型市场中被广泛部署，如高性能计算、数据仓库、在线交易处理、金融服务和大规模的Web2.0部署。为了实现高效率地透明式分布式计算，市场中的应用需要最高的I/O带宽和尽可能低的延迟。这些需求与提供一个网络、虚拟化、存储和其他应用程序和接口的大型交互式生态系统相结合。

开放式结构联盟[www.openfabrics.org](www.openfabrics.org)的OFED已经通过与提供高性能I/O的供应商的协作开发测试来加强。Mellanox OFED（MLNX_OFED）是一种Mellanox测试和打包版本的OFED，并支持使用相同的RDMA（远程DMA）和内核旁路API（称为OFED verbs - InfiniBand and Ethernet)两种互连类型。Mellanox制作的10/20/40Gb/s的InfiniBand和超过10/40GbE的RoCE(基于RDMA over Converged Ethernet标准)使OEM和系统集成商能够满足市场中最终用户的需求。

## Linux Inbox 驱动
适用于以太网和InfiniBand的Mellanox适配器的Linux VPI驱动程序也可用于所有主要发行版，RHEL、SLES、Ubuntu等等的Inbox。Inbox驱动程序使用Mellanox高性能云、HPC、存储、金融服务和更多的企业级Linux发行版的开箱即用的经验。
## VPI Support
该设备通过一个高效的多层设备驱动架构使得在相同硬件I/O适配器（HCA或NIC）上的多重I/O连接选项生效。一台服务器上安装的相同的OFED栈可以deliver(实现)InfiniBand和以太网的同时I/O服务，端口也根据应用和终端用户需求来重新安排。比如适配器上的一个端口可以用来作为标准NIC或者RoCE以太网NIC，而另一个端口何以操作InfiniBand，或者两个端口用用来运行IB后者以太网。这样就使得IT管理人员能够实现功能聚合和高端I/O服务。

## 提供聚合和高端I/O服务
数据中心环境的网络中包含server to server通信（IPC）、server to LAN和server to SAN。在数据中心，特别是那些属于目标市场的应用程序中，应用程序期望针对不同类型的流量进行优化的不同API或接口，以实现最高性能。

## 效率 & 高性能HPC（高性能计算）簇
在HPC应用中，MPI(信息通过接口)已经被广泛作为了并行编程通信库。在正在形成的大规模HPC应用中，I/O瓶颈不仅仅存在于架构中，还被拓展到了通信库。为了对MPI、SHMEM和PGAS应用提供可扩展的解决方法，Mellanox提供了一个新的名叫MXM(MellanoX Messaging)的增量库，该库通过改善内存和延迟的相关效率，使MPI、SHMEM和PGAS编程语言能够扩展到非常大的集群，并确保通信库可以通过Mellanox互连解决方案进行全面优化。

## 金融服务应用的最低延迟
使用InfiniBand、RoCE和VMA的MLNX_OFED实现了金融应用的最低延迟和最高PPS(Packet Per Second)性能。

## 集群数据库应用的最低延迟和带宽
UDAPL是应用程序（如IBM DB2 pureScale）使用的另一个用户级RDMA接口。

## Web2.0和其他传统的基于Socket的应用
包含IP-IB现场验证的MLNX_OFED，使基于IP的应用程序能够在InfiniBand上无缝工作。对于对最低延迟不敏感的应用程序，通过L2 NIC（以太网）驱动程序实现支持标准UDP/TCP/IP Socket接口。

## 存储应用
为了使传统的基于SCSI和iSCSI的存储应用程序享受类似的RDMA性能优势，MLNX_OFED包括基于RDMA协议（SRP）的SCSI启动器和目标，以及与行业中可用的各种目标组件进行互操作的iSCSI RDMA协议（iSER）。 iSER可以
在InfiniBand或RoCE上实施。 MLNX_OFED支持Mellanox存储加速，这是一个统一的计算和存储网络，可以显着优于多种网络的性价比。

## 高可用性 High Availability (HA)
MLNX_OFED包括消息传递，套接字和存储应用程序的高可用性支持。 通过IPoIB支持标准的Linux通道绑定模块，可以跨同一个适配器上的端口或跨适配器进行故障切换。 类似地，使用标准Linux实现（例如设备映射器多路径（dm-multipath）驱动程序）的SCSI应用程序支持通过SRP启动器的标准多路径和故障转移。 还支持一些供应商特定的故障切换/负载平衡驱动程序模型。

# RDMA Aware Networks Programing User Manual

version 1.7

## Contents

1. RDMA Architer Overview
2. RDMA-Aware Programming Overview
3. VPI Verbs API
4. RDMA_CM_API
5. RDMA Verbs API
6. Events
7. Programming Excmples Using IBV verbs
8. Progamming Examples Using RDMA Verbs
9. Experimental APIs
10. Verbs API for Extended Atomics Support
11. User-Mode Memory Registration (UMR)
12. Cross-Channel Communications Support

## Glossary 术语

| Term                                     | Description                              |
| ---------------------------------------- | ---------------------------------------- |
| Access Layer                             | 用于访问互连结构（VPI™，InfiniBand®，以太网，FCoE）的低级操作系统基础设施（管道）<br>它包括支持上层网络协议，中间件和管理代理所需的所有基本传输服务 |
| AH(Address Handle)                       | 描述在UD(Unreliable Datagram) QP(Queue Pair)中使用的远程端的路径的对象 |
| CA(Channel Adapter)                      | 中断InfiniBand链路、执行传输层功能的设备                |
| CI(Channel Interface)                    | 面向Verbs客户的通道介绍，通过网络适配器、相关固件和设备驱动软件的组合实现  |
| CM(Communication Manager)                | 负责建立、维护和发布RC和UC QP(Queue Pair)服务类型通信的实体<br>服务ID解析协议允许UD服务的用户定位支持其所需服务的QP(Queue Pair)<br>终端节点的每个IB端口都有一个CM |
| Compare & Swap                           | 指示远程QP(Queue Pair)读取64位值，将其与提供的比较数据进行比较，如果相等，则将其替换为QP(Queue Pair)中提供的交换数据。 |
| CQ(Completion Queue)                     | 包含CQEs的先进先出队列                            |
| CQE(Completion Queue Entry)              | CQ(Completion Queue)中描述完成WR(Work Request)信息的条目（状态大小等） |
| DMA(Direct Memory Access)                | 允许硬件绕过CPU直接将数据块移入或移出存储器                  |
| Fetch & Add                              | 指示远程QP(Queue Pair)读取64位值，并将其替换为QP(Queue Pair)中提供的64位值和添加的数据值之和 |
| GUID(Global Unique IDentifier)           | 唯一标识子网中设备或组件的64位数                        |
| GID(Global IDentifier)                   | 用于标识网络适配器上的端口、路由器上的端口或组播组的128位标识符<br>GID是有效的128位IPv6地址（根据RFC 2373），在IBA中定义了额外的属性/限制，以促进有效的发现，通信和路由 |
| GRH(Global Routing Header)               | 用于通过子网边界递送数据包并用于传送多播消息的数据包头<br>该Packet报头基于IPv6协议 |
| Network Adapter                          | 允许在网络中的计算机之间进行通信的硬件设备                    |
| Host                                     | 执行可控制一个或多个网络适配器的操作系统的计算机平台               |
| IB                                       | InfiniBand                               |
| Join Operation                           | IB端口必须通过向SA(Subnet Administrator)发送请求来接收多播数据包来显式加入组播组 |
| lkey                                     | 在注册MR(Memory Region)时接收到的号码，由本地WR(Work Request)来使用，以识别内存区域及其相关权限 |
| LID(Local Identifier)                    | 由子网管理器分配给端节点的16位地址<br>每个LID在其子网内是唯一的     |
| LLE(Low Latency Ethernet)                | 通过CEE(Converged Enhanced Ethernet融合增强以太网)的RDMA服务，允许IB通过以太网传输 |
| NA(Network Adapter)                      | 终止链接并执行传输层功能的设备                          |
| MGID(Multicast Group ID)                 | 由MGID标识的IB组播组由SM(Subnet Manager)管理<br>SM将MLID(Multicast Local Identifier)与每个MGID相关联，并显式地对结构中的IB交换机进行编程，以确保所有端口接收到该组播组的数据包 |
| MR(Memory Region)                        | 已经通过准入权限注册的连续的一组内存缓冲区。这些缓冲区需要注册才能使网络适配器使用它们。 在注册期间，创建一个L_Key和R_Key，并与创建的内存区域相关联 |
| MTU (Maximum Transfer Unit)              | 从端口发送/接收的数据包有效载荷（不包括报头）的最大大小             |
| MW (Memory Window)                       | 在绑定到现有内存注册中的指定区域后，使得远程访问生效的一块被分配的资源。每个内存窗口都有一个关联的窗口句柄，一组访问权限以及当前的R_Key |
| Outstanding Work Request                 | WR(Work Request)被发布到工作队列，并且没有轮询完成        |
| pkey (Partition key)                     | pkey标识端口所属的分区<br>pkey大致类似于以太网网络中的VLAN ID<br>它用于指向端口的分区键（pkey）表中的条目<br>每个端口由子网管理器（SM）分配至少一个pkey |
| PD (Protection Domain)                   | 其成员只能沟通彼此<br>AHs(Address Handle)与QP(Queue Pair)进行交互，MR(Memory Region)与WQ(Work Queue)进行交互 |
| QP (Queue Pair)                          | 在一个对象中打包在一起的独立工作队列（发送队列和接收队列），用于在网络节点之间传输数据。<br>Post用于发送或接收数据的初始化<br>有三种类型的QP：UD(Unreliable Datagram)不可靠的数据报，UC(Unreliable Conection)不可靠的连接和 RC(Reliable Connection)可靠的连接 |
| RC (Reliable Connection)                 | 基于面向连接协议的QP传输服务类型<br>QP（队列对）与另一个单个QP相关联 <br>消息以可靠的方式发送（根据信息的正确性和顺序） |
| RDMA (Remote Direct Memory Access)       | 远程访问内存而不涉及远程CPU                          |
| RDMA_CM (Remote Direct Memory Access Communication Manager) | 用于设置可靠，连接和不可靠的数据报数据传输的API<br>它提供了用于建立连接的RDMA传输中立接口<br>API基于套接字，但适用于基于队列对（QP）的语义：通信必须通过特定的RDMA设备，数据传输是基于消息的 |
| Requestor                                | 将发起数据传输的连接端（通过发出发送请求）                    |
| Responder                                | 该连接端将响应来自请求者的命令，该命令可以包括写入响应者存储器或从响应者存储器读取的请求，最后是请求响应者接收消息的命令 |
| rkey                                     | MR注册后收到的数字，用于强制进入RDMA操作的权限               |
| RNR (Receiver Not Ready)                 | 在可靠的连接队列对中，两边之间有一个连接但是RR(Receive Request)没有出现在接收方 |
| RQ (Receive Queue)                       | 保存用户发出的RR(Receive Request)的工作队列          |
| RR (Receive Request)                     | 一个WR(Work Request)被发布到一个RQ(Receive Queue)，它描述了输入数据使用Send操作码而将要Write的位置<br>另请注意，RDMA的一次立即写入将消耗RR(Reveive Request) |
| RTR (Ready To Receive)                   | 一种QP(Queue Pair)的状态，标志一个RR(Receive Request)可以被发出和处理 |
| RTS (Ready To Send)                      | 一种QP(Queue Pair)的状态，标志一个SR(Send Request)可以被发出和处理 |
| SA (Subnet Administrator)                | 用于查询和操作子网管理数据的界面                         |
| SGE (Scatter /Gather Elements)           | 分散/收集元素<br>指向全局或本地注册内存块的一部分的指针的条目<br>该元素保存块的起始地址，大小和lkey（及其相关权限） |
| S/G Array                                | 存在于WR(Work Request)中的根据所使用的操作码的S/G元素阵列可以从多个缓冲器收集数据，并将其作为单个流发送或单个流分解为多个缓冲区 |
| SM (Subnet Manager)                      | 配置和管理子网的实体<br>发现网络拓扑<br>分配LID<br>确定路由方案并设置路由表<br>一个主机SM和可能的几个从机（待机模式）<br>管理交换机路由表，从而建立路由 |
| SQ (Send Queue)                          | 保存用户发起的SRs(Send Request)的工作队列            |
| SR (Send Request)                        | 一个WR(Work Request)被发布到一个SQ(Send Queue)，描述了要传输多少数据，其方向和方式（操作码将指定转移） |
| SRQ (Shared Receive Queue)               | 存储WQEs(Work Queue Element)的队列，输入信息包括任何与它有关的RC(Reliable Connection)/UC(Unreliable Connection)/UD(Unreliable Datagram)的QP(Queue Pair)<br>可以将多个QP(Queue Pair)与一个SRQ相关联 |
| TCA (Target Channel Adapter)             | 通道适配器，不需要支持verbs，通常用于I/O设备               |
| UC (Unreliable Connection)               | 基于面向连接协议的QP传输服务类型，其中QP（队列对）与另一单个QP相关联<br>QP不执行可靠的协议，消息可能丢失 |
| UD (Unreliable Datagram)                 | 消息可以是一个分组长度并且每个UD QP可以从子网中的另一个UD QP发送/接收消息的QP传输服务类型消息可能会丢失，并且不能保证顺序。 UD QP是唯一支持多播消息的类型。 UD数据包的消息大小限于路径MTU |
| Verbs                                    | 网络适配器功能的抽象描述<br>使用verbs，任何应用程序都可以创建/管理所需的对象，以便使用RDMA进行数据传输 |
| VPI (Virtual Protocol Interface)         | 允许用户更改端口的第2层协议                           |
| WQ (Work Queue)                          | 发送队列或接收队列之一                              |
| WQE (Work Queue Element)                 | WQE发音为“wookie”，是工作队列中的一个元素               |
| WR (Work Request)                        | 用户发布到工作队列的请求。                            |

## 1. RDMA Architecture Overview

### 1.5 Key Components

这些只是在部署IB和RoCE的优势的背景下提出的。我们不讨论电缆和连接器。

#### Host Channel Adapter 主机通道适配器
HCAs提供IB端节点（例如服务器）连接到IB网络的点。  
这些相当于以太网（NIC）卡，但它们做得更多。 HCAs在操作系统的控制下提供地址转换机制，允许应用程序直接访问HCA。相同的地址转换机制是HCA代表用户级应用访问存储器的手段。该应用程序是指虚拟地址，而HCA具有将这些地址转换为物理地址的能力，以影响实际的消息传输。  

#### Range Extenders 范围扩展器 
InfiniBand范围扩展通过将InfiniBand流量封装到WAN链路上并扩展足够的缓冲区来确保WAN上的带宽被全部利用。

#### Subnet Manager 子网管理器
InfiniBand子网管理器为连接到InfiniBand结构的每个端口分配本地标识符（LID），并根据分配的LID开发路由表。 IB子网管理器是软件定义网络（SDN）的概念，它消除了互连复杂性，并创建了非常大规模的计算和存储基础架构。

#### Switches 开关
IB交换机在概念上类似于标准网络交换机，但是被设计为满足IB性能要求。实现IB链路层的流量控制，防止丢包，支持拥塞避免和自适应路由功能，以及高级服务质量。许多交换机包括子网管理器。需要至少一个子网管理器来配置IB结构。

## 2. Available Communication Operations

### 2.1 Available Communication on Operations

#### 2.1.1 Send / Send With Immediate

发送操作允许您将数据发送到远程QP的接收队列。 接收者必须事先已经post接收buffer才能接收数据。 发送方无法控制数据在远程主机中的位置。  

可选地，immediate 4字节值可以与数据缓冲器一起发送。 该值作为接收通知的一部分呈现给接收者，并且不包含在数据缓冲器中。

#### 2.1.2 Receive

这是与发送操作相相应的操作。 接收主机被通知数据缓冲区已经到达，可能具有内联immediate值。 接收应用程序负责接收缓冲区维护和发布。

#### 2.1.3 RDMA Read

从远程主机读取一段内存。 调用者指定要复制到的远程虚拟地址就像指定本地内存地址一样。 在执行RDMA操作之前，远程主机必须提供适当的访问其内存的权限。 一旦设置了这些权限，RDMA读取操作就不会对远程主机发出任何通知。 对于RDMA读写，远程端不知道此操作已完成（除了准备权限和资源）。

#### 2.1.4 RDMA Write / RDMA Write With Immediate

类似于RDMA读取，但数据被写入远程主机。 执行RDMA写入操作，而不通知远程主机。 

RDMA Write with immediate 操作，however, do notify the remote host of the immediate value。

### 2.2 Transport Modes

建立QP时可以选择几种不同的传输模式。 各种模式下的有效操作如下表所示。 此API不支持RD。

| Operation                          | UD   | UC   | RC   | RD   |
| ---------------------------------- | ---- | ---- | ---- | ---- |
| Send(with immediate)               | X    | X    | X    | X    |
| Receive                            | X    | X    | X    | X    |
| RDMA Write(with immediate)         |      | X    | X    | X    |
| RDMA Read                          |      |      | X    | X    |
| Atomic: Fetch and Add/Cmp and Swap |      |      | X    | X    |
| Max Message size                   | MTU  | 1GB  | 1GB  | 1GB  |

#### 2.2.1 RC
- QP is "one to one".
- Message transmitted by the send queue of one QP to receive queue of other QP.
- Transmitte is reliably delivered.
- Packets are delivered in order 有序传输.
- RC is similar to TCP.

#### 2.2.2 UC
- QP is "one to one".
- Connection is not reliable, and packet may be lost.
- 信息传输失败不会重发，异常处理需被上层协议保证。

#### 2.2.3 UD
- QP is "one to many".
- UD is similar to UDP.

### 2.3 Key Concepts

#### 2.3.1 Send Request (SR)
- SR define how much data will be sent, from where, to where, how, with RDMA.
- struct **ibv_send_wr** is used to implement SRs.

#### 2.3.2 Receive Request (RR)
- RR定义了对于non-RDMA操作要接受data的buffers。如果没有buffers被定义，传输人会尝试一个send操作或者一个RDMA Write with immediate，同时receive not ready(RNR)错误将被发出。
- struct **ibv_recv_wr** is used to implement RRs.

#### 2.3.3 Completion Queue (CQ)
- CQ是一个对象，其中包含已发布到工作队列（WQ）的完成的工作请求。每一次完成都表示完成了一个特定的WR（不管成功与否）。
- CQ是一种机制，其会通知应用有关结束了的Work Request的信息（状态，操作码，大小，源）。
- CQ有n个完成队列条目（CQE）。创建CQ时指定CQE的数量。
- 当CQE被轮询时，它将从CQ中删除。
- CQ是CQE的FIFO。
- CQ可以服务发送队列，接收队列或两者。
- 来自多个QP的Work Queue可以与单个CQ相关联。
- struct **ibv_cq** is used to implement a CQ。

#### 2.3.4 Memory Registration (MR)
- MR是一种机制，其允许应用程序将一组虚拟连续的内存位置或一组物理上连续的内存位置描述为网络适配器，该适配器将作为使用虚拟地址的虚拟连续缓冲区。
- 注册过程引导memory pages（以防止pages被换出并同时保留physical<->virtual mapping）。
- 在注册期间，操作系统检查注册块的权限。
- 注册过程将virtual to pyysical地址表写入网络适配器。
- 注册内存时，会为该区域设置权限。权限是本地写入，远程读取，远程写入，原子和绑定。
- 每个MR都有一个远程key和一个本地key（r_key，l_key）。本地keys由本地HCA用于访问本地存储器，例如在接收操作期间。远程键被提供给远程HCA，以允许远程进程在RDMA操作期间访问系统内存。
- 相同的内存缓冲区可以注册几次（即使具有不同的访问权限），并且每次注册都会产生一组不同的keys。
- struct **ibv_mr** is used to implement MR.

#### 2.3.5 Memory Window (MW)
- MW允许应用程序对其内存的远程访问进行更灵活的控制。
- MWs适用于应用程序的情况：
    - 希望以比使用注销/注册或注册更少的性能损失的动态方式授予和撤销对注册地区的远程访问权限。
    - 希望为不同的远程代理授予不同的远程访问权限 和/或 在注册的区域内的不同范围内授予这些权限。
- 将MW与MR相关联的操作称为绑定。
- 不同的MW可以与相同的MR重叠（具有不同访问权限的事件）。

#### 2.3.6 Address Vector
- Address Vector是描述从本地节点到远程节点的路由的对象。
- 在每个UC/RC QP中，QP context中都有一个Address Vector。
- 在UD QP中，应为发送的每个SR定义Address Vector。
- struct **ibv_ah** is used to implement address vectors.

#### 2.3.7 Global Routing Header (GRH)
- GRH用于子网之间的路由。当使用RoCE时，GRH用于在子网内进行路由，因此是强制性的。 使用GRH是强制性的，以便应用程序支持IB和RoCE。
- 当UD QP使用全局路由时，接收缓冲区的前40个字节中将包含GRH。 该区域用于存储全局路由信息，因此可以生成适当的地址向量来响应所接收的分组。 如果GRH与UD一起使用，RR应该总有40个字节用于此GRH。
- struct **ibv_grh** is used to implement GRHs.

#### 2.3.8 Protection Domain (PD)
- PD的组件只能互相交互。 这些组件可以是AH，QP，MR和SRQ。
- PD用于将队列对与MR和MW相关联，作为启用和控制网络适配器访问主机系统内存的方法。
- PD也用于将不可靠的UD QP与地址句柄相AH关联，作为控制对UD目的地的访问的手段。
- struct **ibv_pd** is used to implement protection domains.

#### 2.3.9 Asynchromous Events
- 网络适配器可以发送异步事件以通知SW关于系统中发生的事件。
- 有两种类型的异步事件：
    - 附属事件：个人对象发生的事件（CQ，QP，SRQ）。 这些事件将被发送到一个特定的过程。
    - 无关联的事件：全局对象发生的事件（网络适配器，端口错误）。 这些事件将被发送到所有进程。

#### 2.3.10 Scatter Gather
- 数据正在使用散点收集元素进行收集/散布，其中包括：
    - Address：数据将被收集或散布到的本地数据缓冲区的地址。
    - Size：将读取/写入此地址的数据的大小。
    - L_key：注册到此缓冲区的MR的本地密钥。
- struct **ibv_sge** implements scatter gather elements.

#### 2.3.11 Polling
- 轮询CQ是获取有关发布的WR（发送或接收）的详细信息。
- 如果WR完成状态bad，其余的完成将全部bad掉（并且工作队列将被移动到错误状态）。
- 每个没有完成的WR（被轮询）仍然很出色。
- 只有在WR完成之后，发送/接收缓冲区可能被使用/重新使用/释放。
- 应始终检查完成状态。
- 当CQE被轮询时，它将从CQ中删除。
- Polling is accomplished with the **ibv_poll_cq** operation.

## 3 VPI Verbs API

### 3.1 Initialization

#### 3.1.1 ibv_fork_init

    int ibv_fork_init(void)

    Input Parameters:
    None
    Output Parameters:
    None
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

- ibv_fork_init初始化libibverbs的数据结构以安全地处理fork()函数，并避免数据损坏，不管fork()是否被明确地或隐式地调用，如在system()调用。    
- 如果所有父进程线程总是被阻塞，直到所有子进程通过exec()操作结束或更改地址空间，则不需要调用ibv_fork_init。  
- 该功能适用​​于支持madvise()（2.6.17及更高版本）的MADV_DONTFORK标志的Linux内核。  
- 将环境变量RDMAV_FORK_SAFE或IBV_FORK_SAFE设置为任何值与调用ibv_fork_init()具有相同的效果。  
- 将环境变量RDMAV_HUGEPAGES_SAFE设置为任何值将告知库检查内核对内存区域使用的底层页大小。如果应用程序通过库（例如libhugetlbfs）直接或间接使用庞大的页面，这是必需的。  
- 调用ibv_fork_init()将会降低性能，因为每个内存注册需要额外的系统调用，并且分配给追踪内存区域的额外内存。精确的性能影响取决于工作负载，通常不会很大。    
- 设置RDMAV_HUGEPAGES_SAFE为所有内存启用增加了额外的开销。

### 3.2 Device Operations

以下命令用于常规设备操作，允许用户查询有关系统上的设备以及打开和关闭特定设备的信息。

#### 3.2.1 ibv_get_device_list

    struct ibv_device **ibv_get_device_list (int *num_devices)

    Input Parameters:
    none
    Output Parameters:
    num_devices     (optional) If non-null, the number of devices returned in the array will be stored here
    Return Value:
    NULL terminated array of VPI devices or NULL on failure.

ibv_get_device_list返回系统上可用的VPI设备列表。 列表中的每个条目都是一个指向struct ibv_device的指针。

struct ibv_device结构体定义如下：

    struct ibv_device
    {
        struct ibv_device_ops ops;
        enum ibv_node_type node_type;
        enum ibv_transport_type transport_type;
        char name[IBV_SYSFS_NAME_MAX];
        char dev_name[IBV_SYSFS_NAME_MAX];
        char dev_path[IBV_SYSFS_PATH_MAX];
        char ibdev_path[IBV_SYSFS_PATH_MAX];
    };

| variable       | description                              | example                             |
| -------------- | ---------------------------------------- | ----------------------------------- |
| ops            | pointers to alloc and free functions     |                                     |
| node_type      | IBV_NODE_UNKNOWN<br>IBV_NODE_CA<br>IBV_NODE_SWITCH<br>IBV_NODE_ROUTER<br>IBV_NODE_RNIC | InfiniBand channel adapter          |
| transport_type | IBV_TRANSPORT_UNKNOWN<br>IBV_TRANSPORT_IB<br>IBV_TRANSPORT_IWARP | 0                                   |
| name           | kernel device name eg “mthca0”           | mlx5_0                              |
| dev_name       | uverbs device name eg “uverbs0”          | uverbs0                             |
| dev_path       | path to infiniband_verbs class device in sysfs | /sys/class/infiniband_verbs/uverbs0 |
| ibdev_path     | path to infiniband class device in sysfs | /sys/class/infiniband/mlx5_0        |

ibv_device结构列表将保持有效，直到列表被释放。 调用ibv_get_device_list后，用户应该打开所需的设备，并通过ibv_free_device_list命令及时释放列表。

#### 3.2.2 ibv_free_device_list

    void ibv_free_device_list(struct ibv_device **list)

    Input Parameters:
    list        list of devices provided from ibv_get_device_list command
    Output Parameters:
    none
    Return Value:
    none

ibv_free_device_list释放由ibv_get_device_list提供的ibv_device结构列表。 在调用此命令之前，应打开任何所需的设备。 一旦列表被释放，列表中的所有ibv_device结构都将无效，不能再使用。

#### 3.2.3 ibv_get_device_name

    const char *ibv_get_device_name(struct ibv_device *device)

    Input Parameters:
    device      struct ibv_device for desired device
    Output Parameters:
    none
    Return Value:
    Pointer to device name char string or NULL on failure.

ibv_get_device_name返回了ibv_device结构体中的设备名称的指针。

#### 3.2.4 ibv_get_device_guid

    uint64_t ibv_get_device_guid(struct ibv_device *device)

    Input Parameters:
    device      struct ibv_device for desired device
    Output Parameters:
    none
    Return Value:
    64 bit GUID

ibv_get_device_guid以网络字节顺序返回设备64位全局唯一标识符（GUID）。

#### 3.2.5 ibv_open_device

    struct ibv_context *ibv_open_device(struct ibv_device *device)

    Input Parameters:
    device      struct ibv_device for desired device
    Output Parameters:
    none
    Return Value:
    A verbs context that can be used for future operations on the device or NULL on failure.

ibv_open_device为用户提供了一个verbs context，它将是所有的verbs操作的对象。

#### 3.2.6 ibv_close_device

    int ibv_close_device(struct ibv_context *context)

    Input Parameters:
    context     struct ibv_context from ibv_open_device
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_close_device关闭以前使用ibv_open_device打开的verbs context。 此操作不会释放与context相关联的任何其他对象。 为了避免内存泄漏，所有其他对象必须在调用此命令之前独立释放。

#### 3.2.7 ibv_node_type_str

    const char *ibv_node_type_str (enum ibv_node_type node_type)

    Input Parameters:
    node_type   ibv_node_type enum value which may be an HCA, Switch, Router, RNIC or Unknown
    Output Parameters:
    none
    Return Value:
    A constant string which describes the enum value node_type

ibv_node_type_str返回描述节点类型枚举值node_type的字符串。 该值可以是InfiniBand HCA，交换机，路由器，启用RDMA的NIC或未知。

    enum ibv_node_type {
        IBV_NODE_UNKNOWN = -1,
        IBV_NODE_CA = 1,
        IBV_NODE_SWITCH,
        IBV_NODE_ROUTER,
        IBV_NODE_RNIC
    };

#### 3.2.8 ibv_port_state_str

    const char *ibv_port_state_str (enum ibv_port_state port_state)

    Input Parameters:
    port_state The enumerated value of the port state Output Parameters:
    None
    Return Value:
    A constant string which describes the enum value port_state

ibv_port_state_str返回一个描述端口状态枚举值port_state的字符串。

    enum ibv_port_state {
        IBV_PORT_NOP = 0,
        IBV_PORT_DOWN = 1,
        IBV_PORT_INIT = 2,
        IBV_PORT_ARMED = 3,
        IBV_PORT_ACTIVE = 4,
        IBV_PORT_ACTIVE_DEFER = 5
    };

### 3.3 Verb Context Operations

一旦打开设备，就会使用以下命令。 这些命令允许您获取有关设备或其端口之一的更具体信息，创建可用于进一步操作的完成队列（CQ），完成通道（CC）和保护域（PD）。

#### 3.3.1 ibv_query_device

    int ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr)

    Input Parameters:
    context         struct ibv_context from ibv_open_device 
    Output Parameters:
    device_attr     struct ibv_device_attr containing device attributes
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_device检索与设备关联的各种属性。 用户负责对struct ibv_device_attr进行空间分配和释放。其值将会被填写，如果函数成功返回。

结构体定义如下：

    struct ibv_device_attr
    {
        char        fw_ver[64];
        uint64_t    node_guid;
        uint64_t    sys_image_guid;
        uint64_t    max_mr_size;
        uint64_t    page_size_cap;
        uint32_t    vendor_id;
        uint32_t    vendor_part_id;
        uint32_t    hw_ver;
        int         max_qp;
        int         max_qp_wr;
        int         device_cap_flags;
        int         max_sge;
        int         max_sge_rd;
        int         max_cq;
        int         max_cqe;
        int         max_mr;
        int         max_pd;
        int         max_qp_rd_atom;
        int         max_ee_rd_atom;
        int         max_res_rd_atom;
        int         max_qp_init_rd_atom;
        int         max_ee_init_rd_atom;
        enum ibv_atomic_cap atomic_cap;
        int         max_ee;
        int         max_rdd;
        int         max_mw;
        int         max_raw_ipv6_qp;
        int         max_raw_ethy_qp;
        int         max_mcast_grp;
        int         max_mcast_qp_attach;
        int         max_total_mcast_qp_attach;
        int         max_ah;
        int         max_fmr;
        int         max_map_per_fmr;
        int         max_srq;
        int         max_srq_wr;
        int         max_srq_sge;
        uint16_t    max_pkeys;
        uint8_t     local_ca_ack_delay;
        uint8_t     phys_port_cnt;
    }

| variable                  | description                              | example              |
| ------------------------- | ---------------------------------------- | -------------------- |
| fw_ver                    | Firmware version                         | 12.18.1000           |
| node_guid                 | Node global unique identifier (GUID)     | 9814281471509629476  |
| sys_image_guid            | System image GUID                        | 9814281471509629476  |
| max_mr_size               | Largest contiguous block that can be registered | 18446744073709551615 |
| page_size_cap             | Supported page sizes                     | 18446744073709547520 |
| vendor_id                 | Vendor ID, per IEEE                      | 713                  |
| vendor_part_id            | Vendor supplied part ID                  | 4115                 |
| hw_ver                    | Hardware version                         | 0                    |
| max_qp                    | Maximum number of Queue Pairs (QP)       | 131072               |
| max_qp_wr                 | Maximum outstanding work requests (WR) on any queue | 32768                |
| device_cap_flags          | IBV_DEVICE_RESIZE_MAX_WR<br>IBV_DEVICE_BAD_PKEY_CNTR<br>IBV_DEVICE_BAD_QKEY_CNTR<br>IBV_DEVICE_RAW_MULTI<br>IBV_DEVICE_AUTO_PATH_MIG<br>IBV_DEVICE_CHANGE_PHY_PORT<br>IBV_DEVICE_UD_AV_PORT_ENFORCE<br>IBV_DEVICE_CURR_QP_STATE_MOD<br>IBV_DEVICE_SHUTDOWN_PORT<br>IBV_DEVICE_INIT_TYPE<br>IBV_DEVICE_PORT_ACTIVE_EVENT<br>IBV_DEVICE_SYS_IMAGE_GUID<br>IBV_DEVICE_RC_RNR_NAK_GEN<br>IBV_DEVICE_SRQ_RESIZE<br>IBV_DEVICE_N_NOTIFY_CQ<br>IBV_DEVICE_XRC | -914482122           |
| max_sge                   | Maximum scatter/gather entries (SGE) per WR for non-RD QPs | 30                   |
| max_sge_rd                | Maximum SGEs per WR for RD QPs           | 30                   |
| max_cq                    | Maximum supported completion queues (CQ) | 16777216             |
| max_cqe                   | Maximum completion queue entries (CQE) per CQ | 4194303              |
| max_mr                    | Maximum supported memory regions (MR)    | 16777216             |
| max_pd                    | Maximum supported protection domains (PD) | 16777216             |
| max_qp_rd_atom            | Maximum outstanding RDMA read and atomic operations per QP | 16                   |
| max_ee_rd_atom            | Maximum outstanding RDMA read and atomic operations per End to End (EE) context (RD connections) | 0                    |
| max_res_rd_atom           | Maximum resources used for incoming RDMA read and atomic operations | 2097152              |
| max_qp_init_rd_atom       | Maximium RDMA read and atomic operations that may be initiated per QP | 16                   |
| max_ee_init_atom          | Maximum RDMA read and atomic operations that may be initiated per EE | 0                    |
| atomic_cap                | IBV_ATOMIC_NONE - no atomic guarantees<br>IBV_ATOMIC_HCA - atomic guarantees within this device<br>IBV_ATOMIC_GLOB - global atomic guarantees | 1                    |
| max_ee                    | Maximum supported EE contexts            | 0                    |
| max_rdd                   | Maximum supported RD domains             | 0                    |
| max_mw                    | Maximum supported memory windows (MW)    | 16777216             |
| max_raw_ipv6_qp           | Maximum supported raw IPv6 datagram QPs  | 0                    |
| max_raw_ethy_qp           | Maximum supported ethertype datagram QPs | 0                    |
| max_mcast_grp             | Maximum supported multicast groups       | 2097152              |
| max_mcast_qp_attach       | Maximum QPs per multicast group that can be attached | 48                   |
| max_total_mcast_qp_attach | Maximum total QPs that can be attached to multicast groups | 100663296            |
| max_ah                    | Maximum supported address handles (AH)   | 2147483647           |
| max_fmr                   | Maximum supported fast memory regions (FMR) | 0                    |
| max_map_per_fmr           | Maximum number of remaps per FMR before an unmap operation is required | 2147483647           |
| max_srq                   | Maximum supported shared receive queues (SRCQ) | 8388608              |
| max_srq_wr                | Maximum work requests (WR) per SRQ       | 32767                |
| max_srq_sge               | Maximum SGEs per SRQ                     | 31                   |
| max_pkeys                 | Maximum number of partitions             | 128                  |
| local_ca_ack_delay        | Local CA ack delay                       | 16                   |
| phys_port_cnt             | Number of physical ports                 | 1                    |

#### 3.3.2 ibv_query_port

    int ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr)

    Input Parameters:
    context         struct ibv_context from ibv_open_device
    port_num        physical port number (1 is first port)
    Output Parameters:
    port_attr       struct ibv_port_attr containing port attributes
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_port检索与端口关联的各种属性。 用户应该分配一个结构体ibv_port_attr，将其传递给函数，并在成功返回时填写它。 用户有责任释放这个结构。

结构体定义如下：

    struct ibv_port_attr
    {
        enum ibv_port_state     state;
        enum ibv_mtu            max_mtu;
        enum ibv_mtu            active_mtu;
        int                     gid_tbl_len;
        uint32_t                port_cap_flags;
        uint32_t                max_msg_sz;
        uint32_t                bad_pkey_cntr;
        uint32_t                qkey_viol_cntr;
        uint16_t                pkey_tbl_len;
        uint16_t                lid;
        uint16_t                sm_lid;
        uint8_t                 lmc;
        uint8_t                 max_vl_num;
        uint8_t                 sm_sl;
        uint8_t                 subnet_timeout;
        uint8_t                 init_type_reply;
        uint8_t                 active_width;
        uint8_t                 active_speed;
        uint8_t                 phys_state;
    };

| variable        | description                              |
| --------------- | ---------------------------------------- |
| state           | IBV_PORT_NOP<br>IBV_PORT_DOWN<br>IBV_PORT_INIT<br>IBV_PORT_ARMED<br>IBV_PORT_ACTIVE<br>IBV_PORT_ACTIVE_DEFER |
| max_mtu         | Maximum Transmission Unit (MTU) supported by port. Can be:<br>IBV_MTU_256<br>IBV_MTU_512<br>IBV_MTU_1024<br>IBV_MTU_2048<br>IBV_MTU_4096 |
| active_mtu      | Actual MTU in use                        |
| gid_tbl_len     | Length of source global ID (GID) table   |
| port_cap_flags  | Supported capabilities of this port. There are currently no enumerations/defines declared in verbs.h |
| max_msg_sz      | Maximum message size                     |
| bad_pkey_cntr   | Bad P_Key counter                        |
| qkey_viol_cntr  | Q_Key violation counter                  |
| pkey_tbl_len    | Length of partition table                |
| lid First local | identifier (LID) assigned to this port   |
| sm_lid          | LID of subnet manager (SM)               |
| lmc LID         | Mask control (used when multiple LIDs are assigned to port) |
| max_vl_num      | Maximum virtual lanes (VL)               |
| sm_sl           | SM service level (SL)                    |
| subnet_timeout  | Subnet propagation delay                 |
| init_type_reply | Type of initialization performed by SM   |
| active_width    | Currently active link width              |
| active_speed    | Currently active link speed              |
| phys_state      | Physical port state                      |

#### 3.3.3 ibv_query_gid

    int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid)

    Input Parameters:
    context         struct ibv_context from ibv_open_device
    port_num        physical port number (1 is first port)
    index           which entry in the GID table to return (0 is first)
    Output Parameters:
    gid             union ibv_gid containing gid information
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_gid检索端口全局标识符（GID）表中的条目。 每个端口由子网管理器（SM）分配至少一个GID。 GID是由全局唯一标识符（GUID）和由SM分配的前缀组成的有效IPv6地址。 GID [0]是唯一的，包含端口的GUID。  
用户应该分配一个联合ibv_gid，将其传递给命令，并在成功返回时填写它。 用户有责任释放这个联合。  
union ibv_gid定义如下：  

    union ibv_gid
    {
        uint8_t         raw[16];
        struct
        {
            uint64_t    subnet_prefix;
            uint64_t    interface_id;
        } global;
    }

#### 3.3.4 ibv_query_pkey

    int ibv_query_pkey(struct ibv_context *context, uint8_t port_num, int index, uint16_t *pkey)

    Input Parameters:
    context     struct ibv_context from ibv_open_device
    port_num    physical port number (1 is first port)
    index       which entry in the pkey table to return (0 is first)
    Output Parameters:
    pkey        desired pkey
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_query_pkey检索端口分区密钥（pkey）表中的条目。 每个端口由子网管理器（SM）分配至少一个pkey。 pkey标识端口所属的分区。 pkey类似于以太网中的VLAN ID。  
用户将一个指针传递给一个将使用请求的pkey填充的uint16。 用户有责任释放这个uint16。

#### 3.3.5 ibv_alloc_pd

    struct ibv_pd *ibv_alloc_pd(struct ibv_context *context)

    Input Parameters:
    context struct ibv_context from ibv_open_device
    Output Parameters:
    none
    Return Value:
    Pointer to created protection domain or NULL on failure.

ibv_alloc_pd创建一个保护域（PD）。 PD限制可以通过哪些队列对（QP）提供一定程度的保护以防止未经授权的访问来访问哪些存储器区域。  
用户必须创建至少一个PD才能使用VPI verbs。

#### 3.3.6 ibv_dealloc_pd

    int ibv_dealloc_pd(struct ibv_pd *pd)

    Input Parameters:
    pd      struct ibv_pd from ibv_alloc_pd
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_dealloc_pd释放保护域（PD）。 如果任何其他对象当前与指定的PD相关联，则此命令将失败。

#### 3.3.7 ibv_create_cq

    struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector)

    Input Parameters:
    context     struct ibv_context from ibv_open_device
    cqe         Minimum number of entries CQ will support
    cq_context  (Optional) User defined value returned with completion events
    channel     (Optional) Completion channel
    comp_vector (Optional) Completion vector
    Output Parameters:
    none
    Return Value:
    pointer to created CQ or NULL on failure.

ibv_create_cq创建一个完成队列（CQ）。完成队列保存完成队列条目（CQE）。每个队列对（QP）具有相关联的发送和接收CQ。单个CQ可以共享用于发送和接收，并且可以跨多个QP共享。  
参数cqe定义队列的最小大小。队列的实际大小可能大于指定的值。  
参数cq_context是用户定义的值。如果在创建CQ时指定，则在使用完成通道（CC）时，该值将作为ibv_get_cq_event中的参数返回。  
参数通道用于指定CC。 CQ只是一个没有内置通知机制的队列。当使用轮询范例进行CQ处理时，不需要CC。用户只需定期轮询CQ。但是，如果您希望使用挂钩范例，则需要CC。 CC是允许用户通知新CQE在CQ上的机制。  
参数comp_vector用于指定用于表示完成事件的完成向量。它必须是 >= 0和 < context->num_comp_vectors。  

#### 3.3.8 ibv_resize_cq

    int ibv_resize_cq(struct ibv_cq *cq, int cqe)

    Input Parameters:
    cq      CQ to resize
    cqe     Minimum number of entries CQ will support
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_resize_cq调整完成队列（CQ）的大小。  
参数cqe必须至少为队列中未决条目的数量。 队列的实际大小可能大于指定的值。 当CQ调整大小时，CQ可以（或可能不）包含完成，可以在CQ工作期间调整大小。

#### 3.3.9 ibv_destroy_cq

    int ibv_destroy_cq(struct ibv_cq *cq)

    Input Parameters:
    cq      CQ to destroy
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_destroy_cq释放一个完成队列（CQ）。 如果存在与其相关联的指定CQ的队列对（QP），则此命令将失败。

#### 3.3.10 ibv_create_comp_channel

    struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context)

    Input Parameters:
    context         struct ibv_context from ibv_open_device
    Output Parameters:
    none
    Return Value:
    pointer to created CC or NULL on failure.

ibv_create_comp_channel创建一个完成通道。 完成通道是当新的完成队列事件（CQE）已经被放置在完成队列（CQ）上时用户接收通知的机制。

#### 3.3.11 ibv_destroy_comp_channel

    int ibv_destroy_comp_channel(struct ibv_comp_channel *channel)

    Input Parameters:
    channel         struct ibv_comp_channel from ibv_create_comp_channel
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_destroy_comp_channel释放完成通道。 如果任何完成队列（CQ）仍然与此完成通道相关联，则此命令将失败。

### 3.4 Protection Domain Operations

建立保护域（PD）后，您可以在该域内创建对象。 本节介绍PD上可用的操作。 这些包括注册存储器区域（MR），创建队列对（QP）或共享接收队列（SRQ）和地址句柄（AH）。

#### 3.4.1 ibv_reg_mr

    struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, enum ibv_access_flags access)    

    Input Parameters:
    pd          protection domain, struct ibv_pd from ibv_alloc_pd
    addr        memory base address
    length      length of memory region in bytes
    access      access flags
    Output Parameters:
    none
    Return Value:
    pointer to created memory region (MR) or NULL on failure.    

ibv_reg_mr注册一个内存区域（MR），将其与保护域（PD）相关联，并为其分配本地和远程密钥（lkey，rkey）。 用到内存的所有VPI命令都需要通过该命令注册内存。 相同的物理内存可以被映射到不同的MR，甚至根据用户需求，允许不同的权限或PD被分配给相同的存储器。    
访问标志可能是按位或以下枚举之一：  

| Enum                     | Descript     |
| ------------------------ | ------------ |
| IBV_ACCESS_LOCAL_WRITE   | 允许本地主机写访问    |
| IBV_ACCESS_REMOTE_WRITE  | 允许远程主机写入访问   |
| IBV_ACCESS_REMOTE_READ   | 允许远程主机读取访问权限 |
| IBV_ACCESS_REMOTE_ATOMIC | 允许远程主机进行原子访问 |
| IBV_ACCESS_MW_BIND       | 允许此MR上的内存窗口  |

本地读取访问是隐含和自动的。    
任何违反给定内存操作访问权限的VPI操作都将失败。 请注意，队列对（QP）属性还必须具有正确的权限，否则操作将失败。  
如果设置了IBV_ACCESS_REMOTE_WRITE或IBV_ACCESS_REMOTE_ATOMIC，则也必须设置IBV_ACCESS_LOCAL_WRITE。  

结构体ibv_mr定义如下：

    struct ibv_mr
    {
        struct ibv_context      *context;
        struct ibv_pd           *pd;
        void                    *addr;
        size_t                  length;
        uint32_t                handle;
        uint32_t                lkey;
        uint32_t                rkey;
    };

#### 3.4.2 ibv_dereg_mr

    int ibv_dereg_mr(struct ibv_mr *mr)

    Input Parameters:
    mr              struct ibv_mr from ibv_reg_mr
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_dereg_mr释放内存区域（MR）。 如果任何存储窗口（MW）仍然绑定到MR，操作将失败。

#### 3.4.3 ibv_create_qp

    struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr)

    Input Parameters:
    pd                  struct ibv_pd from ibv_alloc_pd
    qp_init_attr        initial attributes of queue pair
    Output Parameters:
    qp_init_attr        actual values are filled in
    Return Value:
    pointer to created queue pair (QP) or NULL on failure.

ibv_create_qp创建QP。 当创建QP时，将其置于RESET状态。  
struct qp_init_attr定义如下：

    struct ibv_qp_init_attr
    {
        void                *qp_context;
        struct ibv_cq       *send_cq;
        struct ibv_cq       *recv_cq;
        struct ibv_srq      *srq;
        struct ibv_qp_cap   cap;
        enum ibv_qp_type    qp_type;
        int                 sq_sig_all;
        struct ibv_xrc_domain *xrc_domain;
    };

| variable   | description                              | example        |
| ---------- | ---------------------------------------- | -------------- |
| qp_context | (optional) user defined value associated with QP | 未指定            |
| send_cq    | send CQ. This must be created by the user prior to calling ibv_create_qp | adapter.cq_    |
| recv_cq    | receive CQ. This must be created by the user prior to calling ibv_create_qp. It may be the same as send_cq. | adapter.cq_    |
| srq        | (optional) shared receive queue. Only used for SRQ QP’s. | 未指定            |
| cap        | defined below.                           | defined below. |
| qp_type    | must be one of the following:<br>IBV_QPT_RC = 2IBV_QPT_UC<br>IBV_QPT_UD<br>IBV_QPT_XRC<br>IBV_QPT_RAW_PACKET = 8<br>IBV_QPT_RAW_ETH = 8 | IBV_QPT_RC     |
| sq_sig_all | If this value is set to 1, all send requests (WR) will generate completion queue events (CQE). If this value is set to 0, only WRs that are flagged will generate CQE’s (see ibv_post_send). | 未指定            |
| xrc_domain | (Optional) Only used for XRC operations. | 未指定            |

    struct ibv_qp_cap
    {
        uint32_t max_send_wr;
        uint32_t max_recv_wr;
        uint32_t max_send_sge;
        uint32_t max_recv_sge;
        uint32_t max_inline_data;
    };

| variable        | description                              | example                     |
| --------------- | ---------------------------------------- | --------------------------- |
| max_send_wr     | Maximum number of outstanding send requests in the send queue | MAX_CONCURRENT_WRITES(1000) |
| max_recv_wr     | Maximum number of outstanding receive requests (buffers) in the receive queue. | MAX_CONCURRENT_WRITES(1000) |
| max_send_sge    | Maximum number of scatter/gather elements (SGE) in a WR on the send queue. | 1                           |
| max_recv_sge    | Maximum number of SGEs in a WR on the receive queue. | 1                           |
| max_inline_data | Maximum size in bytes of inline data on the send queue. | 未指定                         |

#### 3.4.4 ibv_destroy_qp

    int ibv_destroy_qp(struct ibv_qp *qp)

    Input Parameters:
    qp              struct ibv_qp from ibv_create_qp
    Output Parameters:
    none
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_destroy_qp释放队列对（QP）。

#### 3.4.5 ibv_create_srq

    struct ibv_srq *ibv_create_srq(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr)

    Input Parameters:
    pd              The protection domain associated with the shared receive queue (SRQ)
    srq_init_attr   A list of initial attributes required to create the SRQ
    Output Parameters:
    ibv_srq__attr   Actual values of the struct are set
    Return Value:
    A pointer to the created SRQ or NULL on failure

ibv_create_srq创建一个共享接收队列（SRQ）。 读取srq_attr-> max_wr和srq_attr-> max_sge以确定所需的SRQ大小，并将其设置为返回时分配的实际值。 如果ibv_create_srq成功，那么max_wr和max_sge将至少与请求的值一样大。  
struct ibv_srq定义如下：

    struct ibv_srq {
        struct ibv_context      *context;       struct ibv_context from ibv_open_device
        void                    *srq_context;
        struct ibv_pd           *pd;            Protection domain
        uint32_t                handle;
        pthread_mutex_t         mutex;
        pthread_cond_t          cond;
        uint32_t                events_completed;
    }

struct ibv_srq_init_attr定义如下：

    struct ibv_srq_init_attr
    {
        void *srq_context;
        struct ibv_srq_attr attr;
    };

--- | ---
srq_context  | struct ibv_context from ibv_open_device
attr         | An ibv_srq_attr struct defined as follows

struct ibv_srq_attr定义如下：

    struct ibv_srq_attr
    {
        uint32_t    max_wr;
        uint32_t    max_sge;
        uint32_t    srq_limit;
    };

--- | ---
max_wr      | Requested maximum number of outstanding WRs in the SRQ
max_sge     | Requested number of scatter elements per WR
srq_limit   | The limit value of the SRQ (irrelevant for ibv_create_srq)

#### 3.4.6 ibv_modify_srq

    int ibv_modify_srq (struct ibv_srq *srq, struct ibv_srq_attr *srq_attr, int srq_attr_mask)

    Input Parameters:
    srq             The SRQ to modify
    srq_attr        Specifies the SRQ to modify (input)/the current values of the selected SRQ attributes are returned (output)
    srq_attr_mask   A bit-mask used to specify which SRQ attributes are being modified
    Output Parameters:
    srq_attr        The struct ibv_srq_attr is returned with the updated values
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.

ibv_modify_srq使用srq_attr中的属性值根据掩码srq_attr_mask修改SRQ srq的属性。 srq_attr是ibv_create_srq下面定义的ibv_srq_attr结构。 参数srq_attr_mask指定要修改的SRQ属性。 它是一个或多个标志的0或按位OR：

--- | ---
IBV_SRQ_MAX_WR  | Resize the SRQ
IBV_SRQ_LIMIT   | Set the SRQ limit

如果要修改的任何属性无效，则不会修改任何属性。 此外，并非所有设备都支持调整SRQ的大小。 要检查设备是否支持调整大小，请检查设备能力标志中是否设置了IBV_DEVICE_SRQ_RESIZE位。  
一旦SRQ中的WR数降到SRQ限制以下，修改SRQ限制就会使SRQ产生一个IBV_EVENT_SRQ_LIMIT_REACHED'低水印'异步事件。

#### 3.4.7 ibv_destroy_srq

#### 3.4.8 ibv_open_xrc_domain

#### 3.4.9 ibv_create_rc_srq

#### 3.4.10 ibv_close_xrc_domain

#### 3.4.11 ibv_create_xrc_rcv_qp

#### 3.4.12 ibv_modify_xrc_rcv_qp

#### 3.4.13 ibv_reg_xrc_rcv_qp

#### 3.4.14 ibv_unreg_xrc_rcv_qp

#### 3.4.15 ibv_create_ah

#### 3.4.16 ibv_destroy_ah

### 3.5 Queue Pair Bringup (ibv_modify_qp)

#### 3.5.1 ibv_modify_qp

    int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, enum ibv_qp_attr_mask attr_mask)

    struct ibv_qp_attr
    {
        enum ibv_qp_state qp_state;
        enum ibv_qp_state cur_qp_state;
        enum ibv_mtu path_mtu;
        enum ibv_mig_state path_mig_state;
        uint32_t qkey;
        uint32_t rq_psn;
        uint32_t sq_psn;
        uint32_t dest_qp_num;
        int qp_access_flags;
        struct ibv_qp_cap cap;
        struct ibv_ah_attr ah_atpsntr;
        struct ibv_ah_attr alt_ah_attr;
        uint16_t pkey_index;
        uint16_t alt_pkey_index;
        uint8_t en_sqd_async_notify;
        uint8_t sq_draining;
        uint8_t max_rd_atomic;
        uint8_t max_dest_rd_atomic;
        uint8_t min_rnr_timer;
        uint8_t port_num;
        uint8_t timeout;
        uint8_t retry_cnt;
        uint8_t rnr_retry;
        uint8_t alt_port_num;
        uint8_t alt_timeout;
    };
    
    IBV_QP_STATE
    IBV_QP_CUR_STATE
    IBV_QP_EN_SQD_ASYNC_NOTIFY
    IBV_QP_ACCESS_FLAGS
    IBV_QP_PKEY_INDEX
    IBV_QP_PORT
    IBV_QP_QKEY
    IBV_QP_AV
    IBV_QP_PATH_MTU
    IBV_QP_TIMEOUT
    IBV_QP_RETRY_CNT
    IBV_QP_RNR_RETRY
    IBV_QP_RQ_PSN
    IBV_QP_MAX_QP_RD_ATOMIC
    IBV_QP_ALT_PATH
    IBV_QP_MIN_RNR_TIMER
    IBV_QP_SQ_PSN
    IBV_QP_MAX_DEST_RD_ATOMIC
    IBV_QP_PATH_MIG_STATE
    IBV_QP_CAP
    IBV_QP_DEST_QPN

#### 3.5.2 RESET to INIT

#### 3.5.3 INIT to RTR

#### 3.5.4 RTR to RTS

### 3.6 Active Queue Pair Operations

#### 3.6.1 ibv_query_qp

#### 3.6.2 ibv_query_srq

#### 3.6.3 ibv_query_xrc_rcv_qp

#### 3.6.4 ibv_post_recv

    int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr)

    Input Parameters:

    qp struct ibv_qp from ibv_create_qp
    wr first work request (WR) containing receive buffers
    
    Output Parameters:
    bad_wr pointer to first rejected WR
    
    Return Value:
    0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.
    
    struct ibv_recv_wr
    {
        uint64_t wr_id;             //user assigned work request ID
        struct ibv_recv_wr *next;   //pointer to next WR, NULL if last one.
        struct ibv_sge *sg_list;    //scatter array for this WR
        int num_sge;                //number of entries in sg_list
    };
    
    struct ibv_sge
    {
        uint64_t addr;              //address of buffer
        uint32_t length;            //length of buffer
        uint32_t lkey;              //local key (lkey) of buffer from ibv_reg_mr
    };

ibv_post_recv将一系列WRs发布到QP的接收队列。应将至少一个接收缓冲区发布到接收队列，以将QP转换为RTR。 随着远程执行发送、立即发送和使用即时操作的RDMA写入、接收缓冲区被消耗。 接收缓冲区不用于其他RDMA操作。 WR列表的处理在第一个错误上停止，并且在bad_wr中返回指向违规WR的指针。

#### 3.6.5 ibv_post_send

```c++
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr)

Input Parameters:
qp      struct ibv_qp from ibv_create_qp
Output Parameters:
bad_wr  pointer to first rejected WR
Return Value:
0 on success, -1 on error. If the call fails, errno will be set to indicate the reason for the failure.
```

ibv_post_send将一个WR list链接到队列对（QP）发送队列。 此操作用于启动所有通信，包括RDMA操作。 WR列表的处理在第一个错误上停止，并且在bad_wr中返回指向违规WR的指针。
用户不应该更改或销毁与WR相关联的AH，直到请求完全执行，并从对应的完成队列（CQ）中检索到完成队列条目（CQE）以避免意外的行为。
WR完全执行后才能安全地重复使用WR缓冲区，并从对应的CQ中检索出WCE。 但是，如果设置了IBV_SEND_INLINE标志，缓冲区可以在调用返回后立即重新使用。
struct ibv_send_wr定义如下：

```c++
struct ibv_send_wr
{
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
    enum ibv_wr_opcode opcode;
    enum ibv_send_flags send_flags;
    uint32_t imm_data;/* network byte order */
    union
    {
        struct
        {
            uint64_t remote_addr;
            uint32_t rkey;
        } rdma;
        struct
        {
            uint64_t remote_addr;
            uint64_t compare_add;
            uint64_t swap;
            uint32_t rkey;
        } atomic;
        struct
        {
            struct ibv_ah *ah;
            uint32_t remote_qpn;
            uint32_t remote_qkey;
        } ud;
    } wr;
    uint32_t xrc_remote_srq_num;
};
```

| variable           | description                              |
| ------------------ | ---------------------------------------- |
| wr_id              | user assigned work request ID            |
| next               | pointer to next WR, NULL if last one.    |
| sg_list            | scatter/gather array for this WR         |
| num_sge            | number of entries in sg_list             |
| opcode             | IBV_WR_RDMA_WRITE<br>IBV_WR_RDMA_WRITE_WITH_IMM<br>IBV_WR_SEND<br>IBV_WR_SEND_WITH_IMM<br>IBV_WR_RDMA_READ<br>IBV_WR_ATOMIC_CMP_AND_SWP<br>IBV_WR_ATOMIC_FETCH_AND_ADD |
| send_flags         | (optional) - this is a bitwise OR of the flags. See the details below. |
| imm_data           | immediate data to send in network byte order |
| remote_addr        | remote virtual address for RDMA/atomic operations |
| rkey               | remote key (from ibv_reg_mr on remote) for RDMA/atomic operations |
| compare_add        | compare value for compare and swap operation |
| swap               | swap value                               |
| ah                 | address handle (AH) for datagram operations |
| remote_qpn         | remote QP number for datagram operations |
| remote_qkey        | Qkey for datagram operations             |
| xrc_remote_srq_num | shared receive queue (SRQ) number for the destination extended reliable connection (XRC). Only used for XRC operations. |



| send flags              | description                              |
| ----------------------- | ---------------------------------------- |
| IBV_SEND_FENCE          | set fence indicator                      |
| IBV_SEND_SIGNALED       | send completion event for this WR. Only meaningful for QPs that had the sq_sig_all set to 0<br>发送此WR的完成事件。 只对sq_sig_all设置为0的QP有意义 |
| IBV_SEND_SEND_SOLICITED | set solicited event indicator            |
| IBV_SEND_INLINE         | send data in sge_list as inline data.    |

#### 3.6.6 ibv_post_srq_recv

#### 3.6.7 ibv_req_notify_cq

    int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only)

#### 3.6.8 ibv_get_cq_event

#### 3.6.9 ibv_ack_cq_events

```c++
void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents)
Input Parameters:
cq          struct ibv_cq from ibv_create_cq
nevents     number of events to acknowledge (1...n)
Output Parameters:
None
Return Value:
None
```

ibv_ack_cq_events确认从ibv_get_cq_event接收到的事件。 虽然从ibv_get_cq_event接收到的每个通知只算一个事件，但用户可以通过单次调用ibv_ack_cq_events来确认多个事件。 要确认的事件数量在nevents中传递，应至少为1个。由于此操作需要互斥体，因此在某个调用中确认多个事件可能会提供更好的性能。
有关其他详细信息，请参阅ibv_get_cq_event。

#### 3.6.10 ibv_poll_cq

```c++
int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc)
Input Parameters:
cq              struct ibv_cq from ibv_create_cq
num_entries     maximum number of completion queue entries (CQE) to return
Output Parameters:
wc              CQE array
Return Value:
Number of CQEs in array wc or -1 on error
```

ibv_poll_cq从完成队列（CQ）中检索CQE。 用户应该分配一个struct ibv_wc的数组，并将其传递给wc中的调用。 wc中可用的条目数应该以num_entries的形式传递。 释放这个内存是用户的责任。
实际检索的CQE数量作为返回值。
定期检查CQ以防止超限。 在超载的情况下，CQ将被关闭，并且将发送异步事件IBV_EVENT_CQ_ERR。
struct ibv_wc定义如下：

```c++
struct ibv_wc
{
    uint64_t wr_id;
    enum ibv_wc_status status;
    enum ibv_wc_opcode opcode;
    uint32_t vendor_err;
    uint32_t byte_len;
    uint32_t imm_data;/* network byte order */
    uint32_t qp_num;
    uint32_t src_qp;
    enum ibv_wc_flags wc_flags;
    uint16_t pkey_index;
    uint16_t slid;
    uint8_t sl;
    uint8_t dlid_path_bits;
};
```

| variable       | 描述                                       |
| -------------- | ---------------------------------------- |
| wr_id          | user specified work request id as given in ibv_post_send or ibv_post_recv |
| status         | IBV_WC_SUCCESS<br>IBV_WC_LOC_LEN_ERR<br>IBV_WC_LOC_QP_OP_ERR<br>IBV_WC_LOC_EEC_OP_ERR<br>IBV_WC_LOC_PROT_ERR<br>IBV_WC_WR_FLUSH_ERR<br>IBV_WC_MW_BIND_ERR<br>IBV_WC_BAD_RESP_ERR<br>IBV_WC_LOC_ACCESS_ERR<br>IBV_WC_REM_INV_REQ_ERR<br>IBV_WC_REM_ACCESS_ERR<br>IBV_WC_REM_OP_ERR<br>IBV_WC_RETRY_EXC_ERR<br>IBV_WC_RNR_RETRY_EXC_ERR<br>IBV_WC_LOC_RDD_VIOL_ERR<br>IBV_WC_REM_INV_RD_REQ_ERR<br>IBV_WC_REM_ABORT_ERR<br>IBV_WC_INV_EECN_ERR<br>IBV_WC_INV_EEC_STATE_ERR<br>IBV_WC_FATAL_ERR<br>IBV_WC_RESP_TIMEOUT_ERR<br>IBV_WC_GENERAL_ERR |
| opcode         | IBV_WC_SEND,<br>IBV_WC_RDMA_WRITE,<br>IBV_WC_RDMA_READ,<br>IBV_WC_COMP_SWAP,<br>IBV_WC_FETCH_ADD,<br>IBV_WC_BIND_MW,<br>IBV_WC_RECV= 1 << 7,<br>IBV_WC_RECV_RDMA_WITH_IMM |
| vendor_err     | vendor specific error                    |
| byte_len       | number of bytes transferred              |
| imm_data       | immediate data                           |
| qp_num         | local queue pair (QP) number             |
| src_qp         | remote QP number                         |
| wc_flags       | IBV_WC_GRH global route header (GRH) is present in UD packet<br>IBV_WC_WITH_IMM immediate data value is valid |
| pkey_index     | index of pkey (valid only for GSI QPs)   |
| slid           | source local identifier (LID)            |
| sl             | service level (SL)                       |
| dlid_path_bits | destination LID path bits                |

#### 3.6.11 ibv_init_ah_from_wc

#### 3.6.12 ibv_create_ah_from_wc

#### 3.6.13 ibv_attach_mcast

#### 3.6.14 ibv_detach_mcast

### 3.7 Event Handing Operations

#### 3.7.1 ibv_get_async_event

#### 3.7.2 ibv_ack_async_enentibv_reg_mr

#### 3.7.3 ibv_event_type_str

### 3.8 Experimental APIs

#### 3.8.1 ibv_exp_query_device

#### 3.8.2 ibv_exp_create_qp

#### 3.8.3 ibv_exp_post_send

## 7 IBV Verbs编程示例

### 7.1 使用IBV Verbs的RDMA_RC示例概要

以下是编程示例中的函数的概要，按照它们的调用顺序。

代码如 RDMA_RC_example.c

解析命令行。 用户可以设置测试的TCP端口，设备名称和设备端口。 如果设置，这些值将覆盖config中的默认值。 最后一个参数是服务器名称。 如果设置了服务器名称，则指定要连接的服务器，因此将程序放入客户端模式。 否则程序处于服务器模式。

1. 调用**print_config**打印输出配置信息。
2. 调用**resources_init**清除资源结构。
3. 调用**resources_create**
    1. 调用**sock_connect**使用TCP套接字与对方进行连接。
        1. 如果是客户端，则解析服务器的DNS地址并发起连接。
        2. 如果是服务器，则在指定端口上侦听传入连接。
    2. 获取设备列表，找到我们想要的设备，并打开它。
    3. 释放设备列表。
    4. 获取端口信息。
    5. 创建一个PD。
    6. 创建一个CQ。
    7. 分配一个缓冲区，初始化并注册。
    8. 创建一个QP。
4. 调用**connect_qp**
    1. 调用**sock_sync_data**来在服务器和客户端之间交换地址信息，使用由sock_connect创建的TCP套接字，在客户端和服务器之间同步给定的一组数据。 由于这个功能是阻塞的，所以也使用dummy data来同步客户端和服务器的时序。
    2. 调用**modify_qp_to_init** 将QP转换到INIT状态。
    3. 如果是客户端，则调用**post_receive**。
        1. 为接收缓冲区准备一个 scatter/gather 实体。
        2. 准备一个RR。
        3. post这个RR。
    4. 调用**modify_qp_to_rtr** 将QP转换到RTR状态。
    5. 调用**modify_qp_to_rts** 将QP转换到RTS状态。
    6. 调用**sock_sync_data**来同步客户端<->服务器的时序。
5. 如果在服务器模式下，请使用 IBV_WR_SEND 操作调用**post_send**。
    1. 为要发送的数据（或在RDMA读取情况下接收的数据）准备分散/收集条目。
    2. 创建SR。 请注意，IBV_SEND_SIGNALED是多余的。
    3. 如果是RDMA操作，请设置地址和密钥。
    4. post SR。
6. 调用**poll_completion**。 请注意，服务器端期望从SEND请求完成，并且客户端期望RECEIVE完成。
    1. 轮询CQ直到找到条目或达到MAX_POLL_CQ_TIMEOUT毫秒。
7. 如果在客户端模式下，则显示通过RECEIVE操作收到的消息。
8. 如果在服务器模式下，则使用新的消息填充缓冲区。
9. 同步 客户端<->服务器 的时序。
10. 在这一点上，服务器直接进入下一个时序同步。 所有RDMA操作都由客户端完成。  
11. **客户端**操作
    1. 使用IBV_WR_RDMA_READ调用**post_send**，执行服务器缓冲区的RDMA读取。  
    2. 调用**poll_completion**。  
    3. 显示服务器的消息。  
    4. 使用新消息设置发送缓冲区。  
    5. 使用IBV_WR_RDMA_WRITE调用**post_send**以执行服务器缓冲区的RDMA写入。  
    6. 调用**poll_completion**。  
12. 同步客户端< - >服务器的时序。  
13. 如果服务器模式，显示缓冲区，证明RDMA已经写入。  
14. 调用**resources_destroy** release/free/deallocate 资源结构中的所有项目。。  
15. 释放设备名称字符串。  
16. 完成。  
