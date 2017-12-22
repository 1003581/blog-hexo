---
title: ibverbs文档翻译1
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

![image](http://upload-images.jianshu.io/upload_images/5952841-d362fbc0a1e98aad.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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
