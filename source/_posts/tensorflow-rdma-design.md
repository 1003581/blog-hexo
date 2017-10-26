---
title: TensorFlow中rdma的设计细节
date: 2017-09-14 15:59:41
tags: 
- tensorflow
- rdma
categories: tensorflow
---

# [Code Link](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/verbs)

<!-- more -->

## 如何编译和使用启用RDMA的TensorFlow

1. 按照常规的TF编译说明进行操作。 在配置步骤中，如果您想要支持基于ibverbs的RDMA，请回答此问题：

`Do you wish to build TensorFlow with VERBS-RDMA support [y/N]`

1. 要打开RDMA连接，请在服务器定义中添加协议“grpc + verbs”：

`server = tf.train.Server(cluster, job_name="local", task_index=0, protocol='grpc+verbs') # default protocol is 'grpc'`

## 概要

该设计基于TensorFlow r1.0。在服务器之间添加一个RDMA路径用于tensor传递（权重，梯度等）。现有的gRPC路径保留并负责“管理”任务，如设置RDMA路径，交换计算图等。

在服务器设置期间，创建一个RDMA管理器来管理低级RDMA组件，如RDMA通道和RDMA适配器，创建一个RDMA会合管理器来监视服务器之间的发送/重启操作。按照分布式TensorFlow设计理念，发送操作是被动的，即仅在本地输出表中放置tensor。而实际启动tensor传递的是接收操作。

TensorFlow为要发送或接收的tensor动态分配内存。这对于需要固定存储器的RDMA操作来说是困难的。可以使用两种补救措施：存储器被固定，传输，然后对每个tensor进行去传输，或者为每个tensor预先分配和固定缓冲区。前者引起显着的操作开销，因为针对每个动态生成的tensor的固定和解除内存是缓慢的。后者则引起大量的内存开销，并包含从tensor到其固定缓冲区的额外复制，但仍可能比前者快。本设计采用第二种方法。RDMA通道代表为一对tensor建立的RDMA连接，这个通道包含一张表，表里存放每个需要被传输的tensor的对应固定存储器。假设tensor大小在不同的步骤之间很少改变。因此，在所有步骤中只为同一tensor创建一个缓冲区。在tensor大小确实增加的罕见情况下，旧的缓冲区被丢弃，同时创建并固定较大尺寸的新缓冲区。

当tensor准备传输时，首先将其转换为TensorProto，然后将proto序列化为字节数组并复制到固定缓冲区。缓冲区的内容通过RDMA写入传输到远程节点。在远程端，该过程相反。这在下图中说明。引入TensorProto的转换来简化tensor字符串的传递。此外，由于TensorProto存在主机内存中，即使原始tensor存在于设备中，固定的缓冲区也全部分配在主机内存中。

![TensorFlow RDMA path](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/verbs/design_diagram.png?raw=true)

以后可以做出以下改进。首先，数字（float / int）tensor可以避免转换为TensorProto和进行序列化，因为它们的内部缓冲区可以直接作为字节数组访问。第二，如果tensor位于设备中，则可以在设备上分配固定缓冲区。这样可以避免额外的设备到主机的拷贝，而不需要额外的设备内存消耗。

## 设计细节

### RDMA components

* **RDMA适配器**：RDMA通信的基础。它可能包含多个通道和缓冲区。它负责处理各种传入的RDMA消息。
* **RDMA通道**：负责与特定节点的RDMA连接。它管理多个缓冲区。一个通道有一个回调表，用于存储所有请求传输tensor的回调。
* **RDMA缓冲区**：负责发送或接收数据。它有一个固定大小的内存来存储数据。它有一个队列来存储挂起的作业。有三种类型的缓冲区，消息缓冲区，ACK缓冲区和tensor缓冲区。一个通道有两个消息缓冲区，两个ack缓冲区和许多tensor缓冲区。
* **RDMA manager**：管理适配器和通道，包括通道创建，通过gRPC服务进行通道设置，通道查找等。
* **RDMA rendezvous manager**：管理多个rdma rendezvous。
* **RDMA rendezvous:** 是BaseRemoteRendezvous的派生类。是send和recv操作的后端。当send和recv操作要发送或接收tensor时，它会分别调用rendezvous的send和recv功能。随机数step_id用来标识rendezvous，因此不同的tensor不会混淆。

### The SEND operation

在TensorFlow中，当rendezvous发送tensor时，它只是将tensor放在相应rendezvous的本地table中。如果tensor已经被请求，那么table中就有一个callback。 send将激活callback(remote)，该callback将尝试通过节点发送tensor。

### The RECV operation

当请求tensor时，rendezvous的recv将被调用。首先recv函数将callback放在channel的回调表中，该callback将在tensor（从源发送的）到达后被调用。接下来，接收者发送一条Message（TENSOR_REQUEST）以通知要请求tensor的源。一旦源接收到此Message，它将在本地检查tensor，如果没有找到，则在本地channel的回调表中放置接收方的callback，否则tensor id将被放置到相应的RDMA buffer的job queue中以供将来传输。当一个tensor被调度传输时，RDMA Buffer需要分配和初始化内存（用远程缓冲区信息注册(大小)）。如果存储器未准备就绪，则传输被延迟，同时先发一条消息到目的地来建立存储器。传输可以推迟的另一种情况是当缓冲器仍被正在进行的传输使用时。

### 三种类型的RDMA缓冲区

* **消息缓冲区**：仅负责发送消息。

* **Ack缓冲区**：一旦发送消息，收件人需要通过ack缓冲区发送ack以释放消息缓冲区。 一个ack缓冲区专用于其耦合的消息缓冲区。

* **tensor缓冲区**：负责发送tensor。收件人需要发送一个消息来释放发送缓冲区。

### RDMA packet format

`|type|name_size|name|step_id|buffer_size|remote_addr|rkey|is_dead|data_type|tensor_shape|tensor_bytes|tensor_buffer|`

### 六种类型的RDMA消息

* RDMA_MESSAGE_ACK
* RDMA_MESSAGE_BUFFER_IDLE
* RDMA_MESSAGE_BUFFER_REQUEST
* RDMA_MESSAGE_BUFFER_RESPONSE
* RDMA_MESSAGE_TENSOR_REQUEST
* RDMA_MESSAGE_TENSOR_WRITE

### 接收RDMA消息时的操作

* RDMA_MESSAGE_ACK
  * sender: mark local ack buffer idle.置本地ack缓冲区空闲。
  * receiver: mark remote message buffer idle, send next item.置远程message缓冲区空闲，本地message缓冲区开始SendNext。
* RDMA_MESSAGE_BUFFER_IDLE
  * sender: mark local message buffer idle, send next item.标记本地消息缓冲区空闲，发送下一个项目。
  * receiver: send ack, set remote tensor buffer idle, send next item.发送ack，设置远程tensor缓冲区空闲，发送下一个项目。
* RDMA_MESSAGE_BUFFER_REQUEST
  * sender: mark local message buffer idle, send next item.标记本地消息缓冲区空闲，发送下一个项目。
  * receiver: send ack, find or create tensor buffer, send BUFFER_RESPONSE.发送ack，查找或创建tensor缓冲区，发送BUFFER_RESPONSE。
* RDMA_MESSAGE_BUFFER_RESPONSE
  * sender: mark local message buffer idle, send next item.标记本地消息缓冲区空闲，发送下一个项目。
  * receiver: send ack, set remote buffer info, set local and remote buffer idle, send next item.发送ack，设置远程缓冲区信息，设置本地和远程缓冲区空闲，发送下一个项目。
* RDMA_MESSAGE_TENSOR_REQUEST
  * sender: mark local message buffer idle, send next item.标记本地消息缓冲区空闲，发送下一个项目。
  * receiver: send ack, find or create tensor buffer, enqueue tensor id, send next item.发送ack，查找或创建tensor缓冲区，入库tensorid，发送下一个项目。
* RDMA_MESSAGE_TENSOR_WRITE
  * sender: mark local message buffer idle, send next item.标记本地消息缓冲区空闲，发送下一个项目。
  * receiver: run callback.运行回调。

