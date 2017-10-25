---
title: CUDA简介
date: 2017-10-25 10:30:00
tags: cuda
categories: gpu
---

## 资料

<!-- more -->

## CPU与GPU

CPU（中央处理器）和GPU（图形处理器）由于其设计目标的不同，它们分别针对了两种不同的应用场景。

CPU需要很强的通用性来处理各种不同的数据类型，同时引入大量的分支跳转和中断来处理逻辑判断。这些使得CPU的内部结构异常复杂。

GPU面对的则是类型高度统一的、相互无依赖的大规模数据和不需要被打断的纯净的计算环境。

![CPU与GPU-1](http://outz1n6zr.bkt.clouddn.com/cuda-cpu-gpu-1.png)

上图中，绿色的是计算单元，橙红色的是存储单元，橙黄色的是控制单元。GPU采用了数量众多的计算单元和超长的流水线，但只有非常简单的控制逻辑并省去了Cache。而CPU不仅被Cache占据了大量空间，而且还有复杂的控制逻辑和诸多优化电路，相比之下计算能力只是CPU很小的一部分。cu

![CPU与GPU-2](http://outz1n6zr.bkt.clouddn.com/cuda-cpu-gpu-2.png)

 . | 缓存、内存 | 线程 | 寄存器 | SIMD单元
 --- | --- | --- | --- | ---
 CPU | 多 | 少 | 少 | 少
 GPU | 少|多|多|多

> SIMD Unit(单指令多数据流,以同步方式，在同一时间内执行同一条指令)

### CPU的设计理念——低延时

- 通过加强算术运算单元ALU来降低延时
    - 可以在很少的时钟周期内完成算术运算，执行双精度浮点运算也只需1 ~ 3个时钟周期。
    - Intel Core i7-7700K的主频高达4.2GHz。
- 通过增大缓存来降低延时
    - 增大缓存从而提升缓存命中率。  
- 复杂的逻辑控制单元
    - 分支预测降低分支延时
        - 当CPU遇到分支处理指令时(if-else)，无需等待判断结果，而由CPU自行决定执行哪个分支。当分支选择错误时，则放弃该分支，重新执行。
        - 常用的分支预测方法包括静态分支预测法和动态分支预测法。
        - 静态分支预测。任选一条分支，这样平均命中率为50%。更精确的办法是根据原先运行的结果进行统计从而尝试预测分支是否会跳转。
        - 动态分支预测。最简单的动态分支预测策略是分支预测缓冲区（Branch Prediction Buff)或分支历史表(branch history table)。
    - 通过数据转发降低数据延时
        - 当一些指令依赖前面的指令结果时，数据转发的逻辑控制单元决定这些指令在pipeline中的位置并且尽可能快的转发一个指令的结果给后续的指令。这些动作需要很多的对比电路单元和转发电路单元。

### GPU的设计理念——大吞吐量

- 小而少的缓存
    - 设计缓存的目的不是为了保存后面需要访问的数据(CPU)，而是为thread提供服务的。
    - 当很多threads需要访问同一个数据时，缓存会合并这些访问，然后去访问dram(数据存放地)。
    - 当缓存获取到数据后，会转发这个数据到对应的线程，这个时候是数据转发的角色。
    - 由于需要访问dram，所以带来了延时问题。
- 简单的控制单元
    - 配合缓存将多个访问合并成少的访问
    - 没有分支预测、数据转发
- 非常多的ALU
- 非常多的thread
- 通过大量线程的并行去忽略访问存储器的延时

### 形象的比喻

- CPU
    - 一个老教授，积分微分无所不能，可以完成十分复杂的任务。
    - 衡量CPU的指标是这个老教授有多厉害！
- GPU
    - 很多很多个小学生，每个小学生只会简单的加减乘除。
	- 衡量GPU的指标是有多少个小学生！

若有一项需要运算几亿次加法的任务，你选择哪个？

### GPU的应用场景

什么类型的程序适合在GPU上运行？

- 计算密集型的程序。即大部分运算时间花在寄存器运算上进行的运算。
- 易于并行的程序。GPU拥有成百上千个核，每一个核在同一时间最好能做同样的事情。

破解密码、挖矿、图形渲染、图像处理、深度学习、金融、生物学、……

### NVIDIA最新GPU一览

型号 | 计算能力 | 单精度性能(TFLOPS) | 显存 | CUDA Core
--- | --- | --- | --- | ---
高性能计算GPU Tesla V100 | 暂无 | 14 | 16GB | 5120
高性能计算GPU Tesla P100 | 6.0 | 9.3 | 16GB | 3584
专业制图GPU Quadro GP100 | 6.0 | 10.3 | 16GB | 3584
专业制图GPU Quadro P6000 | 6.1 | 12 | 24GB | 3840
桌面GPU GeForce NVIDIA TITAN Xp | 6.1 | 11 | 12GB | 3584
桌面GPU GeForce GTX TITAN X | 5.2 | 7 | 12GB | 3072 


> 计算能力Compute Capability。与计算速度无关，代表了其硬件层次的规格和可用功能，整数部分为GPU大的架构（1 Tesla 2 Fermi 3 Kepler 5 Maxwell 6 Pascal 7 Volta）。  
> 每秒浮点运算次数。1 TFLOPS=每秒1012次浮点运算

## GPU架构

### 硬件架构

SP：最基本的处理单元，streaming processor，也称为CUDA Core。最后具体的指令和任务都是在SP上处理的。GPU进行并行计算，也就是很多个SP同时做处理。

SM：多个SP加上其他的一些资源组成一个streaming multiprocessor。也叫GPU大核，可以看做GPU的心脏（对比CPU核心）。其他资源包括warp scheduler、register、shared memory等。

Tesla P100

- Tesla P100中包含一块GP100芯片，该芯片中包含60个SM。
- 1个SM中包含64个SP。
- 则Tesla P100总的SP个数为`60*64=3840`，而实际情况，P100只用了56个SM，则SP(CUDA Core)总数为`56*64=3584`。
- 软件逻辑上是所有SP是并行的，但是物理上并不是所有SP都能同时执行计算，由于资源问题会导致一些SP会处于挂起、就绪等其他状态，这有关GPU的线程调度。

### 软件架构

![软件架构](http://outz1n6zr.bkt.clouddn.com/04795d344f33b2a88945624f9428b27e.png)

thread：一个CUDA的并行程序会被以许多个threads来执行。

block：若干个threads会被群组成一个block，同一个block中的threads可以同步，也可以通过shared memory通信。可以是一维，二维或者三维。

grid：多个blocks则会再构成grid。可以是一维、二维或者三维。

warp：GPU执行程序时的调度单位，在CUDA架构中, 线程束是指包含32个线程的集合, 这些线程被"编织在一起", 并且以"步调一致(Lockstep)"的形式执行. 程序的每一行,线程束中的每个线程都将在不同的数据上执行相同的指令.

### 软硬件架构对应关系

当一个kernel函数启动后，thread会被分配到SM中执行。

大量的thread可能会被分配到不同的SM，同一个block中的threads必然在同一个SM中并行执行。

每个thread拥有它自己的程序计数器和状态寄存器，并且用该线程自己的数据执行指令。

一个SP可以执行一个thread，但是实际上并不是所有的thread能够在同一时刻执行。

Nvidia把32个threads组成一个warp，warp是调度和运行的基本单元。

warp中所有threads并行的执行相同的指令。一个warp需要占用一个SM运行，多个warps需要轮流进入SM。

由SM的硬件warp scheduler负责调度。目前每个warp包含32个threads。所以，一个GPU上resident thread最多只有 SM*warp个。 

![对应关系](http://outz1n6zr.bkt.clouddn.com/3b73373d63529ff8b36e10ccb8141b00.png)

block是软件概念，一个block只会由一个SM调度，程序员在开发时，通过设定block的属性来告诉GPU我需要多少个线程。而具体怎么调度由SM的warps scheduler负责，block一旦被分配好SM，该block就会一直驻留在该SM中，直到执行结束。一个SM可以同时拥有多个blocks，但需要序列执行。

需要注意的是，大部分threads只是逻辑上并行，并不是所有的thread可以在物理上同时执行。例如，遇到分支语句（if else，while，for等）时，各个thread的执行条件不一样必然产生分支执行，这就导致同一个block中的线程可能会有不同步调。另外，并行thread之间的共享数据会导致竞态：多个线程请求同一个数据会导致未定义行为。CUDA提供了同步函数来同步同一个block的thread以保证在进行下一步处理之前，所有thread都到达某个时间点。 

同一个warp中的thread可以以任意顺序执行，active warps被SM资源限制。当一个warp空闲时，SM就可以调度驻留在该SM中另一个可用warp。在并发的warp之间切换是没什么消耗的，因为硬件资源早就被分配到所有thread和block，所以该新调度的warp的状态已经存储在SM中了。不同于CPU，CPU切换线程需要保存/读取线程上下文（register内容），这是非常耗时的，而GPU为每个threads提供物理register，无需保存/读取上下文。 

### 编程查看GPU的信息

```c++
#include <stdio.h>
 
int main() {
    cudaDeviceProp  prop;
 
    int count;
    cudaGetDeviceCount( &count );
    for (int i=0; i< count; i++) {
        cudaGetDeviceProperties( &prop, i );
        printf( "   --- Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute Capability:  %d.%d\n", prop.major, prop.minor );
        printf( "   --- Hardware Information ---\n");
        printf( "Total Global Memory:  %ld\n", prop.totalGlobalMem );
        printf( "Total Constant Memory:  %ld\n", prop.totalConstMem );
        printf( "Streaming Multiprocessor count:  %d\n",prop.multiProcessorCount);
        printf( "Shared Memory per SM:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per SM:  %d\n", prop.regsPerBlock );
        printf( "   --- Software Information ---\n");
        printf( "Threads in Warp:  %d\n", prop.warpSize );
        printf( "Max Threads per Block:  %d\n",prop.maxThreadsPerBlock );
        printf( "Max Thread Dimensions:  (%d, %d, %d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        printf( "Max Grid Dimensions:  (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
        printf( "\n" );
    }
}
```

输出如下

```
   --- Information for device 0 ---
Name:  Tesla P100-PCIE-16GB
Compute Capability:  6.0
   --- Hardware Information ---
Total Global Memory:  17066885120
Total Constant Memory:  65536
Streaming Multiprocessor count:  56
Shared Memory per SM:  49152
Registers per SM:  65536
   --- Software Information ---
Threads in Warp:  32
Max Threads per Block:  1024
Max Thread Dimensions:  (1024, 1024, 64)
Max Grid Dimensions:  (2147483647, 65535, 65535)
```

## CUDA C基础

### CUDA发展史

2007年以前，通过OpenGL和DirectX的图形API来执行通用计算，难度极大。

2007年NVIDIA推出了CUDA（Computert Unified Device Architecture，统一计算架构）编程环境。

CUDA Toolkit（CUDA SDK）不断升级，从2007年推出的1.0到2017年推出的9.0（latest）。

### 并发与并行

并发：并发是一个程序、算法或者问题的可分解属性，它由多个顺序不依赖性或者局部顺序依赖性的结构或单元组成。这就意味着这些单元无论以何种顺序执行或者运算，最终结果都是一样的。

并行：并行(parallelism)是指在具有多个处理单元(如GPU或者多核CPU)的系统上，通过将计算或数据划分为多个部分，将各个部分分配到不同的处理单元上，各处理单元相互协作，同时运行，已达到加快求解速度或提高求解问题规模的目的。
