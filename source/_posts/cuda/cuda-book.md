---
title: 《GPU高性能编程CUDA实战》
date: 2017-09-14 16:03:16
tags: cuda
categories: gpu
---

## 资料
<!-- more -->
[官网书籍介绍](https://developer.nvidia.com/cuda-example)

[书籍代码 Github](https://github.com/liqiang311/cuda_book)

[[GPU高性能编程CUDA实战].(桑德斯).聂雪军等.扫描版.pdf 百度云](https://pan.baidu.com/s/1i5f2TNZ)

[[GPU高性能编程CUDA实战].(桑德斯).聂雪军等.扫描版.pdf 七牛云](http://outz1n6zr.bkt.clouddn.com/%5BGPU%E9%AB%98%E6%80%A7%E8%83%BD%E7%BC%96%E7%A8%8BCUDA%E5%AE%9E%E6%88%98%5D.%28%E6%A1%91%E5%BE%B7%E6%96%AF%29.%E8%81%82%E9%9B%AA%E5%86%9B%E7%AD%89.%E6%89%AB%E6%8F%8F%E7%89%88.pdf)

[其他人学习笔记](http://blog.csdn.net/w09103419/article/category/6402290/1)


## 概念

### 函数类型

`__global__` 在CPU上调用GPU函数

`__device__` 在GPU上调用GPU函数

`__host__` 在CPU上调用CPU函数

### 并行线程块

`kernel<<<blocks,threads>>>();` `kernel<<<dim3(DIM1,DIM2,DIM3), dim3(DIM1,DIM2,DIM3)>>>();`

第一个参数`blocks`表示要启动的线程块的个数, 第一个参数`threads`表示每个线程块中线程的个数. `blocks`, `threads`可以为常数, 也可以为二维或三维数.

在核函数中, 内置变量`blockDim`保存线程块中每一维的线程个数, 对于一维线程块, `blockDim.x`即为线程个数. `GridDim`中包含线程格中线程块的个数.

理论上CUDA支持允许运行一个二维线程格(Grid), 其中的每个线程块包含三维的线程数组.

所以, 核函数中线程id运算公式如下:

```
//一维
int tid = threadIdx.x + blockIdx.x * blockDim.x;

//二维
int x = threadIdx.x + blockIdx.x * blockDim.x; //左边有x个线程
int y = threadIdx.y + blockIdx.y * blockDim.y; //上方有y个线程
int offset = x + y * blockDim.x * gridDim.x; //实际线程Idx
```

### 全局内存

```
HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice ) );
HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost ) );
HANDLE_ERROR( cudaFree( dev_a ) );
```

可以用`cudaMemcpyDefault`来代替`cudaMemcpyHostToDevice`和`cudaMemcpyDeviceToHost`。

### 共享内存

将`__share__`添加到变量声明中, 对于GPU中的每个线程块, CUDA C编译器都创建该变量的一个副本. 同一线程块中的线程共享这块内存, 不同线程块之间内存不可见. 此内存常驻在GPU上.

### 线程同步

`__syncthreads();` 该函数调用将确保线程块中的每个线程都执行完`__syncthreads()`前面的语句后, 才会执行下一条语句. 

注意: `__syncthreads()` 不能写在条件判断中, 部分会线程会无限等待一个不会执行的`__syncthreads()`操作. 

### 常量内存

将`__constant__`添加到变量声明中, 无需`cudaMalloc`, 而是直接静态分配数组空间。复制时使用`cudaMemcpyToSymbol`.

```
__constant__ Sphere s[SPHERES];
HANDLE_ERROR( cudaMemcpyToSymbol( s, temp_s, sizeof(Sphere) * SPHERES) );
```

常量内存中的数据为只读权限, 从常量内存中读取数据相比全局内存有2点好处. 

- 对常量内存的单次读操作可以广播到其他的"邻近(Nearby)线程", 这将节约15次读取操作.
- 常量内存的数据将缓存起来, 因此对相同地址的连续读操作将不会产生额外的内存通信量.

线程束(Wrap): 在CUDA架构中, 线程束是指包含32个线程的集合, 这些线程被"编织在一起", 并且以"步调一致(Lockstep)"的形式执行. 程序的每一行,线程束中的每个线程都将在不同的数据上执行相同的指令.

### 纹理内存

CUDA架构中的另一种只读内存, 同样为了加速读取, 减少对内存的请求从而提高高效的内存带宽. GPU将缓存该块内存附近的数据, 并分享到其他线程. 

一维纹理内存访问

```
// 创建纹理内存引用, 必须为全局变量
texture<float>  texConstSrc;

// 初始化函数中
// 首先在GPU上分配内存
HANDLE_ERROR( cudaMalloc( (void**)&dev_constSrc, imageSize ) );
// 告诉GPU我们希望将指定的缓冲区作为纹理来使用
// 我们希望将纹理引用作为纹理的"名字"
// 当读取内存时, 用纹理引用texConstSrc, 当要写内存时, 需用全局变量dev_constSrc
HANDLE_ERROR( cudaBindTexture( NULL, texConstSrc, dev_constSrc, imageSize ) );

// 核函数中
// 核函数中读取方式不再是[]读取, 而要调用函数
float c = tex1Dfetch(texConstSrc, offset)

// 释放函数
cudaUnbindTexture( texConstSrc );
HANDLE_ERROR( cudaFree( dev_constSrc ) );
```

二维纹理内存访问

```
texture<float>  texConstSrc; //必须为全局变量

HANDLE_ERROR( cudaMalloc( (void**)&dev_constSrc, imageSize ) );
cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
HANDLE_ERROR( cudaBindTexture2D( NULL, texConstSrc, dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM ) );
```

### 时间统计

```
cudaEvent_t     start, stop;
HANDLE_ERROR( cudaEventCreate( &start ) );
HANDLE_ERROR( cudaEventCreate( &stop ) );
HANDLE_ERROR( cudaEventRecord( start, 0 ) );

...

HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
HANDLE_ERROR( cudaEventSynchronize( stop ) );
float   elapsedTime;
HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
printf( "Time to generate:  %3.1f ms\n", elapsedTime );

HANDLE_ERROR( cudaEventDestroy( start ) );
HANDLE_ERROR( cudaEventDestroy( stop ) );
```

### 原子性

`atomicAdd(var, 1);`

### 页锁定主机内存

`malloc()`函数分配的主机内存, 是标准的, 可分页的主机内存. `cudaHostAlloc()` 将分配页锁定的主机内存, 这块内存将不会被操作系统交换到磁盘上, 或者被重新定位. 

```
cudaHostAlloc( (void**)&a, sizeof * sizeof(*a), cudaHostAllocDefault) );
cudaFreeHost(a)
```

### 流

类似于一个队列, 把所有的copy任务,核函数计算任务,全部放入这个队列. 然后异步执行, 最后等待它结束. 

宽度优先, 而非深度优先.

```
cudaStream_t    stream;
HANDLE_ERROR( cudaStreamCreate( &stream ) );
HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
HANDLE_ERROR( cudaHostAlloc( (void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault ) );
for (int i=0; i<FULL_DATA_SIZE; i+= N) {
  HANDLE_ERROR( cudaMemcpyAsync( dev_a, host_a+i, N * sizeof(int), cudaMemcpyHostToDevice, stream ) );
  kernel<<<N/256,256,0,stream>>>( dev_a, dev_b, dev_c );
  HANDLE_ERROR( cudaMemcpyAsync( host_c+i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream ) );
}
HANDLE_ERROR( cudaStreamSynchronize( stream ) );
HANDLE_ERROR( cudaFreeHost( host_a ) );
HANDLE_ERROR( cudaFree( dev_a ) );
HANDLE_ERROR( cudaStreamDestroy( stream ) );
```

### 零拷贝主机内存

对于集成GPU来说, 零拷贝主机内存会带来性能提升. 对于独立GPU来说, 如果输入内存和输出内存都只使用一次的话, 可以使用零拷贝主机内存, 因为GPU不会缓存此内存, 所以多次访问会降低性能. 

`cudaHostAllocMapped` 标识表示运行时将从GPU中访问这块内存(零拷贝主机内存).

`cudaHostAllocWriteCombined` 表示运行时应该将内存分配为"合并式写入"内存. 会显著地提升GPU读取内存时的性能, 但会降低CPU的读取性能. 

`cudaHostAllocPortable` 标识每个线程都把这块内存当作固定内存

`cudaHostGetDevicePointer()`获取此块内存在GPU上的指针

```
HANDLE_ERROR( cudaHostAlloc( (void**)&a, N*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped ) );
HANDLE_ERROR( cudaHostGetDevicePointer( &dev_a, a, 0 ) );
```

## 常用函数

### 检查每次调用是否正常

参考文件 `common/book.h`

```
static void HandleError( cudaError_t err, const char *file, int line ) 
{
    if (err != cudaSuccess) 
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
```

### 查询显卡信息

参考文件 `chapter03/enum_gpu.cu`

[官方DOC](http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#axzz4oTEAox3L) 

```
cudaDeviceProp  prop;
int count;
HANDLE_ERROR( cudaGetDeviceCount( &count ) );
for (int i=0; i< count; i++) {
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
}
```

常用信息

- `prop.name` 设备名称
- `prop.totalGlobalMem` 全局内存的总量, 单位为字节
- `prop.sharedMemPerBlock` 一个线程块(Block)中可使用的最大共享内存总量, 单位为字节
- `prop.warpSize` 一个线程束(Wrap)中包含的线程总量
- `prop.maxTreadsPerBlock` 一个线程块中可以包含的最大线程数量
- `prop.maxThreadsDim[3]` 多维线程块数组中, 每一维可以包含的最大线程个数
- `prop.maxGridSize[3]` 一个线程格中, 每一维可以包含的最大线程块个数
- `prop.major` 计算主版本号
- `prop.minor` 计算次版本号

示例输出

```
   --- General Information for device 0 ---
Name:  Tesla P100-PCIE-16GB
Compute capability:  6.0
Clock rate:  1328500
Device copy overlap:  Enabled
Kernel execution timeout :  Disabled
   --- Memory Information for device 0 ---
Total global mem:  17066885120
Total constant Mem:  65536
Max mem pitch:  2147483647
Texture Alignment:  512
   --- MP Information for device 0 ---
Multiprocessor count:  56
Shared mem per mp:  49152
Registers per mp:  65536
Threads in warp:  32
Max threads per block:  1024
Max thread dimensions:  (1024, 1024, 64)
Max grid dimensions:  (2147483647, 65535, 65535)
```

### 选择显卡

参考文件 `chapter03/set_gpu.cu`

```
cudaDeviceProp  prop;
memset( &prop, 0, sizeof( cudaDeviceProp ) );
prop.major = 1;
prop.minor = 3;
int dev;
HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
HANDLE_ERROR( cudaSetDevice( dev ) );
```

### 核函数维度

```
dim3 grid(DIM1,DIM2,DIM3=1)

kernel<<<grid,1>>>();
```


### 特殊代码

- `julia(高维线程格)` -> `chapter04/julia_gpu.cu`
- `点积(共享内存, 线程同步)` -> `chapter05/dot.cu`
- `波纹图(共享内存)` -> `chapter05/ripple.cu`
- `位图(共享内存)` -> `chapter05/shared_bitmap.cu`
- `光线跟踪(人眼观察3维环境)(常量内存)` -> `chapter06/ray.cu`
- `热传导模拟(纹理内存)` -> `chapter07/heat_2d.cu`
- `直方图计算(原子性, 共享内存)` -> `chapter09/hist_gpu_shmem_atomics.cu`
- `单个流调度` -> `chapter10/basic_single_stream.cu`
- `2个流调度(宽度优先)` -> `chapter10/basic_double_stream_correct.cu`

