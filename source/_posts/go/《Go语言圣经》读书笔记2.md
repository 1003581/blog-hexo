---
title: 《Go语言圣经》读书笔记2
date: 2017-10-19 09:04
tags: go
categories: go
---
## 链接

- [GitHub](https://github.com/golang-china/gopl-zh)
- [读书地址](https://docs.hacknode.org/gopl-zh/)
- [另一本书《the-way-to-go》](https://github.com/Unknwon/the-way-to-go_ZH_CN)
<!-- more -->

## 基于共享变量的并发

### 竞争条件

数据竞争会在两个以上的goroutine并发访问相同的变量且至少其中一个为写操作时发生

不要使用共享数据来通信；使用通信来共享数据

### sync.Mutex互斥锁

### sync.RWMutex读写锁

“多读单写”锁(multiple readers, single writer lock)

```go
var mu sync.RWMutex
var balance int
func Balance() int {
    mu.RLock() // readers lock
    defer mu.RUnlock()
    return balance
}
```

### 内存同步

```go
var x, y int
go func() {
    x = 1 // A1
    fmt.Print("y:", y, " ") // A2
}()
go func() {
    y = 1                   // B1
    fmt.Print("x:", x, " ") // B2
}()
```

可能出现`x:0 y:0`

因为赋值和打印指向不同的变量，编译器可能会断定两条语句的顺序不会影响执行结果，并且会交换两个语句的执行顺序。如果两个goroutine在不同的CPU上执行，每一个核心有自己的缓存，这样一个goroutine的写入对于其它goroutine的Print，在主存同步之前就是不可见的了。

### sync.Once初始化

```go
var loadIconsOnce sync.Once
var icons map[string]image.Image
// Concurrency-safe.
func Icon(name string) image.Image {
    loadIconsOnce.Do(loadIcons)
    return icons[name]
}
```

每一次对Do(loadIcons)的调用都会锁定mutex，并会检查boolean变量。在第一次调用时，boolean变量的值是false，Do会调用loadIcons并会将boolean变量设置为true。随后的调用什么都不会做，但是mutex同步会保证loadIcons对内存(这里其实就是指icons变量啦)产生的效果能够对所有goroutine可见。用这种方式来使用sync.Once的话，我们能够避免在变量被构建完成之前和其它goroutine共享该变量。

### 竞争条件检测

竞争检查器(the race detector)。

只要在go build，go run或者go test命令后面加上-race的flag，就会使编译器创建一个你的应用的“修改”版或者一个附带了能够记录所有运行期对共享变量访问工具的test，并且会记录下每一个读或者写共享变量的goroutine的身份信息。

### 示例: 并发的非阻塞缓存

### Goroutines和线程

动态栈

- 每一个OS线程都有一个固定大小的内存块(一般会是2MB)来做栈，这个栈会用来存储当前正在被调用或挂起(指在调用其它函数时)的函数的内部变量。
- 一个goroutine会以一个很小的栈开始其生命周期，一般只需要2KB。
- 一个goroutine的栈，和操作系统线程一样，会保存其活跃或挂起的函数调用的本地变量
- 但是和OS线程不太一样的是，一个goroutine的栈大小并不是固定的；栈的大小会根据需要动态地伸缩。而goroutine的栈的最大值有1GB，比传统的固定大小的线程栈要大得多，尽管一般情况下，大多goroutine都不需要这么大的栈。

Goroutine调度

- OS线程会被操作系统内核调度。
- 线程切换：每几毫秒，一个硬件计时器会中断处理器，这会调用一个叫作scheduler的内核函数。这个函数会挂起当前执行的线程并将它的寄存器内容保存到内存中，检查线程列表并决定下一次哪个线程可以被运行，并从内存中恢复该线程的寄存器信息，然后恢复执行该线程的现场并开始执行线程。
- Go的运行时包含了其自己的调度器，这个调度器使用了一些技术手段，比如m:n调度，因为其会在n个操作系统线程上多工(调度)m个goroutine。
- Go调度器的工作和内核的调度是相似的，但是这个调度器只关注单独的Go程序中的goroutine。
- 和操作系统的线程调度不同的是，Go调度器并不是用一个硬件定时器，而是被Go语言“建筑”本身进行调度的。例如当一个goroutine调用了time.Sleep，或者被channel调用或者mutex操作阻塞时，调度器会使其进入休眠并开始执行另一个goroutine，直到时机到了再去唤醒第一个goroutine。因为这种调度方式不需要进入内核的上下文，所以重新调度一个goroutine比调度一个线程代价要低得多。

GOMAXPROCS

- Go的调度器使用了一个叫做GOMAXPROCS的变量来决定会有多少个操作系统的线程同时执行Go的代码。其默认的值是运行机器上的CPU的核心数
- 你可以用GOMAXPROCS的环境变量来显式地控制这个参数，或者也可以在运行时用runtime.GOMAXPROCS函数来修改它。
```go
for {
    go fmt.Print(0)
    fmt.Print(1)
}

$ GOMAXPROCS=1 go run hacker-cliché.go
111111111111111111110000000000000000000011111...

$ GOMAXPROCS=2 go run hacker-cliché.go
010101010101010101011001100101011010010100110...
```

Goroutine没有ID号

- 线程都有一个独特的身份(id)

## 包和工具

### 包简介

### 导入路径

### 包声明

默认包名一般采用导入路径名的最后一段的约定也有三种例外情况。

- 包对应一个可执行程序，也就是main包
- 包所在的目录中可能有一些文件名是以`_test.go`为后缀的Go源文件
- 一些依赖版本号的管理工具会在导入路径后追加版本号信息`gopkg.in/yaml.v2`

### 导入声明

导入重命名，别名

```go
import (
    "crypto/rand"
    mrand "math/rand" // alternative name mrand avoids conflict
)
```

- 导入包的重命名只影响当前的源文件
- 导入包重命名是一个有用的特性，它不仅仅只是为了解决名字冲突。如果导入的一个包名很笨重，特别是在一些自动生成的代码中，这时候用一个简短名称会更方便。

### 包的匿名导入

用途：计算包级变量的初始化表达式和执行导入包的init初始化函数

```go
import _ "image/png" // register PNG decoder
```

### 包和命名

- 当创建一个包，一般要用短小的包名，但也不能太短导致难以理解。标准库中最常用的包有bufio、bytes、flag、fmt、http、io、json、os、sort、sync和time等包。
- 包名一般采用单数的形式。

### 工具

工作区结构

- `go env`

下载包

- `go get`
- `git clone` & `go build`

构建包

- `go build` -> 构建指定的包和它依赖的包，然后丢弃除了最后的可执行文件之外所有的中间编译结果。
- `go install` ->被编译的包保存到$GOPATH/pkg，可执行程序被保存到$GOPATH/bin

包文档

- `go doc`

内部包

- internal包
- 一个internal包只能被和internal目录有同一个父目录的包所导入。

查询包

- `go list ...`

## 测试

### go test

`_test.go`

在`*_test.go`文件中，有三种类型的函数：测试函数、基准测试(benchmark)函数、示例函数。

- 一个测试函数是以Test为函数名前缀的函数，用于测试程序的一些逻辑行为是否正确；
	- go test命令会调用这些测试函数并报告测试结果是PASS或FAIL。
- 基准测试函数是以Benchmark为函数名前缀的函数，它们用于衡量一些函数的性能；
	- go test命令会多次运行基准测试函数以计算一个平均的执行时间。
- 示例函数是以Example为函数名前缀的函数，提供一个由编译器保证正确性的示例文档。

go test命令会遍历所有的*_test.go文件中符合上述命名规则的函数，生成一个临时的main包用于调用相应的测试函数，接着构建并运行、报告测试结果，最后清理测试中生成的临时文件。

### 测试函数
