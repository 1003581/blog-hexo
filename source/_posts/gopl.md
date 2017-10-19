---
title: 《Go语言圣经》读书笔记
date: 2017-10-19 09:04
tags: go
categories: go
---

## 链接

- [GitHub](https://github.com/golang-china/gopl-zh)
- [读书地址](https://docs.hacknode.org/gopl-zh/)

<!-- more -->

## 前言

### Go与C++的比较

技术选型

- Python+Redis 脚本语言开发迅速、数据结构丰富、也有多线程等模式 无法处理复杂数据结构、需要做C扩展或者使用很多复杂的实现方式，常驻内存后端程序还是不够稳定，无法处理非常高性能的场合，也不方便优化 
- C++ +Redis 高性能、极强的定制性、可以实现任何内存和数据结构、可以达到任何功能和性能的要求和强大的底层操控能力 开发维护成本高、出现问题几率较大、没有专职C++开发人员，需要构建很多基础库、包括异步网络框架，采用多线程还是异步IO、配置文件解析、内存管理、日志等基础库的开发 
- Go+Redis 学习曲线短、高性能、高并发，支持强大的的类C的内存和各种数据结构操作可以满足一般场景需求

`=` 和 `:= `

- C++中没有`:=`这个符号
- Go语言中：=用来声明一个变量的同时给这个变量赋值 并且只能在函数体内使用
- 主要是为了省略类型声明，系统会自己选择相应的类型识别定义的变量

New和make操作符

- C++中new操作符用于给一个对象 分配内存但是没有清零
- go语言中调用new（T）被求值时 所做的是为T类型的新值分配并且清零一块内存空间，然后将内存空间的地址作为结果返回 所以不用担心乱码问题 可以直接拿来使用
- make用于内建类型（map、slice 和channel）的内存分配。new用于各种类型的内存分配。 
- make只能创建slice、map和channel，并且返回一个有初始值(非零)的T类型（引用），而不是*T。

面向对象

- go语言和c++在面向对象方面完全没有可比性，尤其是没有继承、泛型、虚函数、函数重载、构造函数、析构函数等。

## 入门

Go语言编译过程没有警告信息，争议特性之一

在表达式x + y中，可在+后换行，不能在+前换行（译注：以+结尾的话不会被插入分号分隔符，但是以x结尾的话则会被分号分隔符，从而导致编译错误）

根据代码需要, 自动地添加或删除import声明 `go get golang.org/x/tools/cmd/goimports`

### 命令行参数

```go
package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	var s, sep string
	for _, args := range os.Args {
		s += sep + args
		sep = " "
	}
	fmt.Println(s)

	fmt.Println(strings.Join(os.Args, " "))
}
```

### 查找重复的行

```
%d          十进制整数
%x, %o, %b  十六进制，八进制，二进制整数。
%f, %g, %e  浮点数： 3.141593 3.141592653589793 3.141593e+00
%t          布尔：true或false
%c          字符（rune） (Unicode码点)
%s          字符串
%q          带双引号的字符串"abc"或带单引号的字符'c'
%v          变量的自然形式（natural format）
%T          变量的类型
%%          字面上的百分号标志（无操作数）
```

```go
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	counts := make(map[string]int)
	files := os.Args[1:]
	if len(files) == 0 {
		countLines(os.Stdin, counts)
	} else {
		for _, arg := range files {
			f, err := os.Open(arg)
			if err != nil {
				fmt.Fprintf(os.Stderr, "dup2: %v\n", err)
				continue
			}
			countLines(f, counts)
			f.Close()
		}
	}
	for line, n := range counts {
		if n > 1 {
			fmt.Printf("%d\t%s\n", n, line)
		}
	}
}

func countLines(f *os.File, counts map[string]int) {
	input := bufio.NewScanner(f)
	for input.Scan() {
		counts[input.Text()]++
	}
	// NOTE: ignoring potential errors from input.Err()
}
```

### GIF动画

```go
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"io"
	"math"
	"math/rand"
	"os"
	"time"
)

//var palette = []color.Color{color.White, color.Black}
var palette = []color.Color{color.RGBA{0x00, 0xff, 0x00, 0xff}, color.RGBA{0xff, 0x00, 0x00, 0xff}}

const (
	backgroudIndex = 0 // first color in palette
	foregroudIndex = 1 // next color in palette
)

func main() {
	// The sequence of images is deterministic unless we seed
	// the pseudo-random number generator using the current time.
	// Thanks to Randall McPherson for pointing out the omission.
	rand.Seed(time.Now().UTC().UnixNano())
	fout, err := os.Create("out.gif")
	defer fout.Close()
	if err != nil {
		fmt.Println("out.gif", err)
		return
	}
	//lissajous(os.Stdout)
	lissajous(fout)
}

func lissajous(out io.Writer) {
	const (
		cycles  = 5     // number of complete x oscillator revolutions
		res     = 0.001 // angular resolution
		size    = 100   // image canvas covers [-size..+size]
		nframes = 64    // number of animation frames
		delay   = 8     // delay between frames in 10ms units
	)

	freq := rand.Float64() * 3.0 // relative frequency of y oscillator
	anim := gif.GIF{LoopCount: nframes}
	phase := 0.0 // phase difference
	for i := 0; i < nframes; i++ {
		rect := image.Rect(0, 0, 2*size+1, 2*size+1)
		img := image.NewPaletted(rect, palette)
		for t := 0.0; t < cycles*2*math.Pi; t += res {
			x := math.Sin(t)
			y := math.Sin(t*freq + phase)
			img.SetColorIndex(size+int(x*size+0.5), size+int(y*size+0.5),
				foregroudIndex)
		}
		phase += 0.1
		anim.Delay = append(anim.Delay, delay)
		anim.Image = append(anim.Image, img)
	}
	gif.EncodeAll(out, &anim) // NOTE: ignoring encoding errors
}
```

### 并发获取多个URL

```go
package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"time"
)

func main() {
	start := time.Now()
	ch := make(chan string)
	for _, url := range os.Args[1:] {
		go fetch(url, ch) // start a goroutine
	}
	for range os.Args[1:] {
		fmt.Println(<-ch) // receive from channel ch
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}

func fetch(url string, ch chan<- string) {
	start := time.Now()
	if ok := strings.HasPrefix(url, "http"); !ok {
		url = "http://" + url
	}
	resp, err := http.Get(url)
	if err != nil {
		ch <- fmt.Sprint(err) // send to channel ch
		return
	}

	nbytes, err := io.Copy(ioutil.Discard, resp.Body)
	resp.Body.Close() // don't leak resources
	if err != nil {
		ch <- fmt.Sprintf("while reading %s: %v", url, err)
		return
	}
	secs := time.Since(start).Seconds()
	ch <- fmt.Sprintf("%.2fs  %7d  %s", secs, nbytes, url)
}
```

### Web服务

```go
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"sync"
)

var mu sync.Mutex
var count int

func main() {
	http.HandleFunc("/", handler)
	http.HandleFunc("/gif", handlerGif)
	http.HandleFunc("/count", counter)
	log.Fatal(http.ListenAndServe("localhost:8000", nil))
}

// handler echoes the Path component of the requested URL.
func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "%s %s %s\n", r.Method, r.URL, r.Proto)
	for k, v := range r.Header {
		fmt.Fprintf(w, "Header[%q] = %q\n", k, v)
	}
	fmt.Fprintf(w, "Host = %q\n", r.Host)
	fmt.Fprintf(w, "RemoteAddr = %q\n", r.RemoteAddr)
	if err := r.ParseForm(); err != nil {
		log.Print(err)
	}
	for k, v := range r.Form {
		fmt.Fprintf(w, "Form[%q] = %q\n", k, v)
	}
}

// counter echoes the number of calls so far.
func counter(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	fmt.Fprintf(w, "Count %d\n", count)
	mu.Unlock()
}

func handlerGif(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		log.Print(err)
	}
	cycles, _ := strconv.ParseFloat(r.Form.Get("cycles"), 10)
	log.Println(cycles)
	lissajous(w, cycles)
}

//var palette = []color.Color{color.White, color.Black}
var palette = []color.Color{color.RGBA{0x00, 0xff, 0x00, 0xff}, color.RGBA{0xff, 0x00, 0x00, 0xff}}

const (
	backgroudIndex = 0 // first color in palette
	foregroudIndex = 1 // next color in palette
)

func lissajous(out io.Writer, cycles float64) {
	const (
		//cycles  = 5     // number of complete x oscillator revolutions
		res     = 0.001 // angular resolution
		size    = 100   // image canvas covers [-size..+size]
		nframes = 64    // number of animation frames
		delay   = 8     // delay between frames in 10ms units
	)

	freq := rand.Float64() * 3.0 // relative frequency of y oscillator
	anim := gif.GIF{LoopCount: nframes}
	phase := 0.0 // phase difference
	for i := 0; i < nframes; i++ {
		rect := image.Rect(0, 0, 2*size+1, 2*size+1)
		img := image.NewPaletted(rect, palette)
		for t := 0.0; t < cycles*2*math.Pi; t += res {
			x := math.Sin(t)
			y := math.Sin(t*freq + phase)
			img.SetColorIndex(size+int(x*size+0.5), size+int(y*size+0.5),
				foregroudIndex)
		}
		phase += 0.1
		anim.Delay = append(anim.Delay, delay)
		anim.Image = append(anim.Image, img)
	}
	gif.EncodeAll(out, &anim) // NOTE: ignoring encoding errors
}
```

### Switch

tag switch(tagless switch) ---- Go语言里的switch还可以不带操作对象（译注：switch不带操作对象时默认用true值代替，然后将每个case的表达式和true值进行比较）

## 程序结构

### 命名

关键字

```
break      default       func     interface   select
case       defer         go       map         struct
chan       else          goto     package     switch
const      fallthrough   if       range       type
continue   for           import   return      var
```

预先定义的名字并不是关键字，你可以在定义中重新使用它们

```
内建常量: true false iota nil

内建类型: int int8 int16 int32 int64
          uint uint8 uint16 uint32 uint64 uintptr
          float32 float64 complex128 complex64
          bool byte rune string error

内建函数: make len cap new append copy close delete
          complex real imag
          panic recover
```

在习惯上，Go语言程序员推荐使用 驼峰式 命名

### 声明

Go语言主要有四种类型的声明语句：var、const、type和func，分别对应变量、常量、类型和函数实体对象的声明。

### 变量

Go语言中不存在未初始化的变量。

在包级别声明的变量会在main入口函数执行前完成初始化（§2.6.2），局部变量将在声明语句被执行到的时候完成初始化。

变量的生命周期

- 包一级声明的变量 和整个程序的运行周期是一致的
- 局部变量的生命周期则是动态的：每次从创建一个新变量的声明语句开始，直到该变量不再被引用为止，然后变量的存储空间可能被回收。
    - 函数的参数变量和返回值变量都是局部变量。它们在函数每次被调用的时候创建。

垃圾回收

- 实现思路是，从每个包级的变量和每个当前运行函数的每一个局部变量开始，通过指针或引用的访问路径遍历，是否可以找到该变量。如果不存在这样的访问路径，那么说明该变量是不可达的，也就是说它是否存在并不会影响程序后续的计算结果。
- 因为一个变量的有效周期只取决于是否可达，因此一个循环迭代内部的局部变量的生命周期可能超出其局部作用域。同时，局部变量可能在函数返回之后依然存在。
- 编译器会自动选择在栈上还是在堆上分配局部变量的存储空间，但可能令人惊讶的是，这个选择并不是由用var还是new声明变量的方式决定的。
    ```go
    var global *int

    func f() {
        var x int
        x = 1
        global = &x
    }

    func g() {
        y := new(int)
        *y = 1
    }
    ```
- f函数里的x变量必须在堆上分配，因为它在函数退出后依然可以通过包一级的global变量找到，虽然它是在函数内部定义的；用Go语言的术语说，这个x局部变量从函数f中逃逸了。相反，当g函数返回时，变量`*y`将是不可达的，也就是说可以马上被回收的。因此，`*y`并没有从函数g中逃逸，编译器可以选择在栈上分配`*y`的存储空间（译注：也可以选择在堆上分配，然后由Go语言的GC回收这个变量的内存空间），虽然这里用的是new方式。

### 赋值

自增和自减是语句，而不是表达式，因此x = i++之类的表达式是错误的）

元组赋值
```
x, y = y, x
```

### 类型

```go
type Celsius float64    // 摄氏温度
type Fahrenheit float64 // 华氏温度
```
- Celsius和Fahrenheit分别对应不同的温度单位。它们虽然有着相同的底层类型float64，但是它们是不同的数据类型，因此它们不可以被相互比较或混在一个表达式运算。
- 因此需要一个类似Celsius(t)或Fahrenheit(t)形式的显式转型操作才能将float64转为对应的类型。

### 包和文件

因为汉字不区分大小写，因此汉字开头的名字是没有导出的

Import

- 按照惯例，一个包的名字和包的导入路径的最后一个字段相同，例如`gopl.io/ch2/tempconv`包的名字一般是tempconv。
- import别名
    ```
    import (
        sort "wuxu.bit/example/alg/sort"
        "fmt"
    )
    ```

golang在引入包的时候还有一些tricks：

- `import . "fmt"` 这样在调用fmt包的导出方法时可以省略fmt
- `import _ "fmt"` 这样引入该包但是不引入该包的导出函数，而是为了使用该导入操作的副作用: 调用包里面的init函数

包的初始化

- 包的初始化首先是解决包级变量的依赖顺序，然后按照包级变量声明出现的顺序依次初始化
- 初始化工作是自下而上进行的，main包最后被初始化。以这种方式，可以确保在main函数执行之前，所有依赖的包都已经完成初始化工作了。(包括init函数)
 
### 作用域



## 基础数据类型



### 整型



### 浮点数

### 复数

### 布尔型

### 字符串

### 常量